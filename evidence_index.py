#!/usr/bin/env python3
"""
Evidence index: what the RMG-database and earlier imports already know, in one file that
importChemkin.py can query in milliseconds, instead of loading hundreds of libraries into RMG.

Sources, all read-only:
    RMG-database thermo libraries     <database>/thermo/libraries/*.py
    RMG-database reaction libraries   <database>/kinetics/libraries/**/reactions.py (+ dictionary.txt)
    each imported model               <models>/**/RMG-Py-thermo-library/ThermoLibrary.py
                                      <models>/**/RMG-Py-kinetics-library/reactions.py (+ dictionary.txt)

The index is a SQLite file:
    sources     one row per library (or imported model), with a fingerprint of its files
    structures  one row per distinct structure, keyed by structure_key()
    names       the label each source gives a structure
    thermo      thermo entries, with Cp, H, S and G at the temperatures RMG compares them at
    reactions   library reactions as structure keys, plus a formula signature (the index used
                at import time) and a rate fingerprint (log10 k at 500, 1000 and 1500 K, 1 bar)

The importer looks up only the reactions that can still say something: those whose formula
signature matches a CHEMKIN reaction containing an unidentified species. Matching them is a
comparison of structure keys, with no RMG reaction generation. RMG's reaction families keep
generating new reactions and species as before; this adds a second, labelled source of votes.

Commands:
    python evidence_index.py build     --index FILE --database DIR --models DIR
    python evidence_index.py refresh   --index FILE --database DIR --models DIR
    python evidence_index.py stats     --index FILE
    python evidence_index.py benchmark --index FILE
`refresh` re-reads only libraries whose files changed, so it is cheap after each import.
"""
import argparse
import collections
import fcntl
import hashlib
import itertools
import json
import logging
import math
import os
import shutil
import sqlite3
import sys
import time

logger = logging.getLogger('evidence_index')

SCHEMA_VERSION = '1'

# The temperatures and tolerance of rmgpy.thermo.model.HeatCapacityModel.is_identical_to,
# so index matches agree with the importer's existing thermo-library matches
THERMO_TEMPERATURES = (300, 400, 500, 600, 800, 1000, 1500, 2000)
THERMO_TOLERANCE = 0.05
# Rate fingerprint: log10 k (SI units) at these temperatures, at 1 bar
RATE_TEMPERATURES = (500.0, 1000.0, 1500.0)
RATE_PRESSURE = 1e5
RATE_TOLERANCE = 0.01  # in log10 k, about 2%

SCHEMA = """
CREATE TABLE meta (name TEXT PRIMARY KEY, value TEXT);
CREATE TABLE sources (
    id INTEGER PRIMARY KEY, name TEXT UNIQUE NOT NULL, kind TEXT NOT NULL, path TEXT NOT NULL,
    fingerprint TEXT NOT NULL, structures INTEGER, thermo INTEGER, reactions INTEGER,
    skipped INTEGER, built_at TEXT);
CREATE TABLE structures (key TEXT PRIMARY KEY, formula TEXT NOT NULL, smiles TEXT, adjlist TEXT NOT NULL);
CREATE TABLE names (source_id INTEGER NOT NULL, label TEXT NOT NULL, key TEXT NOT NULL);
CREATE TABLE thermo (source_id INTEGER NOT NULL, label TEXT, key TEXT NOT NULL, formula TEXT NOT NULL,
                     thermo_values TEXT NOT NULL);
CREATE TABLE reactions (source_id INTEGER NOT NULL, label TEXT, signature TEXT NOT NULL,
                        reactants TEXT NOT NULL, products TEXT NOT NULL, rate TEXT);
CREATE INDEX structures_formula ON structures (formula);
CREATE INDEX names_source ON names (source_id);
CREATE INDEX thermo_formula ON thermo (formula);
CREATE INDEX thermo_source ON thermo (source_id);
CREATE INDEX reactions_signature ON reactions (signature);
CREATE INDEX reactions_source ON reactions (source_id);
"""


################################################################################
# Comparing structures, thermo and rates. Used by the builder and by the importer.

def structure_key(molecule):
    """
    A key for a structure that is the same for all its resonance forms but differs between
    spin states: the standard InChIKey plus the multiplicity. (Singlet and triplet CH2 share
    an InChIKey; the two resonance forms of allyl share one too, as they should.)
    """
    return '{0}-{1}'.format(molecule.to_inchi_key(), molecule.multiplicity)


def thermo_values(thermo):
    """Cp, H, S and G at THERMO_TEMPERATURES, in SI units, or None if any can't be evaluated."""
    values = []
    try:
        for T in THERMO_TEMPERATURES:
            values.extend((thermo.get_heat_capacity(T), thermo.get_enthalpy(T),
                           thermo.get_entropy(T), thermo.get_free_energy(T)))
    except Exception:
        return None
    return values if all(math.isfinite(v) for v in values) else None


def thermo_values_match(library_values, chemkin_values):
    """The comparison of HeatCapacityModel.is_identical_to(library, chemkin), on stored values."""
    if not library_values or not chemkin_values:
        return False
    for mine, theirs in zip(library_values, chemkin_values):
        if mine == theirs:
            continue
        if theirs == 0 or not (1 - THERMO_TOLERANCE < mine / theirs < 1 + THERMO_TOLERANCE):
            return False
    return True


def rate_fingerprint(kinetics):
    """log10 k at RATE_TEMPERATURES and RATE_PRESSURE, or None if it can't be evaluated."""
    values = []
    try:
        for T in RATE_TEMPERATURES:
            k = kinetics.get_rate_coefficient(T, RATE_PRESSURE)
            if not (k > 0 and math.isfinite(k)):
                return None
            values.append(round(math.log10(k), 4))
    except Exception:
        return None
    return values


def rates_match(a, b):
    return bool(a and b) and all(abs(x - y) <= RATE_TOLERANCE for x, y in zip(a, b))


def formula_signature(reactant_formulas, product_formulas):
    """'C2H4 + H = C2H5': what a CHEMKIN reaction must look like to match a library reaction."""
    return '{0} = {1}'.format(' + '.join(sorted(reactant_formulas)), ' + '.join(sorted(product_formulas)))


def side_assignments(chemkin_side, library_side):
    """
    Every way to pair each species on one side of a CHEMKIN reaction with a different species
    on the same side of a library reaction. chemkin_side is a list of (known structure key or
    None, formula); library_side a list of (structure key, formula). A known species must pair
    with the same structure, an unknown one with any structure of the same formula.
    Returns a set of tuples of library structure keys, one per CHEMKIN species.
    """
    assignments = set()
    if len(chemkin_side) != len(library_side):
        return assignments
    for permutation in itertools.permutations(library_side):
        if all(library_formula == formula and (known is None or known == key)
               for (known, formula), (key, library_formula) in zip(chemkin_side, permutation)):
            assignments.add(tuple(key for key, _ in permutation))
    return assignments


def match_library_reaction(chemkin_reactants, chemkin_products, library_reactants, library_products):
    """
    What a library reaction says about the unknown species of a CHEMKIN reaction, in either
    direction. Sides are lists as in side_assignments. Returns a list of (forward, reactant
    keys, product keys) for each consistent way of pairing the species.
    """
    found = []
    for forward, lib_r, lib_p in ((True, library_reactants, library_products),
                                  (False, library_products, library_reactants)):
        for r_keys in side_assignments(chemkin_reactants, lib_r):
            for p_keys in side_assignments(chemkin_products, lib_p):
                found.append((forward, r_keys, p_keys))
    return found


################################################################################
# Reading RMG library files

class LibraryFiles:
    """One source of evidence: an RMG-database library, or one imported model's libraries."""

    def __init__(self, name, kind, path, thermo_file=None, kinetics_dir=None):
        self.name, self.kind, self.path = name, kind, path
        self.thermo_file, self.kinetics_dir = thermo_file, kinetics_dir

    @property
    def files(self):
        files = [self.thermo_file] if self.thermo_file else []
        if self.kinetics_dir:
            files += [os.path.join(self.kinetics_dir, f) for f in ('reactions.py', 'dictionary.txt')]
        return [f for f in files if os.path.exists(f)]

    def fingerprint(self):
        digest = hashlib.sha1()
        for f in sorted(self.files):
            st = os.stat(f)
            digest.update('{0}:{1}:{2}\n'.format(f, st.st_size, int(st.st_mtime)).encode())
        return digest.hexdigest()


def find_sources(database_dir, models_dir):
    """All the libraries to index, as LibraryFiles."""
    sources = []
    if database_dir:
        thermo_dir = os.path.join(database_dir, 'thermo', 'libraries')
        for f in sorted(os.listdir(thermo_dir)) if os.path.isdir(thermo_dir) else []:
            if f.endswith('.py'):
                sources.append(LibraryFiles('rmg-database/thermo/' + f[:-3], 'rmg-database',
                                            os.path.join(thermo_dir, f), thermo_file=os.path.join(thermo_dir, f)))
        kinetics_dir = os.path.join(database_dir, 'kinetics', 'libraries')
        for root, dirs, files in sorted(os.walk(kinetics_dir)):
            if 'reactions.py' in files and 'dictionary.txt' in files:
                name = os.path.relpath(root, kinetics_dir)
                sources.append(LibraryFiles('rmg-database/kinetics/' + name, 'rmg-database', root, kinetics_dir=root))
    if models_dir:
        for root, dirs, files in sorted(os.walk(models_dir)):
            dirs.sort()
            if 'RMG-Py-thermo-library' in dirs or 'RMG-Py-kinetics-library' in dirs:
                thermo = os.path.join(root, 'RMG-Py-thermo-library', 'ThermoLibrary.py')
                kinetics = os.path.join(root, 'RMG-Py-kinetics-library')
                sources.append(LibraryFiles(
                    os.path.relpath(root, models_dir), 'imported', os.path.realpath(root),
                    thermo_file=thermo if os.path.exists(thermo) else None,
                    kinetics_dir=kinetics if os.path.exists(os.path.join(kinetics, 'reactions.py')) else None))
                dirs[:] = [d for d in dirs if not d.startswith('RMG-Py-')]
    return sources


class LibraryReader:
    """Reads RMG library files into plain values, caching structures by adjacency list."""

    def __init__(self):
        from rmgpy.data.thermo import ThermoDatabase
        from rmgpy.data.kinetics.database import KineticsDatabase
        from rmgpy.molecule import Molecule
        try:
            from rdkit import RDLogger
            RDLogger.DisableLog('rdApp.*')
        except ImportError:
            pass
        # Fail now, loudly, if RMG can't read structures (e.g. a broken build), rather than
        # skipping every structure and writing an empty index
        structure_key(Molecule().from_adjacency_list('multiplicity 2\n1 H u1 p0 c0'))
        self.Molecule = Molecule
        from rmgpy.data import reference  # some libraries cite their sources with Article(...) etc.
        references = {name: getattr(reference, name) for name in ('Reference', 'Article', 'Book', 'Thesis')
                      if hasattr(reference, name)}
        thermo, kinetics = ThermoDatabase(), KineticsDatabase()
        self.thermo_context = (dict(thermo.global_context), dict(thermo.local_context, **references))
        self.kinetics_context = (dict(kinetics.global_context), dict(kinetics.local_context, **references))
        self.structures = {}  # adjacency list text -> (key, formula, smiles, adjlist) or None

    def structure(self, adjlist):
        """(key, formula, smiles, adjlist) for an adjacency list, or None if RMG can't read it."""
        text = adjlist.strip()
        if text not in self.structures:
            try:
                molecule = self.Molecule().from_adjacency_list(text)
                self.structures[text] = (structure_key(molecule), molecule.get_formula(),
                                         molecule.to_smiles(), molecule.to_adjacency_list())
            except Exception:
                self.structures[text] = None
        return self.structures[text]

    @staticmethod
    def _entries(path, context):
        entries = []
        global_context, local_context = context
        local_context = dict(local_context)
        local_context['entry'] = lambda **kwargs: entries.append(kwargs)
        with open(path, encoding='utf-8', errors='replace') as f:
            exec(compile(f.read(), path, 'exec'), dict(global_context), local_context)  # trusted RMG data
        return entries

    def thermo_library(self, path):
        """[(label, structure, thermo values)] for a ThermoLibrary file."""
        out = []
        for entry in self._entries(path, self.thermo_context):
            structure = self.structure(entry.get('molecule') or '') if entry.get('molecule') else None
            values = thermo_values(entry['thermo']) if entry.get('thermo') is not None else None
            if structure and values:
                out.append((entry.get('label'), structure, values))
        return out

    def dictionary(self, path):
        """{label: structure} for a species dictionary file."""
        species, label, block = {}, None, []
        with open(path, encoding='utf-8', errors='replace') as f:
            lines = f.read().split('\n') + ['']
        for line in lines:
            if not line.strip():
                if label and block:
                    structure = self.structure('\n'.join(block))
                    if structure:
                        species[label] = structure
                label, block = None, []
            elif label is None:
                label = line.strip()
            elif not line.lstrip().startswith('//'):
                block.append(line)
        return species

    def kinetics_library(self, directory):
        """([(label, reactant structures, product structures, rate)], species dict, skipped)."""
        species = self.dictionary(os.path.join(directory, 'dictionary.txt'))
        out, skipped = [], 0
        for entry in self._entries(os.path.join(directory, 'reactions.py'), self.kinetics_context):
            label = entry.get('label') or ''
            for arrow in ('<=>', '=>', '='):
                if arrow in label:
                    sides = [[s.strip() for s in side.split(' + ')] for side in label.split(arrow, 1)]
                    break
            else:
                skipped += 1
                continue
            try:
                reactants = [species[s] for s in sides[0]]
                products = [species[s] for s in sides[1]]
            except KeyError:
                skipped += 1
                continue
            rate = rate_fingerprint(entry['kinetics']) if entry.get('kinetics') is not None else None
            out.append((label, reactants, products, rate))
        return out, species, skipped


################################################################################
# Building and refreshing the index

def _add_source(conn, reader, source):
    started = time.time()
    names, thermo_rows, reaction_rows, structures, skipped = [], [], [], {}, 0
    if source.thermo_file:
        try:
            for label, structure, values in reader.thermo_library(source.thermo_file):
                structures[structure[0]] = structure
                thermo_rows.append((label, structure[0], structure[1], json.dumps(values)))
                if label:
                    names.append((label, structure[0]))
        except Exception as e:
            logger.warning("Couldn't read %s: %s", source.thermo_file, e)
            skipped += 1
    if source.kinetics_dir:
        try:
            reactions, species, n_skipped = reader.kinetics_library(source.kinetics_dir)
            skipped += n_skipped
            for label, structure in species.items():
                structures[structure[0]] = structure
                names.append((label, structure[0]))
            for label, reactants, products, rate in reactions:
                reaction_rows.append((
                    label, formula_signature([s[1] for s in reactants], [s[1] for s in products]),
                    json.dumps([s[0] for s in reactants]), json.dumps([s[0] for s in products]),
                    json.dumps(rate) if rate else None))
        except Exception as e:
            logger.warning("Couldn't read %s: %s", source.kinetics_dir, e)
            skipped += 1
    cursor = conn.execute(
        "INSERT INTO sources (name, kind, path, fingerprint, structures, thermo, reactions, skipped, built_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))",
        (source.name, source.kind, source.path, source.fingerprint(), len(structures), len(thermo_rows),
         len(reaction_rows), skipped))
    source_id = cursor.lastrowid
    conn.executemany("INSERT OR IGNORE INTO structures (key, formula, smiles, adjlist) VALUES (?, ?, ?, ?)",
                     structures.values())
    conn.executemany("INSERT INTO names (source_id, label, key) VALUES (?, ?, ?)",
                     [(source_id, label, key) for label, key in set(names)])
    conn.executemany("INSERT INTO thermo (source_id, label, key, formula, thermo_values) VALUES (?, ?, ?, ?, ?)",
                     [(source_id,) + row for row in thermo_rows])
    conn.executemany("INSERT INTO reactions (source_id, label, signature, reactants, products, rate) "
                     "VALUES (?, ?, ?, ?, ?, ?)", [(source_id,) + row for row in reaction_rows])
    logger.info("Indexed %-60s %6d thermo, %6d reactions, %4d skipped (%.1f s)", source.name,
                len(thermo_rows), len(reaction_rows), skipped, time.time() - started)


def _delete_source(conn, source_id):
    for table in ('names', 'thermo', 'reactions'):
        conn.execute("DELETE FROM {0} WHERE source_id = ?".format(table), (source_id,))
    conn.execute("DELETE FROM sources WHERE id = ?", (source_id,))


def build(index_path, database_dir, models_dir, refresh=False):
    """
    Build the index (or, with refresh=True, re-read only the sources whose files changed).
    The new index is written to a temporary file and moved into place at the end, so an
    importer that is reading the index never sees it half-built.
    """
    started = time.time()
    lock = open(index_path + '.lock', 'w')
    fcntl.flock(lock, fcntl.LOCK_EX)  # one builder at a time
    try:
        temporary = index_path + '.building'
        if os.path.exists(temporary):
            os.remove(temporary)
        if refresh and os.path.exists(index_path):
            shutil.copyfile(index_path, temporary)
            conn = sqlite3.connect(temporary)
        else:
            conn = sqlite3.connect(temporary)
            conn.executescript(SCHEMA)
            conn.execute("INSERT INTO meta VALUES ('schema_version', ?)", (SCHEMA_VERSION,))
        existing = {name: (source_id, fingerprint) for source_id, name, fingerprint in
                    conn.execute("SELECT id, name, fingerprint FROM sources")}
        sources = find_sources(database_dir, models_dir)
        reader, changed = None, 0
        for name in set(existing) - {s.name for s in sources}:
            _delete_source(conn, existing[name][0])
            changed += 1
        for source in sources:
            previous = existing.get(source.name)  # (id, fingerprint) from the last build
            if previous:
                if previous[1] == source.fingerprint():
                    continue  # unchanged since the last build
                _delete_source(conn, previous[0])
            reader = reader or LibraryReader()
            _add_source(conn, reader, source)
            changed += 1
        conn.execute("DELETE FROM structures WHERE key NOT IN (SELECT key FROM names UNION SELECT key FROM thermo)")
        conn.execute("INSERT OR REPLACE INTO meta VALUES ('built_at', datetime('now'))")
        n_structures, = conn.execute("SELECT COUNT(*) FROM structures").fetchone()
        if sources and not n_structures:
            conn.close()
            os.remove(temporary)
            raise RuntimeError("No structures were indexed, so the index was not replaced. "
                               "Check that RMG can read adjacency lists in this environment.")
        conn.commit()
        conn.close()
        os.replace(temporary, index_path)
        logger.info("%s %s: %d of %d sources re-read in %.0f s, %d structures", 'Refreshed' if refresh else 'Built',
                    index_path, changed, len(sources), time.time() - started, n_structures)
        return changed
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


################################################################################
# Reading the index (what the importer uses)

class EvidenceIndex:
    """
    Read-only access to an evidence index. Sources whose path is in `exclude_paths` (the model
    being imported) are left out of every answer.
    """

    def __init__(self, path, exclude_paths=()):
        self.path = path
        self.conn = sqlite3.connect('file:{0}?mode=ro'.format(path), uri=True, check_same_thread=False)
        excluded = [os.path.realpath(p) for p in exclude_paths]
        self.source_names, self.excluded = {}, set()
        for source_id, name, kind, source_path in self.conn.execute("SELECT id, name, kind, path FROM sources"):
            self.source_names[source_id] = name
            # An imported model's name is its path inside RMG-models (e.g. PCI2013/527-Sheen), so
            # it is still recognised if RMG-models has moved, is reached through a link, or is a copy
            if any(os.path.realpath(source_path) == p or
                   (kind == 'imported' and (p + os.sep).endswith(os.sep + name + os.sep)) for p in excluded):
                self.excluded.add(source_id)
        self._formulas = {}

    def _rows(self, sql, values):
        values = list(values)
        for i in range(0, len(values), 500):
            chunk = values[i:i + 500]
            yield from self.conn.execute(sql.format(','.join('?' * len(chunk))), chunk)

    def counts(self):
        """{kind: number of sources}, and totals, for logging."""
        kinds = collections.Counter(kind for kind, in self.conn.execute("SELECT kind FROM sources"))
        totals = self.conn.execute("SELECT (SELECT COUNT(*) FROM thermo), (SELECT COUNT(*) FROM reactions), "
                                   "(SELECT COUNT(*) FROM structures)").fetchone()
        return dict(kinds), totals

    def thermo_for_formulas(self, formulas):
        """[(source name, label, key, formula, thermo values)] for thermo entries with these formulas."""
        return [(self.source_names[s], label, key, formula, json.loads(values))
                for s, label, key, formula, values in self._rows(
                    "SELECT source_id, label, key, formula, thermo_values FROM thermo WHERE formula IN ({0})",
                    set(formulas))
                if s not in self.excluded]

    def reactions_for_signatures(self, signatures):
        """[(source name, label, signature, reactant keys, product keys, rate)] with these signatures."""
        return [(self.source_names[s], label, sig, json.loads(r), json.loads(p), json.loads(rate) if rate else None)
                for s, label, sig, r, p, rate in self._rows(
                    "SELECT source_id, label, signature, reactants, products, rate FROM reactions "
                    "WHERE signature IN ({0})", set(signatures))
                if s not in self.excluded]

    def structures(self, keys):
        """{key: (formula, smiles, adjlist)}."""
        return {key: (formula, smiles, adjlist) for key, formula, smiles, adjlist in self._rows(
            "SELECT key, formula, smiles, adjlist FROM structures WHERE key IN ({0})", set(keys))}

    def formulas(self, keys):
        missing = set(keys) - set(self._formulas)
        if missing:
            for key, formula in self._rows("SELECT key, formula FROM structures WHERE key IN ({0})", missing):
                self._formulas[key] = formula
        return self._formulas


def library_votes(index, chemkin_reactions):
    """
    Votes from library reactions, for CHEMKIN reactions given as
        (reaction id, reactants, products, rate fingerprint or None)
    where each side is a list of (label, known structure key or None, formula). Only reactions
    with an unidentified species (known key None) can vote.
    Returns {label: {candidate key: {reaction id: [evidence]}}}, where each piece of evidence is
    a dict with the library reaction label, its sources, and whether the rate constants match.
    """
    wanted = {}
    for reaction_id, reactants, products, rate in chemkin_reactions:
        if all(known is not None for _, known, _ in reactants + products):
            continue
        forward = formula_signature([f for _, _, f in reactants], [f for _, _, f in products])
        backward = formula_signature([f for _, _, f in products], [f for _, _, f in reactants])
        wanted.setdefault(forward, []).append((reaction_id, reactants, products, rate))
        if backward != forward:
            wanted.setdefault(backward, []).append((reaction_id, reactants, products, rate))
    rows = index.reactions_for_signatures(wanted)
    formulas = index.formulas({k for row in rows for k in row[3] + row[4]})

    # one structural library reaction can come from many sources: group them
    grouped = collections.defaultdict(lambda: {'labels': [], 'sources': [], 'rates': []})
    for source, label, signature, reactant_keys, product_keys, rate in rows:
        group = grouped[(signature, tuple(sorted(reactant_keys)), tuple(sorted(product_keys)))]
        group['labels'].append(label)
        group['sources'].append(source)
        group['rates'].append(rate)

    votes = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(list)))
    for (signature, reactant_keys, product_keys), group in grouped.items():
        lib_r = [(k, formulas.get(k)) for k in reactant_keys]
        lib_p = [(k, formulas.get(k)) for k in product_keys]
        for reaction_id, reactants, products, chemkin_rate in wanted.get(signature, []):
            sides_r = [(known, formula) for _, known, formula in reactants]
            sides_p = [(known, formula) for _, known, formula in products]
            for forward, r_keys, p_keys in match_library_reaction(sides_r, sides_p, lib_r, lib_p):
                # rates are only comparable when the library reaction is written the same way round
                rate_match = forward and any(rates_match(rate, chemkin_rate) for rate in group['rates'])
                evidence = {'label': group['labels'][0], 'sources': sorted(set(group['sources'])),
                            'rate_match': rate_match}
                for (label, known, _), key in zip(reactants + products, r_keys + p_keys):
                    if known is None:
                        votes[label][key][reaction_id].append(evidence)
    return votes


################################################################################
# Benchmark: hide each structure of each imported model, and see what the other sources say

def benchmark(index_path, exclude_near_duplicates=False):
    """
    For every imported model and every structure in its reactions: pretend that one species is
    unidentified (all others known) and ask the other sources about the model's reactions that
    contain it. Coverage is the share of species that get a candidate; precision the share of
    those whose best-supported candidate is the right structure.
    """
    conn = sqlite3.connect('file:{0}?mode=ro'.format(index_path), uri=True)
    sources = {sid: (name, kind) for sid, name, kind in conn.execute("SELECT id, name, kind FROM sources")}
    formula_of = dict(conn.execute("SELECT key, formula FROM structures"))
    by_source = collections.defaultdict(list)
    by_signature = collections.defaultdict(list)
    for sid, signature, r, p in conn.execute("SELECT source_id, signature, reactants, products FROM reactions"):
        reaction = (signature, tuple(json.loads(r)), tuple(json.loads(p)))
        by_source[sid].append(reaction)
        by_signature[signature].append((sid, reaction[1], reaction[2]))
    structural = {sid: {(tuple(sorted(r)), tuple(sorted(p))) for _, r, p in reactions}
                  for sid, reactions in by_source.items()}

    def near_duplicates(sid):
        mine = structural[sid]
        return {other for other, theirs in structural.items()
                if other != sid and len(mine & theirs) > 0.5 * len(mine)}

    hidden = covered = correct = 0
    for sid, reactions in by_source.items():
        if sources[sid][1] != 'imported':
            continue
        skip = {sid} | (near_duplicates(sid) if exclude_near_duplicates else set())
        reactions_with = collections.defaultdict(list)
        for reaction in reactions:
            for k in set(reaction[1] + reaction[2]):
                reactions_with[k].append(reaction)
        for key, containing in reactions_with.items():
            support = collections.Counter()
            for signature, r, p in containing:
                ck_r = [(None if k == key else k, formula_of.get(k)) for k in r]
                ck_p = [(None if k == key else k, formula_of.get(k)) for k in p]
                sig_b = formula_signature([f for _, f in ck_p], [f for _, f in ck_r])
                candidates = set()
                for sig in {signature, sig_b}:
                    for other, lib_r, lib_p in by_signature.get(sig, []):
                        if other in skip:
                            continue
                        lr = [(k, formula_of.get(k)) for k in lib_r]
                        lp = [(k, formula_of.get(k)) for k in lib_p]
                        for _, r_keys, p_keys in match_library_reaction(ck_r, ck_p, lr, lp):
                            for (known, _), k in zip(ck_r + ck_p, r_keys + p_keys):
                                if known is None:
                                    candidates.add(k)
                support.update(candidates)
            hidden += 1
            if support:
                covered += 1
                best = support.most_common()
                if best[0][0] == key and (len(best) == 1 or best[1][1] < best[0][1]):
                    correct += 1
    return hidden, covered, correct


################################################################################

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=['build', 'refresh', 'stats', 'benchmark'])
    parser.add_argument('--index', required=True, help='the index file')
    parser.add_argument('--database', help='RMG-database input directory (contains thermo/ and kinetics/)')
    parser.add_argument('--models', help='the RMG-models directory with the imported models')
    args = parser.parse_args()
    logging.basicConfig(level=logging.ERROR, format='%(message)s')  # RMG logs a lot at INFO
    logger.setLevel(logging.INFO)

    if args.command in ('build', 'refresh'):
        if not args.database and not args.models:
            parser.error('give --database and/or --models')
        build(args.index, args.database, args.models, refresh=args.command == 'refresh')
    elif args.command == 'stats':
        index = EvidenceIndex(args.index)
        kinds, (thermo, reactions, structures) = index.counts()
        print('sources: ' + ', '.join('{0} {1}'.format(n, kind) for kind, n in sorted(kinds.items())))
        print('{0} structures, {1} thermo entries, {2} reactions, {3:.0f} MB'.format(
            structures, thermo, reactions, os.path.getsize(args.index) / 1e6))
    else:
        for exclude in (False, True):
            started = time.time()
            hidden, covered, correct = benchmark(args.index, exclude_near_duplicates=exclude)
            print('{0}: {1} hidden species, {2} ({3:.1%}) get a candidate, {4} ({5:.1%} of those) '
                  'have the right one on top ({6:.0f} s)'.format(
                      'without near-duplicate models' if exclude else 'all other sources',
                      hidden, covered, covered / max(hidden, 1), correct, correct / max(covered, 1),
                      time.time() - started))


if __name__ == '__main__':
    main()
