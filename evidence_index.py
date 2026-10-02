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

The importer uses it in two ways, both lookups with no RMG reaction generation:
  - Copied chemistry (copied_chemistry): most mechanisms reuse sub-mechanisms from earlier ones.
    A CHEMKIN reaction with the same formulas and rate constants as a library reaction was copied,
    so its species are the library's; copied thermo and shared names add to that. This needs
    nothing identified first, and the confident proposals can be confirmed a source at a time.
  - Library votes (library_votes): a CHEMKIN reaction with one unidentified species votes for a
    structure when a library has the same reaction with the same structures for the identified
    species. It votes only when corroborated (see MIN_SOURCES).
RMG's reaction families keep generating new reactions and species as before; they are still the
only way to find a structure that no library has.

Commands:
    python evidence_index.py build     --index FILE --database DIR --models DIR
    python evidence_index.py refresh   --index FILE --database DIR --models DIR
    python evidence_index.py stats     --index FILE
    python evidence_index.py benchmark --index FILE [--copied | --min_sources N]
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
# A library reaction votes only if it is corroborated: its rate constants match the CHEMKIN
# file's (the same reaction, copied from the same place), or at least this many sources have
# it. A single source with other rate constants is often a coincidence of formulas, or a
# mistake made in an earlier import. In the leave-one-model-out benchmark this trades a little
# coverage for precision: 95.7% of hidden species get a candidate instead of 96.7%, and the
# best candidate is right for 91.3% of those instead of 89.2% (without near-duplicate models:
# 76.2% instead of 81.3%, and 86.1% instead of 83.0%). A wrong vote costs more than a missing
# one, since RMG's reaction families still vote either way.
MIN_SOURCES = 2

# Copied chemistry (see copied_chemistry): most mechanisms reuse sub-mechanisms from earlier
# ones, with their rate constants, thermo and species names. Each copied reaction counts once,
# however many sources have it (52 sources with one rate expression are one piece of evidence).
# A proposal needs a score of CONFIDENT_SCORE and more than the next candidate's. In the
# leave-one-model-out benchmark (benchmark --copied), with nothing identified, that proposes
# 86.4% of species, 99.0% of them right (61.6% and 98.0% without near-duplicate models).
# Demanding twice the next score raised precision by under a point and cut proposals by 11-19.
COPIED_THERMO_TOLERANCE = 1e-3  # relative: the same polynomial, not merely similar thermo
THERMO_WEIGHT = 2               # copied thermo counts as much as two copied reactions
LABEL_WEIGHT = 2                # so does a source giving the structure the same name
CONFIDENT_SCORE = 3

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
        # for looking species up by name; added here so that indexes built before it get it too
        conn.execute("CREATE INDEX IF NOT EXISTS names_label ON names (label COLLATE NOCASE)")
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

    def names_for_labels(self, labels):
        """[(source name, label, key)] for the structures that a source gives one of these labels, ignoring case."""
        return [(self.source_names[s], label, key)
                for s, label, key in self._rows(
                    "SELECT source_id, label, key FROM names WHERE label COLLATE NOCASE IN ({0})", set(labels))
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


def corroborated(evidence, min_sources=MIN_SOURCES):
    """True if one CHEMKIN reaction's evidence for one candidate is enough to vote (see MIN_SOURCES)."""
    return (any(e['rate_match'] for e in evidence)
            or len({source for e in evidence for source in e['sources']}) >= min_sources)


def library_votes(index, chemkin_reactions, min_sources=MIN_SOURCES):
    """
    Votes from library reactions, for CHEMKIN reactions given as
        (reaction id, reactants, products, rate fingerprint or None)
    where each side is a list of (label, known structure key or None, formula). Only reactions
    with an unidentified species (known key None) can vote, and only with corroborated evidence.
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

    kept = {}
    for label, candidates in votes.items():
        for key, by_reaction in candidates.items():
            for reaction_id, evidence in by_reaction.items():
                if corroborated(evidence, min_sources):
                    kept.setdefault(label, {}).setdefault(key, {})[reaction_id] = evidence
    return kept


################################################################################
# Copied chemistry: what a new mechanism copied from earlier ones says about its species

def thermo_copied(a, b):
    """True if two lists of thermo values (see thermo_values) are the same within COPIED_THERMO_TOLERANCE."""
    return bool(a and b) and all(abs(x - y) <= COPIED_THERMO_TOLERANCE * max(abs(x), abs(y), 1.0)
                                 for x, y in zip(a, b))


def pair_by_formula(labels, keys, formula_of_label, formula_of_key):
    """
    Pair the species on one side of a CHEMKIN reaction with those on the same side of a library
    reaction that has the same formulas: [(label, key)] for every formula that is one species on
    both sides. Isomers on the same side can't be told apart, so they are left unpaired.
    """
    by_formula = collections.defaultdict(lambda: (set(), set()))
    for label in labels:
        by_formula[formula_of_label[label]][0].add(label)
    for key in keys:
        by_formula[formula_of_key.get(key)][1].add(key)
    return [(next(iter(ls)), next(iter(ks))) for ls, ks in by_formula.values() if len(ls) == 1 and len(ks) == 1]


class CopiedCandidate:
    """A structure an unidentified species may have, with the copied chemistry that says so."""

    def __init__(self, key):
        self.key = key
        self.reactions = {}  # CHEMKIN reaction id -> sources that have it with the same rate constants
        self.thermo = set()  # sources with the same thermo
        self.names = set()   # sources that give this structure the same name

    @property
    def score(self):
        """Each copied reaction counts once, plus THERMO_WEIGHT for copied thermo and LABEL_WEIGHT for the name."""
        return len(self.reactions) + THERMO_WEIGHT * bool(self.thermo) + LABEL_WEIGHT * bool(self.names)

    def source_weights(self):
        """How much of the evidence each source gave."""
        weights = collections.Counter()
        for sources in self.reactions.values():
            weights.update(sources)
        weights.update({source: THERMO_WEIGHT for source in self.thermo})
        for source in self.names:
            weights[source] += LABEL_WEIGHT
        return weights

    def main_source(self):
        """The source that gave the most evidence (the first by name if several gave as much)."""
        weights = self.source_weights()
        return min(weights, key=lambda source: (-weights[source], source)) if weights else None


def copied_chemistry(index, species, reactions, identified=None, blocked=()):
    """
    Candidates for the unidentified species, from the chemistry that the mechanism being imported
    has in common with the sources in the index. Nothing needs to be identified first.
        species     {label: (formula, thermo values or None)} for every CHEMKIN species
        reactions   [(reaction id, reactant labels, product labels, rate fingerprint or None)]
        identified  {label: structure key} for the species identified so far. They get no
                    candidates, their structures can't be another label's, and a library reaction
                    that pairs one of them with a different structure isn't counted.
        blocked     {(label, structure key)} for matches that were blocked
    A library reaction with the same formulas, written the same way round, and the same rate
    constants was copied, so its species pair up with the CHEMKIN reaction's by formula. Thermo
    entries with the same formula and values were copied too, and a source that gives a structure
    of the right formula the same name adds a little.
    Returns {label: [CopiedCandidate]} for the unidentified labels that have any, best first.
    """
    identified = identified or {}
    taken = set(identified.values())
    formula_of_label = {label: formula for label, (formula, _) in species.items()}
    candidates = collections.defaultdict(dict)

    def candidate(label, key):
        if label in identified or key in taken or (label, key) in blocked:
            return None
        if key not in candidates[label]:
            candidates[label][key] = CopiedCandidate(key)
        return candidates[label][key]

    wanted = collections.defaultdict(list)
    for reaction_id, reactants, products, rate in reactions:
        labels = list(reactants) + list(products)
        if not rate or any(label not in formula_of_label for label in labels) or all(l in identified for l in labels):
            continue
        signature = formula_signature([formula_of_label[l] for l in reactants], [formula_of_label[l] for l in products])
        wanted[signature].append((reaction_id, reactants, products, rate))
    rows = index.reactions_for_signatures(wanted)
    formula_of_key = index.formulas({key for row in rows for key in row[3] + row[4]})
    for source, _, signature, lib_reactants, lib_products, lib_rate in rows:
        for reaction_id, reactants, products, rate in wanted[signature]:
            if not rates_match(rate, lib_rate):
                continue
            pairs = (pair_by_formula(reactants, lib_reactants, formula_of_label, formula_of_key) +
                     pair_by_formula(products, lib_products, formula_of_label, formula_of_key))
            if any(label in identified and identified[label] != key for label, key in pairs):
                continue  # it pairs a species identified here with another structure
            for label, key in pairs:
                found = candidate(label, key)
                if found is not None:
                    found.reactions.setdefault(reaction_id, set()).add(source)

    unidentified = [label for label in species if label not in identified]
    by_formula = collections.defaultdict(list)
    for label in unidentified:
        if species[label][1]:
            by_formula[species[label][0]].append(label)
    for source, _, key, formula, values in index.thermo_for_formulas(by_formula):
        for label in by_formula.get(formula, ()):
            if thermo_copied(values, species[label][1]):
                found = candidate(label, key)
                if found is not None:
                    found.thermo.add(source)

    by_name = collections.defaultdict(list)
    for label in unidentified:
        by_name[label.upper()].append(label)
    named = index.names_for_labels(unidentified)
    formula_of_named = index.formulas({key for _, _, key in named})
    for source, name, key in named:
        for label in by_name.get(name.upper(), ()):
            if formula_of_named.get(key) == formula_of_label[label]:
                found = candidate(label, key)
                if found is not None:
                    found.names.add(source)

    return {label: sorted(found.values(), key=lambda c: (-c.score, c.key))
            for label, found in candidates.items() if found}


def confident_proposals(candidates):
    """
    The labels whose best candidate is clear enough to propose: a score of at least
    CONFIDENT_SCORE, more than the next candidate's, and not the best candidate of another label
    too (two labels for one structure need a person to look).
    Returns {label: CopiedCandidate}.
    """
    best_of = collections.Counter(found[0].key for found in candidates.values())
    proposals = {}
    for label, found in candidates.items():
        top = found[0]
        if (top.score >= CONFIDENT_SCORE and best_of[top.key] == 1 and
                (len(found) == 1 or top.score > found[1].score)):
            proposals[label] = top
    return proposals


################################################################################
# Benchmark: hide each structure of each imported model, and see what the other sources say

def benchmark(index_path, exclude_near_duplicates=False, min_sources=MIN_SOURCES):
    """
    For every imported model and every structure in its reactions: pretend that one species is
    unidentified (all others known) and ask the other sources about the model's reactions that
    contain it, counting only evidence that would vote in the importer (the model's own rate
    constants stand in for the CHEMKIN file's). Coverage is the share of species that get a
    candidate; precision the share of those whose best-supported candidate is the right structure.
    """
    conn = sqlite3.connect('file:{0}?mode=ro'.format(index_path), uri=True)
    sources = {sid: (name, kind) for sid, name, kind in conn.execute("SELECT id, name, kind FROM sources")}
    formula_of = dict(conn.execute("SELECT key, formula FROM structures"))
    by_source = collections.defaultdict(list)
    by_signature = collections.defaultdict(list)
    for sid, signature, r, p, rate in conn.execute(
            "SELECT source_id, signature, reactants, products, rate FROM reactions"):
        reaction = (signature, tuple(json.loads(r)), tuple(json.loads(p)), json.loads(rate) if rate else None)
        by_source[sid].append(reaction)
        by_signature[signature].append((sid,) + reaction[1:])
    structural = {sid: {(tuple(sorted(r)), tuple(sorted(p))) for _, r, p, _ in reactions}
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
            for signature, r, p, own_rate in containing:
                ck_r = [(None if k == key else k, formula_of.get(k)) for k in r]
                ck_p = [(None if k == key else k, formula_of.get(k)) for k in p]
                sig_b = formula_signature([f for _, f in ck_p], [f for _, f in ck_r])
                evidence = collections.defaultdict(list)  # candidate -> evidence, as in library_votes
                for sig in {signature, sig_b}:
                    for other, lib_r, lib_p, lib_rate in by_signature.get(sig, []):
                        if other in skip:
                            continue
                        lr = [(k, formula_of.get(k)) for k in lib_r]
                        lp = [(k, formula_of.get(k)) for k in lib_p]
                        for forward, r_keys, p_keys in match_library_reaction(ck_r, ck_p, lr, lp):
                            for (known, _), k in zip(ck_r + ck_p, r_keys + p_keys):
                                if known is None:
                                    evidence[k].append({'sources': [other],
                                                        'rate_match': forward and rates_match(lib_rate, own_rate)})
                support.update(k for k, e in evidence.items() if corroborated(e, min_sources))
            hidden += 1
            if support:
                covered += 1
                best = support.most_common()
                if best[0][0] == key and (len(best) == 1 or best[1][1] < best[0][1]):
                    correct += 1
    return hidden, covered, correct


def benchmark_copied(index_path, exclude_near_duplicates=False):
    """
    Leave-one-model-out for copied chemistry, with nothing identified. For every imported model,
    keep only what its CHEMKIN files give (formulas, rate constants, thermo and names), ask the
    other sources with copied_chemistry, and compare with the structures the model was given.
    Returns a Counter: species, candidate (has one), best_right (unique best is right),
    confident (proposed by confident_proposals) and confident_right.
    """
    index = EvidenceIndex(index_path)
    conn = index.conn
    kinds = dict(conn.execute("SELECT id, kind FROM sources"))
    formula_of = dict(conn.execute("SELECT key, formula FROM structures"))
    reactions = collections.defaultdict(list)
    for sid, r, p, rate in conn.execute("SELECT source_id, reactants, products, rate FROM reactions"):
        reactions[sid].append((json.loads(r), json.loads(p), json.loads(rate) if rate else None))
    thermo = collections.defaultdict(dict)
    for sid, key, values in conn.execute("SELECT source_id, key, thermo_values FROM thermo"):
        thermo[sid][key] = json.loads(values)
    names = collections.defaultdict(dict)
    for sid, label, key in conn.execute("SELECT source_id, label, key FROM names"):
        names[sid].setdefault(key, label)
    structural = {sid: {(tuple(sorted(r)), tuple(sorted(p))) for r, p, _ in rows} for sid, rows in reactions.items()}

    totals = collections.Counter()
    for sid in [s for s, kind in kinds.items() if kind == 'imported' and s in reactions]:
        mine = structural[sid]
        near = {o for o, theirs in structural.items() if o != sid and len(mine & theirs) > 0.5 * len(mine)}
        index.excluded = {sid} | (near if exclude_near_duplicates else set())
        label_of, used = {}, set()  # each hidden structure gets the model's name for it (unique), or its key
        for key in sorted({k for r, p, _ in reactions[sid] for k in r + p} | set(thermo[sid])):
            label = names[sid].get(key, key)
            label_of[key] = label if label not in used else key
            used.add(label_of[key])
        species = {label: (formula_of[key], thermo[sid].get(key)) for key, label in label_of.items()}
        found = copied_chemistry(index, species, [(i, [label_of[k] for k in r], [label_of[k] for k in p], rate)
                                                  for i, (r, p, rate) in enumerate(reactions[sid])])
        proposals = confident_proposals(found)
        for key, label in label_of.items():
            totals['species'] += 1
            candidates = found.get(label)
            if candidates:
                totals['candidate'] += 1
                totals['best_right'] += (candidates[0].key == key and
                                         (len(candidates) == 1 or candidates[1].score < candidates[0].score))
            if label in proposals:
                totals['confident'] += 1
                totals['confident_right'] += proposals[label].key == key
    return totals


################################################################################

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=['build', 'refresh', 'stats', 'benchmark'])
    parser.add_argument('--index', required=True, help='the index file')
    parser.add_argument('--database', help='RMG-database input directory (contains thermo/ and kinetics/)')
    parser.add_argument('--models', help='the RMG-models directory with the imported models')
    parser.add_argument('--min_sources', type=int, default=MIN_SOURCES,
                        help='benchmark: sources needed when the rate constants differ (1 counts all evidence)')
    parser.add_argument('--copied', action='store_true',
                        help='benchmark: copied chemistry, with nothing identified, instead of library votes')
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
    elif args.copied:
        for exclude in (False, True):
            started = time.time()
            t = benchmark_copied(args.index, exclude_near_duplicates=exclude)
            print('{0}: {1} species, {2:.1%} get a candidate, {3:.1%} of those have the right one on top; '
                  '{4} ({5:.1%}) confident proposals, {6:.2%} of them right ({7:.0f} s)'.format(
                      'without near-duplicate models' if exclude else 'all other sources',
                      t['species'], t['candidate'] / max(t['species'], 1), t['best_right'] / max(t['candidate'], 1),
                      t['confident'], t['confident'] / max(t['species'], 1),
                      t['confident_right'] / max(t['confident'], 1), time.time() - started))
    else:
        print('Counting evidence with matching rate constants, or from at least {0} source(s)'.format(
            args.min_sources))
        for exclude in (False, True):
            started = time.time()
            hidden, covered, correct = benchmark(args.index, exclude_near_duplicates=exclude,
                                                 min_sources=args.min_sources)
            print('{0}: {1} hidden species, {2} ({3:.1%}) get a candidate, {4} ({5:.1%} of those) '
                  'have the right one on top ({6:.0f} s)'.format(
                      'without near-duplicate models' if exclude else 'all other sources',
                      hidden, covered, covered / max(hidden, 1), correct, correct / max(covered, 1),
                      time.time() - started))


if __name__ == '__main__':
    main()
