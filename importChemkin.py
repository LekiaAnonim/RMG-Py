#!/usr/bin/env python
# encoding: utf-8
"""
This script enables the conversion of Chemkin files
into RMG-Py style thermo library file.
Simply pass the paths of the Chemkin files on the 
command-line, e.g.

    $ python importChemkin.py --species /path/to/chem1.inp  --thermo /path/to/therm.dat

If you supply a --species file (containing a SPECIES block) this is used to limit
the species converted.
The resulting file is saved next to the thermo input file.

If running in QTconsole, it draws pictures of the species.
"""
import os
import os.path
import argparse
import logging
import re
import codecs
import traceback

import hashlib
# NOTE: API implementation removed - using VoteLocalDB only
# Django dashboard syncs via SSH using incremental_sync.py

import numpy as np
import cherrypy
from cherrypy.lib.static import serve_file
import json
import threading
import urllib
import socket

import rmgpy
import rmgpy.rmg
import rmgpy.util
import rmgpy.rmg.input
from rmgpy.display import display
import rmgpy.kinetics
import rmgpy.chemkin
import rmgpy.data
import rmgpy.data.kinetics
from rmgpy.chemkin import load_chemkin_file, read_species_block, read_thermo_block, read_reactions_block, remove_comment_from_line
from rmgpy.rmg.model import ReactionModel
from rmgpy.reaction import Reaction

from rmgpy.thermo.thermoengine import generate_thermo_data
from rmgpy.data.thermo import Entry, save_entry
from rmgpy.data.base import Entry as kin_entry
from rmgpy.data.kinetics.common import save_entry as kin_save_entry
from rmgpy.exceptions import KineticsError
from rmgpy.molecule import Molecule
from rmgpy.rmg.model import Species  # you need this one, not the one in rmgpy.species!

from rmgpy.rmg.main import RMG, initialize_log
from rmgpy.molecule.draw import MoleculeDrawer

import time
import sys
# Put the RMG-database project at the start of the python path, so we use that import_old_database script!
database_dictionary = rmgpy.settings['database.directory']
database_project_directory = os.path.abspath(os.path.join(database_dictionary, '..'))
sys.path.insert(0, database_project_directory)

################################################################################


class ReactionPlaceholder:
    """
    Lightweight placeholder for RMG Reaction objects in vote reconstruction.
    
    When restoring votes from the database on restart, we don't need the full
    RMG Reaction object - we only need two things:
    1. The reaction family (str) - displayed in the votes table
    2. The string representation - displayed as the reaction equation
    
    This avoids the expensive reconstruction of full RMG Reaction objects
    which requires parsing species, generating resonance structures, etc.
    
    Usage:
        placeholder = ReactionPlaceholder("H + O2 <=> OH + O", "H_Abstraction")
        str(placeholder)      # Returns "H + O2 <=> OH + O"
        placeholder.family    # Returns "H_Abstraction"
    """
    
    def __init__(self, reaction_str: str, family: str = None):
        """
        Create a reaction placeholder.
        
        Args:
            reaction_str: The string representation of the reaction
            family: The reaction family name (e.g., "H_Abstraction")
        """
        self.reaction_str = reaction_str or ''
        self.family = family or '-'
    
    def __str__(self):
        """Return the reaction string for display"""
        return self.reaction_str
    
    def __repr__(self):
        return f"ReactionPlaceholder({self.reaction_str!r}, family={self.family!r})"
    
    def __hash__(self):
        """Allow use in sets (for vote reconstruction)"""
        return hash((self.reaction_str, self.family))
    
    def __eq__(self, other):
        """Equality check for set membership"""
        if isinstance(other, ReactionPlaceholder):
            return self.reaction_str == other.reaction_str and self.family == other.family
        return False


################################################################################

def make_or_empty_directory(path):
    """Either create a directory at `path` or delete everything in it if it exists"""
    if os.path.exists(path):
        assert os.path.isdir(path), "Path {0} exists but is not a directory".format(path)
        # empty it out
        for root, dirs, files in os.walk(path, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
    else:
        os.makedirs(path)

class MagicSpeciesDict(dict):
    """
    A dictionary that always has the species you're looking for!
    
    (If they key isn't there when you ask, it adds a blank Species() object and returns that)
    """

    def __init__(self, dictionary=None):
        self.dictionary = dictionary or {}

    def __getitem__(self, key):
        if key not in self.dictionary:
            self.dictionary[key] = Species()
        return dict.__getitem__(self.dictionary, key)


def convert_formula(formula_dict):
    """
    Given a formula in dict form {'c':2, 'h':6, 'o':0}
    return a canonical formula string "C2H6"
    
    For comparison reasons, this must be the same algorithm as used in
    rmgpy.molecule.Molecule class.
    """

#    elements = {e.capitalize(): n for e, n in formula_dict.items() if n > 0}
    elements = dict((e.capitalize(), n) for (e, n) in formula_dict.items() if n > 0)
    has_carbon = 'C' in elements
    has_hydrogen = 'H' in elements
    # Use the Hill system to generate the formula
    formula = ''
    # Carbon and hydrogen always come first if carbon is present
    if has_carbon:
        count = elements['C']
        formula += 'C{0:d}'.format(count) if count > 1 else 'C'
        del elements['C']
        if has_hydrogen:
            count = elements['H']
            formula += 'H{0:d}'.format(count) if count > 1 else 'H'
            del elements['H']
    # Other atoms are in alphabetical order
    # (This includes hydrogen if carbon is not present)
    keys = list(elements.keys())
    keys.sort()
    for key in keys:
        count = elements[key]
        formula += '{0}{1:d}'.format(key, count) if count > 1 else key
    return formula


def parse_command_line_arguments():

    parser = argparse.ArgumentParser(description="""
        Import a set of chemkin files, identifying the species therin.
        """)
    parser.add_argument(
        '--species',
        metavar='FILE',
        type=str,
        nargs='?',
        default=None,
        help='the Chemkin file containing the list of species')
    parser.add_argument(
        '--reactions',
        metavar='FILE',
        type=str,
        nargs='?',
        default=None,
        help='the Chemkin file containing the list of reactions')
    parser.add_argument(
        '--thermo',
        metavar='FILE',
        type=str,
        required=True,
        help='the Chemkin files containing the thermo')
    parser.add_argument(
        '--known',
        metavar='FILE',
        type=str,
        nargs='?',
        default='SMILES.txt',
        help='the file containing the list of already known species')
    parser.add_argument(
        '--port',
        metavar='N',
        type=int,
        nargs='?',
        default=8080,
        help='the port to serve the web interface on')
    parser.add_argument(
        '--quit_when_exhausted',
        action='store_true',
        help=("Don't wait for input from the web front end, "
              "but quit when exhausted all matches. "
              "Can also be enabled by setting the environment "
              "variable RMG_QUIT_WHEN_EXHAUSTED"))
    parser.add_argument(
        '--minimal',
        action='store_true',
        help=("Just do a minimal read of the chemkin files to detect errors, then quit.")
    )
    parser.add_argument(
        '-o', '--output-directory',
        type=str,
        nargs='?',
        default='',
        metavar='DIR',
        help='use DIR as output directory')
    parser.add_argument(
        '-s', '--scratch-directory',
        type=str,
        nargs='?',
        default='',
        metavar='DIR',
        help='use DIR as scratch directory')
    parser.add_argument(
        '--pdep',
        action='store_true',
        help='run pressure-dependence calculations')
    parser.add_argument(
        '--mopac',
        action='store_true',
        help='do run QM thermo calculations using MOPAC for cyclic species')
    parser.add_argument(
        '--noqm',
        action='store_true',
        help="Deprecated. See --mopac argument.",
    )

    # Options for controlling the amount of information printed to the console
    # By default a moderate level of information is printed; you can either
    # ask for less (quiet), more (verbose), or much more (debug)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-q', '--quiet', action='store_true', help='only print warnings and errors')
    group.add_argument('-v', '--verbose', action='store_true', help='print more verbose output')
    group.add_argument('-d', '--debug', action='store_true', help='print debug information')

    args = parser.parse_args()
    
    if args.reactions is None:
        logging.warning("Using thermo file for reactions, as no reactions file specified.")
        args.reactions = args.thermo
    if args.species is None:
        logging.warning("Using thermo file for species, as no species file specified.")
        args.species = args.thermo

    # add some args that RMG will want

    args.walltime = '0'
    args.restart = False

    input_directory = os.path.abspath(os.path.dirname(args.thermo))
    if args.output_directory == '':
        args.output_directory = os.path.join(input_directory, 'RMG-Py-output')
    if args.scratch_directory == '':
        args.scratch_directory = args.output_directory

    if 'RMG_QUIT_WHEN_EXHAUSTED' in os.environ:
        logging.warning("Setting --quit_when_exhausted option because RMG_QUIT_WHEN_EXHAUSTED environment variable detected.")
        args.quit_when_exhausted = True

    return args


def reaction_matches_formulas(reaction, reactant_formulas):
    """
    Does the reaction match the reactant_formulas, in either direction?
    
    The reaction must contain real species with identified molecules.
    reactant_formulas is an iterable of Strings like ('CH3','H').
    Returns 'forward', 'backward', or False.
    """
    reactant_formulas = sorted(reactant_formulas)
    if reactant_formulas == sorted([s.get_formula() for s in reaction.reactants]):
        return 'forward'
    elif reactant_formulas == sorted([s.get_formula() for s in reaction.products]):
        return 'backward'
    else:
        return False

def get_relative_referer(default="votes2.html"):
    """
    Get the referer URL as a relative path (just filename + query).
    Avoids OOD proxy path duplication issues.
    """
    import os
    from urllib.parse import urlparse
    
    referer = cherrypy.request.headers.get("Referer", "")
    if not referer:
        return default
    
    parsed = urlparse(referer)
    # Get just the filename from the path
    filename = os.path.basename(parsed.path) or default
    # Preserve query string if present
    if parsed.query:
        return f"{filename}?{parsed.query}"
    return filename

class ModelMatcher():
    """
    For identifying species in an imported model
    """

    def __init__(self, args=None):
        self.args = args
        self.species_dict = {}
        self.thermo_dict = {}
        self.formula_dict = {}
        self.smiles_dict = {}
        self.identified_labels = []
        self.identified_unprocessed_labels = []
        self.identified_by = {}
        """Which user identified which species: self.identified_by[chemkin_labels] = username"""
        self.species_list = None
        self.species_dict_rmg = {}
        self.chemkin_reactions = []
        self.chemkin_reactions_unmatched = []
        "Reactions that contain species that haven't been identified yet."
        #self.chemkin_reactions_unmatchable = []
        #"Reactions that did not match an RMG-generated reaction, despite all the species being known"
        self.chemkin_reactions_to_save = []
        """A list of chemkin reactions that have been fully identified but not yet saved to a file.
        (because the reagents may not have been properly processed yet we have to defer the saving)
        """
        self.chemkin_reactions_dict = {}
        """A dictionary such that self.chemkin_reactions_dict[chemkin_label] = {set of chemkin reactions it is part of}"""
        self.suggested_matches = {}
        self.votes = {}
        """
        self.votes is a dict of dicts of lists of tuples: {'ch3':{<Species CH3>: [ voting_reactions ]}}
        so self.votes[chemkin_label][rmg_species][0] = (chemkin_reaction, rmg_reaction)
        """
        self.pruned_votes = {}
        self.manual_matches_to_process = []
        """A list of tuples of matches not yet processed: [(chemkin_label, rmg_species),...]"""
        self.tentative_matches = []
        self.thermo_matches = {}
        self.thermo_libraries_to_check = []
        self.blocked_matches = {}
        """A dictionary of matches forbidden manually. blocked_matches[ck_label][rmg Species] = username (or None)"""
        self.already_processed_labels = set()
        """A set of chemkin labels that have already been fully processed (limit_enlarge completed).
        On restart, these species are skipped to avoid expensive re-computation."""
        self.known_species_file = ""
        """Filename of the known species file"""
        self.blocked_matches_file = ""
        """Filename of the blocked matches file"""
        self.thermo_library = None
        """An rmgpy.data.thermo.ThermoLibrary object containing identified species and their chemkin-defined thermo"""
        self.kinetics_library = None
        """An rmgpy.data.kinetics.KineticsLibrary object containing chemkin-defined reactions of identified species"""


        # Determine what to call this model, based on its path
        location = os.path.abspath(self.args.reactions or self.args.species)
        name = os.path.split(location)[0]
        try:
            name = name[(name.index('RMG-models') + 11):]
        except ValueError:
            pass
        self.name = name
        "The name of the model (based on its source path)"

        # Generate unique job ID from input files
        job_identifier = f"{args.species or ''}{args.reactions or ''}{args.thermo or ''}"
        self.job_id = hashlib.md5(job_identifier.encode()).hexdigest()
        
        # Initialize local SQLite database for vote storage
        # Django dashboard will sync via SSH + incremental queries
        # Put database in the model directory (same as species/reactions files)
        model_dir = os.path.dirname(os.path.abspath(self.args.reactions or self.args.species))
        default_db_path = os.path.join(model_dir, f'votes_{self.job_id}.db')
        db_path = os.environ.get('VOTE_DB_PATH', default_db_path)
        
        try:
            from vote_local_db import VoteLocalDB
            self.vote_db = VoteLocalDB(db_path=db_path)
            
            # Create or get job record
            self.import_job = self.vote_db.create_or_get_job(
                job_id=self.job_id,
                model_name=self.name,
                species_file=args.species or '',
                reactions_file=args.reactions or '',
                thermo_file=args.thermo or ''
            )
            
            logging.info(f"Vote Storage Initialized:")
            logging.info(f"  Job ID: {self.job_id}")
            logging.info(f"  Local DB: {db_path}")
            logging.info(f"  Django dashboard will sync via SSH")
            
            self.vote_db_enabled = True
            
        except Exception as e:
            logging.error(f"Could not initialize vote database: {e}")
            logging.warning("Continuing without vote persistence")
            self.vote_db = None
            self.vote_db_enabled = False

    def load_votes_from_storage(self):
        """
        Load previously saved data from local database.
        
        Strategy:
        - Load identified species list (to skip re-processing them)
        - Load blocked matches (prevents re-suggesting bad matches)
        - Reconstruct votes dictionary from stored data (AVOIDS expensive re-computation!)
        - Log statistics for debugging
        """
        if not self.vote_db_enabled:
            logging.info("Vote database not enabled, skipping vote loading")
            return
        
        try:
            logging.info("Loading data from local database...")

            # Utilities for reconstructing RMG Species from stored SMILES
            from rmgpy.molecule import Molecule
            
            # 1. Load identified species - CRITICAL for resuming
            identified_from_db = self.vote_db.load_identified_species_full(self.job_id)
            if identified_from_db:
                restored_count = 0
                for species_data in identified_from_db:
                    chemkin_label = species_data.get('chemkin_label')
                    if not chemkin_label or chemkin_label in self.identified_labels:
                        continue

                    # Reconstruct RMG Species from stored SMILES and add to species_dict_rmg
                    rmg_smiles = species_data.get('rmg_species_smiles') or species_data.get('smiles')
                    if rmg_smiles and chemkin_label not in self.species_dict_rmg:
                        try:
                            molecule = Molecule(smiles=rmg_smiles)
                            rmg_species, was_new = self.rmg_object.reaction_model.make_new_species(molecule)
                            if was_new:
                                rmg_species.generate_resonance_structures()
                                rmg_species.thermo = generate_thermo_data(rmg_species)
                            # Ensure label aligns with chemkin label for downstream lookups
                            rmg_species.label = chemkin_label
                            self.species_dict_rmg[chemkin_label] = rmg_species
                            self.identified_labels.append(chemkin_label)
                            restored_count += 1
                            
                            # Restore identified_by info
                            identified_by = species_data.get('identified_by', 'Auto')
                            if identified_by:
                                self.identified_by[chemkin_label] = identified_by
                        except Exception as e:
                            logging.warning(f"Could not reconstruct species {chemkin_label} from SMILES: {e}")
                            
                logging.info(f"✓ Restored {len(identified_from_db)} identified species from database")
            
            # 2. Load blocked matches - prevents re-suggesting bad matches
            blocked_from_db = self.vote_db.load_blocked_matches(self.job_id)
            for blocked in blocked_from_db:
                chemkin_label = blocked['chemkin_label']
                rmg_smiles = blocked.get('rmg_species_smiles', '')
                blocked_by = blocked.get('blocked_by')
                
                if not rmg_smiles:
                    continue
                
                try:
                    from rmgpy.molecule import Molecule
                    molecule = Molecule(smiles=rmg_smiles)
                    rmg_species, was_new = self.rmg_object.reaction_model.make_new_species(molecule)
                    
                    if was_new:
                        rmg_species.generate_resonance_structures()
                        from rmgpy.thermo.thermoengine import generate_thermo_data
                        rmg_species.thermo = generate_thermo_data(rmg_species)
                    
                    if chemkin_label not in self.blocked_matches:
                        self.blocked_matches[chemkin_label] = {}
                    self.blocked_matches[chemkin_label][rmg_species] = blocked_by
                    
                except Exception as e:
                    logging.warning(f"Could not restore blocked match {chemkin_label}: {e}")
            
            if blocked_from_db:
                logging.info(f"✓ Restored {len(blocked_from_db)} blocked matches from database")
            
            # 3. RECONSTRUCT VOTES - This is the key optimization!
            reconstructed_count = self._reconstruct_votes_from_db()
            if reconstructed_count > 0:
                logging.info(f"✓ Reconstructed {reconstructed_count} votes from database (skipping expensive re-computation!)")
                logging.info("Vote source: local DB reconstruction (no full recompute)")
            else:
                logging.info("Vote source: no DB votes reconstructed; will recompute from reactions")
            
            # 4. Load PROCESSED species - KEY for skipping limit_enlarge() on restart!
            # Species marked as processed already had limit_enlarge() run,
            # so we can skip that expensive computation on restart.
            self.already_processed_labels = set(self.vote_db.load_processed_species(self.job_id))
            if self.already_processed_labels:
                logging.info(f"✓ Found {len(self.already_processed_labels)} already-processed species")
                logging.info(f"  These will SKIP limit_enlarge() on restart (major speedup!)")
                
                # Log processing status
                status = self.vote_db.get_processing_status(self.job_id)
                logging.info(f"  Processing status: {status['total_processed']}/{status['total_identified']} processed")
            else:
                self.already_processed_labels = set()
                logging.info("No previously processed species found (fresh start or first run)")
            
            # 5. Log overall statistics
            stats = self.vote_db.get_statistics(self.job_id)
            logging.info(f"Database statistics: {stats.get('total_votes', 0)} votes, "
                        f"{stats.get('identified_species', 0)} identified species, "
                        f"{stats.get('blocked_matches', 0)} blocked matches")
            
            logging.info("✓ Vote database loading completed - resuming from previous state")
            
        except Exception as e:
            logging.error(f"Error loading from database: {e}")
            import traceback
            traceback.print_exc()

    def _reconstruct_votes_from_db(self):
        """
        Reconstruct self.votes dictionary from database using ReactionPlaceholder.
        
        This converts stored vote data back into the format:
        self.votes[chemkin_label][rmg_species] = set([(chemkin_rxn_str, ReactionPlaceholder), ...])
        
        OPTIMIZATION: Instead of reconstructing full RMG Reaction objects (expensive),
        we use ReactionPlaceholder objects that only store:
        - The reaction string (for display)
        - The reaction family (for display in votes table)
        
        This is sufficient because the only usage of these reconstructed reactions is:
        1. Display in votes2_html: r.family and str(r)
        2. Display in votes_html: rxn[1].family
        
        Returns number of vote entries reconstructed.
        """
        if not self.vote_db_enabled or not hasattr(self, 'rmg_object'):
            return 0
        
        try:
            from rmgpy.molecule import Molecule
            
            votes_data = self.vote_db.load_votes_for_reconstruction(self.job_id)
            if not votes_data:
                logging.info("No votes found in database to reconstruct")
                return 0
            
            reconstructed = 0
            skipped_identified = 0
            skipped_errors = 0
            
            # Cache for SMILES -> Species lookup to avoid recreating same species
            smiles_to_species = {}
            
            for vote in votes_data:
                chemkin_label = vote['chemkin_label']
                rmg_smiles = vote['rmg_species_smiles']
                vote_count = vote['vote_count']
                
                # Skip if this species is already identified
                if chemkin_label in self.identified_labels:
                    skipped_identified += 1
                    continue
                
                if not rmg_smiles:
                    skipped_errors += 1
                    continue
                
                try:
                    # Get or create RMG Species from SMILES
                    # (We still need the Species object as the dict key)
                    if rmg_smiles in smiles_to_species:
                        rmg_species = smiles_to_species[rmg_smiles]
                    else:
                        molecule = Molecule(smiles=rmg_smiles)
                        rmg_species, was_new = self.rmg_object.reaction_model.make_new_species(molecule)
                        if was_new:
                            rmg_species.generate_resonance_structures()
                            rmg_species.thermo = generate_thermo_data(rmg_species)
                        smiles_to_species[rmg_smiles] = rmg_species
                    
                    # Initialize votes structure if needed
                    if chemkin_label not in self.votes:
                        self.votes[chemkin_label] = {}
                    if rmg_species not in self.votes[chemkin_label]:
                        self.votes[chemkin_label][rmg_species] = set()
                    
                    # Reconstruct reaction tuples using ReactionPlaceholder (LIGHTWEIGHT!)
                    voting_reactions = vote.get('voting_reactions', [])
                    for rxn_data in voting_reactions:
                        chemkin_reaction_str = rxn_data.get('chemkin_reaction_str', '')
                        edge_reaction_str = rxn_data.get('edge_reaction_str', '')
                        reaction_family = rxn_data.get('reaction_family', '-')
                        
                        # Use ReactionPlaceholder instead of full RMG Reaction
                        # This is MUCH faster - no parsing, no resonance, no thermo!
                        placeholder = ReactionPlaceholder(edge_reaction_str, reaction_family)
                        
                        self.votes[chemkin_label][rmg_species].add((chemkin_reaction_str, placeholder))
                    
                    # If no reactions stored but vote_count > 0, create generic placeholders
                    if not voting_reactions and vote_count > 0:
                        for i in range(vote_count):
                            placeholder = ReactionPlaceholder(f'restored_vote_{i}', '-')
                            self.votes[chemkin_label][rmg_species].add((f'restored_vote_{i}', placeholder))
                    
                    reconstructed += 1
                    logging.debug(f"Reconstructed {len(voting_reactions)} votes for {chemkin_label} -> {rmg_species}")
                    
                except Exception as e:
                    logging.debug(f"Could not reconstruct vote for {chemkin_label}/{rmg_smiles}: {e}")
                    skipped_errors += 1
            
            if skipped_identified > 0:
                logging.info(f"Skipped {skipped_identified} votes for already-identified species")
            if skipped_errors > 0:
                logging.debug(f"Skipped {skipped_errors} votes due to errors")
            
            logging.info(f"✓ Reconstructed {reconstructed} vote entries using ReactionPlaceholder (fast mode)")
            return reconstructed
            
        except Exception as e:
            logging.warning(f"Vote reconstruction failed: {e}")
            import traceback
            traceback.print_exc()
            return 0

    def _normalize_reaction_label(self, label: str) -> str:
        """Normalize a species label parsed from a reaction string."""
        if not label:
            return ''
        label = label.strip()
        # Remove collider notation
        label = label.replace('(+M)', '').replace('(+m)', '')
        label = label.replace('+M', '').replace('+m', '')
        # Remove leading stoichiometric coefficient
        label = re.sub(r'^\s*\d+\s*', '', label)
        # Remove RMG index suffix like H2O(1)
        match = re.match(r'^(.*)\(\d+\)$', label)
        if match:
            label = match.group(1)
        return label.strip()

    def _split_reaction_side(self, side: str) -> list:
        """Split one side of a reaction string into normalized labels."""
        if not side:
            return []
        side = side.replace('(+M)', '').replace('(+m)', '')
        tokens = [self._normalize_reaction_label(token) for token in side.split('+')]
        return [token for token in tokens if token]

    def _get_species_from_label(self, label: str, smiles_to_species: dict):
        """Get or create an RMG Species object for a label using stored SMILES."""
        if label in self.species_dict_rmg:
            return self.species_dict_rmg[label]

        smiles = self.smiles_dict.get(label)
        if not smiles:
            return None

        if smiles in smiles_to_species:
            species = smiles_to_species[smiles]
        else:
            molecule = Molecule(smiles=smiles)
            species, was_new = self.rmg_object.reaction_model.make_new_species(molecule)
            if was_new:
                species.generate_resonance_structures()
                species.thermo = generate_thermo_data(species)
            smiles_to_species[smiles] = species

        species.label = label
        self.species_dict_rmg[label] = species
        return species

    def _build_reaction_from_string(self, reaction_str: str, smiles_to_species: dict = None):
        """Build an RMG Reaction object from a reaction string when possible."""
        if not reaction_str:
            return None

        if smiles_to_species is None:
            smiles_to_species = {}

        reversible = '<=>' in reaction_str
        parts = reaction_str.split('<=>') if reversible else reaction_str.split('=>')
        if len(parts) != 2:
            return None

        reactant_labels = self._split_reaction_side(parts[0])
        product_labels = self._split_reaction_side(parts[1])
        if not reactant_labels or not product_labels:
            return None

        # Reuse cache from vote reconstruction to avoid duplicated species creation
        # smiles_to_species = {}

        reactants = []
        products = []
        for label in reactant_labels:
            species = self._get_species_from_label(label, smiles_to_species)
            if species is None:
                return None
            reactants.append(species)

        for label in product_labels:
            species = self._get_species_from_label(label, smiles_to_species)
            if species is None:
                return None
            products.append(species)

        return Reaction(reactants=reactants, products=products, reversible=reversible)

    def save_votes_to_storage(self, force=False):
        """
        Save votes to local SQLite database
        
        Django dashboard syncs via SSH + incremental queries (no API needed)
        
        Args:
            force (bool): If True, save regardless of whether votes changed.
                         If False, only save if votes were added since last save.
        
        Returns:
            dict: Result of save operation
        """
        if not self.vote_db_enabled:
            return {'success': False, 'message': 'Vote database not enabled'}

        logging.info(f"DEBUG save_votes_to_storage: {len(self.votes)} species have votes")
        total_vote_count = sum(len(candidates) for candidates in self.votes.values())
        logging.info(f"DEBUG save_votes_to_storage: {total_vote_count} total candidate matches")
        if self.votes:
            sample_label = list(self.votes.keys())[0]
            logging.info(f"DEBUG: First species with votes: {sample_label} has {len(self.votes[sample_label])} candidates")
        
        # Check if there are new votes to save
        current_vote_count = len(self.votes)
        if not force and current_vote_count == getattr(self, 'last_saved_vote_count', 0):
            return {'success': True, 'message': 'No new votes to save', 'skipped': True}
        
        try:
            # Save votes to local database with enthalpy calculator
            saved_count = self.vote_db.save_votes(
                self.job_id, 
                self.votes,
                enthalpy_calculator=self.get_enthalpy_discrepancy
            )
            
            # Save thermo matches to local database (NEW)
            thermo_saved_count = 0
            if self.thermo_matches:
                thermo_saved_count = self.vote_db.save_thermo_matches(self.job_id, self.thermo_matches)
                logging.info(f"✓ Saved {thermo_saved_count} thermo matches to local database")
            
            # Update last saved count
            self.last_saved_vote_count = current_vote_count
            
            logging.info(f"✓ Saved {saved_count} votes to local database")
            
            return {
                'success': True,
                'local': saved_count,
                'thermo_matches': thermo_saved_count,
                'message': f'Saved {saved_count} votes and {thermo_saved_count} thermo matches locally'
            }
            
        except Exception as e:
            logging.error(f"Error saving votes to storage: {e}")
            return {'success': False, 'message': str(e)}

    def export_votes_to_json(self):
        """Export votes to a JSON backup file for recovery"""
        if not self.vote_db_enabled or not self.votes:
            return
        
        try:
            import json
            from datetime import datetime
            
            backup_file = os.path.join(
                self.output_path, 
                f'votes_backup_{self.job_id}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
            )
            
            # Convert votes to JSON-serializable format
            votes_export = {}
            for chemkin_label, candidates in self.votes.items():
                votes_export[chemkin_label] = {}
                for rmg_species, reactions in candidates.items():
                    # Store species as SMILES string
                    species_key = rmg_species.molecule[0].to_smiles() if hasattr(rmg_species, 'molecule') else str(rmg_species)
                    votes_export[chemkin_label][species_key] = {
                        'vote_count': len(reactions),
                        'species_label': getattr(rmg_species, 'label', 'unknown'),
                        'reactions': [
                            {
                                'chemkin': str(rxn[0]) if len(rxn) > 0 else 'unknown',
                                'rmg': str(rxn[1]) if len(rxn) > 1 else 'unknown'
                            }
                            for rxn in list(reactions)[:10]  # Limit to first 10 reactions
                        ]
                    }
            
            with open(backup_file, 'w') as f:
                json.dump({
                    'job_id': self.job_id,
                    'timestamp': datetime.now().isoformat(),
                    'total_species_with_votes': len(votes_export),
                    'identified_species': len(self.identified_labels),
                    'votes': votes_export
                }, f, indent=2)
            
            logging.info(f"✓ Exported votes backup to: {backup_file}")
            
        except Exception as e:
            logging.error(f"Error exporting votes to JSON: {e}")

    def show_sync_status(self):
        """Display current vote synchronization status"""
        if not self.vote_db_enabled:
            return
        
        try:
            vote_count = len(self.votes)
            identified_count = len(self.identified_labels)
            
            logging.info("=== Vote Storage Status ===")
            logging.info(f"Job ID: {self.job_id}")
            logging.info(f"Local votes: {vote_count}")
            logging.info(f"Identified species: {identified_count}")
            logging.info(f"Database: {self.vote_db.db_path}")
            logging.info("Note: Django dashboard syncs via SSH")
        except Exception as e:
            logging.error(f"Error displaying sync status: {e}")

    def load_species(self, species_file):
        """
        Load the chemkin list of species
        """
        species_aliases = {}
        species_dict = {}
        if species_file:
            logging.info("Reading species list...")
            species_list = []
            with open(species_file, encoding='utf-8', errors='ignore') as f:
                position_before_line = f.tell()
                line0 = f.readline()
                while line0 != '':
                    line = remove_comment_from_line(line0)[0]
                    tokens_upper = line.upper().split()
                    if tokens_upper and tokens_upper[0] in ('SPECIES', 'SPEC'):
                        # We've found the start of the SPECIES block
                        # Unread the line (we'll re-read it in read_reaction_block())
                        f.seek(position_before_line)
                        read_species_block(f, species_dict, species_aliases, species_list)
                        # end of species block. Assume there's only one!
                        break
                    position_before_line = f.tell()
                    line0 = f.readline()
        else:
            logging.info("No species file to limit species. Will read everything in thermo file")
            species_list = None
            species_dict = MagicSpeciesDict(species_dict)
        self.species_list = species_list
        self.species_dict = species_dict

    def load_thermo(self, thermo_file):
        """
        Load the chemkin thermochemistry file
        """
        logging.info("Reading thermo file...")
        species_dict = self.species_dict
        found_thermo_block = False
        #import codecs
        #with codecs.open(thermo_file, "r", "utf-8") as f:
        with open(thermo_file, 'r', encoding='utf-8', errors='ignore') as f:
            position_before_line = f.tell()
            line0 = f.readline()
            while line0 != '':
                line = remove_comment_from_line(line0)[0]
                tokens_upper = line.upper().split()
                if tokens_upper and tokens_upper[0].startswith('THER'):
                    found_thermo_block = True
                    # Unread the line (we'll re-read it in read_thermo_block())
                    f.seek(position_before_line)
                    try:
                        formula_dict = read_thermo_block(f, species_dict)  # updates species_dict in place
                    except:
                        logging.error("Error reading thermo block around line:\n" + f.readline())
                        raise
                    assert formula_dict, "Didn't read any thermo data"
                position_before_line = f.tell()
                line0 = f.readline()
        assert found_thermo_block, "Couldn't find a line beginning with THERMO or THERM or THER in {0}".format(thermo_file)
        assert formula_dict, "Didn't read any thermo data from {0}".format(thermo_file)

        # Save the formula_dict, converting from {'c':1,'h':4} into "CH4" in the process.
        #self.formula_dict = {label: convert_formula(formula) for label, formula in formula_dict.items()}
        self.formula_dict = dict(
            (label, convert_formula(formula))
            for (label, formula) in formula_dict.items())
        # thermo_dict contains original thermo as read from chemkin thermo file
        #self.thermo_dict = {s.label: s.thermo for s in species_dict.values() }
        self.thermo_dict = dict((s.label, s.thermo)
                               for s in species_dict.values())

    def load_reactions(self, reactions_file):
        logging.info("Reading reactions...")
        with open(reactions_file, encoding='utf-8', errors='ignore') as f:
            reaction_list = read_reactions_block(f, self.species_dict, read_comments=True)
        logging.info("Read {0} reactions from chemkin file.".format(len(reaction_list)))

        # convert from list to Library, so we can detect duplicates
        temporary_library = rmgpy.data.kinetics.library.KineticsLibrary()
        temporary_library.entries = {}
        for index, reaction in enumerate(reaction_list):
            entry = Entry(
                index = index+1,
                item = reaction,
                data = reaction.kinetics,
                label = str(reaction)
            )
            entry.long_desc = reaction.kinetics.comment or ''
            reaction.kinetics.comment = ''
            temporary_library.entries[index+1] = entry
            reaction.kinetics = None
        temporary_library.check_for_duplicates(mark_duplicates=True)  # mark_duplicates=True
        temporary_library.convert_duplicates_to_multi()
        # convert back to list
        new_reaction_list = []
        for entry in temporary_library.entries.values():
            reaction = entry.item
            reaction.kinetics = entry.data
            new_reaction_list.append(reaction)
        logging.info("Read {} reactions. After converting duplicates, have {} reactions".format(
                                                            len(reaction_list), len(new_reaction_list)))
        reaction_list = new_reaction_list

        self.chemkin_reactions = reaction_list
        self.chemkin_reactions_unmatched = self.chemkin_reactions[:]  # make a copy

        # Populate the self.chemkin_reactions_dict such that
        # self.chemkin_reactions_dict[chemkin_label] = {set of reactions it is part of}
        chemkin_reactions_dict = self.chemkin_reactions_dict
        for chemkin_reaction in self.chemkin_reactions:
            for reacting in [chemkin_reaction.reactants, chemkin_reaction.products]:
                for ck_species in reacting:
                    label = ck_species.label
                    if label not in chemkin_reactions_dict:
                        chemkin_reactions_dict[label] = set()
                    chemkin_reactions_dict[label].add(chemkin_reaction)

    
    def save_reactions_to_storage(self):
        """
        Save chemkin reactions to database for dashboard display.
        
        Call this after loading reactions and periodically during processing
        to update match status.
        """
        if not self.vote_db_enabled:
            return {'success': False, 'message': 'Vote database not enabled'}
        
        if not self.chemkin_reactions:
            return {'success': False, 'message': 'No reactions loaded'}
        
        try:
            # Save all reactions with current match status
            saved_count = self.vote_db.save_chemkin_reactions(
                job_id=self.job_id,
                reactions=self.chemkin_reactions,
                unmatched_reactions=self.chemkin_reactions_unmatched
            )
            
            # Update job statistics
            total_reactions = len(self.chemkin_reactions)
            matched_reactions = total_reactions - len(self.chemkin_reactions_unmatched)
            
            cursor = self.vote_db.conn.cursor()
            cursor.execute("""
                UPDATE import_jobs 
                SET total_reactions = ?,
                    matched_reactions = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE job_id = ?
            """, (total_reactions, matched_reactions, self.job_id))
            self.vote_db.conn.commit()
            
            logging.info(f"✓ Saved {saved_count} reactions to database "
                        f"({matched_reactions}/{total_reactions} matched)")
            
            return {
                'success': True,
                'saved': saved_count,
                'total': total_reactions,
                'matched': matched_reactions
            }
            
        except Exception as e:
            logging.error(f"Error saving reactions to storage: {e}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'message': str(e)}


    def update_reaction_status_in_storage(self):
        """
        Update just the match status of reactions (faster than full save).
        
        Call this after reactions are matched/unmatched to keep dashboard current.
        """
        if not self.vote_db_enabled:
            return
        
        try:
            # Determine which reactions are matched
            matched_indices = set()
            identified_indices = set()
            
            for idx, reaction in enumerate(self.chemkin_reactions):
                if reaction not in self.chemkin_reactions_unmatched:
                    matched_indices.add(idx)
                
                # Check if all species are identified
                all_identified = True
                for species in list(reaction.reactants) + list(reaction.products):
                    if species.label not in self.identified_labels:
                        all_identified = False
                        break
                if all_identified:
                    identified_indices.add(idx)
            
            # Bulk update
            updated = self.vote_db.bulk_update_reaction_status(
                self.job_id, matched_indices, identified_indices
            )
            
            logging.debug(f"Updated status for {updated} reactions")
            
        except Exception as e:
            logging.error(f"Error updating reaction status: {e}")

    def prune_inert_species(self):
        """
        Remove from consideration any chemkin species that don't participate in any reactions
        """
        reactive_species = set()
        reactive_molecules = set()
        for s in ['N#N', '[Ar]', ]:
            reactive_molecules.add(Molecule(smiles=s))
        for reaction in self.chemkin_reactions:
            for species in reaction.reactants:
                reactive_species.add(species)
            for species in reaction.products:
                reactive_species.add(species)
            if isinstance(reaction.kinetics, rmgpy.kinetics.PDepKineticsModel):
                for molecule in reaction.kinetics.efficiencies.keys():
                    reactive_molecules.add(molecule)
        unreactive_species = []
        for species in self.species_list:
            if species not in reactive_species:
                label = species.label
                if label in self.identified_labels:
                    this_molecule = self.species_dict_rmg[label].molecule[0]
                    for reactive_molecule in reactive_molecules:
                        if reactive_molecule.is_isomorphic(this_molecule):
                            break
                    else:
                        unreactive_species.append(species)
                else:
                    unreactive_species.append(species)
        for species in unreactive_species:
            label = species.label
            logging.info("Removing species {0} because it doesn't react".format(label))
            self.species_list.remove(species)
            del (self.species_dict[label])
            if label in self.species_dict_rmg:
                del (self.species_dict_rmg[label])
            self.clear_thermo_match(label)
            if label in self.identified_labels:
                self.identified_labels.remove(label)
            if label in self.identified_unprocessed_labels:
                self.identified_unprocessed_labels.remove(label)
            del (self.formula_dict[label])
        logging.info("Removed {0} species that did not react.".format(
            len(unreactive_species)))

    def load_blocked_matches(self):
        """
        Load the list of blocked matches.
        """
        logging.info("Reading blocked matches...")
        if not os.path.exists(self.blocked_matches_file):
            logging.info("Blocked Matches file does not exist. Will create.")
            return
        #: blocked_smiles[chemkin_label][smiles] = username_who_blocked_it
        blocked_smiles = {}
        line = None
        with open(self.blocked_matches_file) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    user = None
                    if '!' in line:
                        line, comments = line.split("!", 1)
                        if comments:
                            usermatch = re.match("\s+Blocked by (.*)", comments)
                            if usermatch:
                                user = usermatch.group(1)
                    tokens = line.split()
                    assert len(tokens) == 2, "Not two tokens on line (was expecting NAME    SMILES)"
                    name, smiles = tokens

                    if name not in blocked_smiles:
                        blocked_smiles[name] = dict()
                    blocked_smiles[name][smiles] = user
                except Exception as e:
                    logging.warning("Error reading line '{0}'".format(line))
                    raise e
        if line is None or not line.endswith('\n'):
            logging.info("Ensuring blocked matches file ends with a line break")
            with open(self.blocked_matches_file, 'a') as f:
                f.write('\n')

        for species_label in blocked_smiles:
            if species_label not in self.formula_dict:
                logging.info("{0} is not in the chemkin model. Skipping".format(species_label))
                continue
            formula = self.formula_dict[species_label]
            for smiles, username in blocked_smiles[species_label].items():
                molecule = Molecule(smiles=smiles)
                if formula != molecule.get_formula():
                    raise Exception("{0} cannot be {1} because the SMILES formula is {2} not required formula {3}.".format(species_label, smiles, molecule.get_formula(), formula))
                logging.info("Blocking {0} from being {1}".format(species_label, smiles))

                rmg_species, was_new = self.rmg_object.reaction_model.make_new_species(molecule)
                rmg_species.generate_resonance_structures()
                if was_new:
                    self.draw_species(rmg_species)
                    rmg_species.thermo = generate_thermo_data(rmg_species)

                if species_label not in self.blocked_matches:
                    self.blocked_matches[species_label] = dict()
                self.blocked_matches[species_label][rmg_species] = username


    def load_known_species(self, known_species_file):
        """
        Load (or make) the list of known species
        """
        logging.info("Reading known species...")
        if not os.path.exists(known_species_file):
            logging.info("Known species file does not exist. Will create on first manual match.")
            return
        known_smiles = {}
        known_names = []
        identified_labels = []
        line = None
        with open(known_species_file) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    user = None
                    if '!' in line:
                        line, comments = line.split("!",1)
                        if comments:
                            usermatch = re.match("\s+Confirmed by ([^.]*)",comments)
                            if usermatch:
                                user = usermatch.group(1)
                    tokens = line.split()
                    if len(tokens) == 0:
                        # blank line (after comments removed)
                        continue
                    assert len(tokens) == 2, "Not two tokens on line (was expecting NAME    SMILES)"
                    name, smiles = tokens
                    if name in known_smiles:
                        assert smiles == known_smiles[name], "{0} defined twice, as {1} and {2}".format(name, known_smiles[name], smiles)
                    known_smiles[name] = smiles
                    if user:
                        self.identified_by[name] = user
                    if name not in known_names:
                        known_names.append(name)
                except Exception as e:
                    logging.warning("Error reading line '{0}'".format(line))
                    raise e
        if not line.endswith('\n'):
            logging.info("Ensuring known species file ends with a line break")
            with open(known_species_file, 'a') as f:
                f.write('\n')

        special_smiles_to_adj_list = {
            'singlet[CH2]': """
                            multiplicity 1
                            1 C u0 p1 {2,S} {3,S}
                            2 H u0 {1,S}
                            3 H u0 {1,S}
                            """,
            'triplet[CH2]': """
                            multiplicity 3
                            1 C u2 {2,S} {3,S}
                            2 H u0 {1,S}
                            3 H u0 {1,S}
                            """,
            'singletC=[C]': """
                            multiplicity 1
                            1 C u0 {2,D} {3,S} {4,S}
                            2 C u0 p1 {1,D}
                            3 H u0 {1,S}
                            4 H u0 {1,S}
                            """,
            'tripletC=[C]': """
                            multiplicity 3
                            1 C u0 {2,D} {3,S} {4,S}
                            2 C u2 {1,D}
                            3 H u0 {1,S}
                            4 H u0 {1,S}
                            """,
            'singlet[CH]O': """
                            1 O u0 p1 c+1 {2,D} {4,S}
                            2 C u0 p1 c-1 {1,D} {3,S}
                            3 H u0 p0 c0 {2,S}
                            4 H u0 p0 c0 {1,S}
                            """, # aka [CH-]=[OH+]
            'triplet[CH]O': """
                            multiplicity 3
                            1 C u2 p0 c0 {2,S} {3,S}
                            2 O u0 p2 c0 {1,S} {4,S}
                            3 H u0 p0 c0 {1,S}
                            4 H u0 p0 c0 {2,S}
                            """, # aka [CH]O
            '[C]':          "1 C u0 p2 c0",
            'excited[OH]':  """
                            multiplicity 2
                            molecularTermSymbol A^2S+
                            1 O u1 p2 c0 {2,S}
                            2 H u0 p0 c0 {1,S}
                            """, # the 'A' in the molecular term symbol means first excited state
            'excited[CH]':  """
                            multiplicity 2
                            molecularTermSymbol A^2S+
                            1 C u1 p1 c0 {2,S}
                            2 H u0 p0 c0 {1,S}
                            """, # the 'A' in the molecular term symbol means first excited state
            '[O]singlet':  "1 O u0 p3 c0",  # RMG (via RDKit?) thinks this is water, and prints the wrong SMILES
            '[NH2+][O-]':    """
                            multiplicity 2
                            1 N u1 p0 c+1 {2,S} {3,S} {4,S}
                            2 O u0 p3 c-1 {1,S}
                            3 H u0 p0 c0 {1,S}
                            4 H u0 p0 c0 {1,S}
                            """,  # workaround a bug
            'singlet[CH]F': """
                            multiplicity 1
                            1 C u0 p1 c0 {2,S} {3,S}
                            2 H u0 p0 c0 {1,S}
                            3 F u0 p3 c0 {1,S}
                            """,
            'singlet[CH]Cl': """
                            multiplicity 1
                            1 C u0 p1 c0 {2,S} {3,S}
                            2 H u0 p0 c0 {1,S}
                            3 Cl u0 p3 c0 {1,S}
                            """,
            'singletCl[C]Cl': """
                            multiplicity 1
                            1 C u0 p1 c0 {2,S} {3,S}
                            2 Cl u0 p3 c0 {1,S}
                            3 Cl u0 p3 c0 {1,S}
                            """,
            'singletF[C]F': """
                            multiplicity 1
                            1 F u0 p3 c0 {2,S}
                            2 C u0 p1 c0 {1,S} {3,S}
                            3 F u0 p3 c0 {2,S}
                            """,
            'singletF[C]Cl': """
                            multiplicity 1
                            1 F u0 p3 c0 {2,S}
                            2 C u0 p1 c0 {1,S} {3,S}
                            3 Cl u0 p3 c0 {2,S}
                            """,
            'doublet[C]F': """
                            multiplicity 2
                            1 C u1 p1 c0 {2,S}
                            2 F u0 p3 c0 {1,S}
                            """,
            'singlet[N]C=C': """
                            multiplicity 1
                            1 N u0 p2 c0 {2,S}
                            2 C u0 p0 c0 {1,S} {3,D} {4,S}
                            3 C u0 p0 c0 {2,D} {5,S} {6,S}
                            4 H u0 p0 c0 {2,S}
                            5 H u0 p0 c0 {3,S}
                            6 H u0 p0 c0 {3,S}
                            """,
            }

        for species_label in known_names:
            if species_label not in self.formula_dict:
                logging.info("{0} is not in the chemkin model. Skipping".format(species_label))
                continue
            formula = self.formula_dict[species_label]
            smiles = known_smiles[species_label]
            if smiles in special_smiles_to_adj_list:
                adjlist = special_smiles_to_adj_list[smiles]
                molecule = Molecule()
                try:
                    molecule.from_adjacency_list(adjlist)
                except:
                    logging.exception(adjlist)
            else:
                molecule = Molecule(smiles=smiles)
            if formula != molecule.get_formula():
                raise Exception("{0} cannot be {1} because the SMILES formula is {2} not required formula {3}. \n{4}".format(species_label, smiles, molecule.get_formula(), formula, molecule.to_adjacency_list()))
            logging.info("I think {0} is {1} based on its label".format(species_label, smiles))
            self.smiles_dict[species_label] = smiles

            species = self.species_dict[species_label]
            species.molecule = [molecule]
            species.generate_resonance_structures()
            identified_labels.append(species_label)

            if self.vote_db_enabled:
                self.vote_db.save_identified_species(
                job_id=self.job_id,
                chemkin_label=species_label,
                rmg_species=species,
            identified_by=self.identified_by.get(species_label, 'Auto'),
                identification_method='formula'
            )

        logging.info("Identified {0} species:".format(len(identified_labels)))
        for species_label in identified_labels:
            logging.info("   {0}".format(species_label))

        self.identified_labels.extend(identified_labels)

    def initializeRMG(self, args):
        """
        Create an RMG object, store it in self.rmg_object, and set it up.
        
        This loads the database, makes some settings, etc.
        `args` should have attributes `output_directory` and `scratch_directory`.
        Also needs a global variable databse_directory
        """
        logging.info("Loading RMG database...")
        rmg = RMG()
        rmg.output_directory = args.output_directory
        rmg.scratch_directory = args.scratch_directory
        rmgpy.util.make_output_subdirectory(rmg.output_directory, 'species')
        rmg.database_directory = database_dictionary
        rmg.thermo_libraries = ['primaryThermoLibrary',
                               'BurkeH2O2',
                               'DFT_QCI_thermo',
                               'CBS_QB3_1dHR',
                               'FFCM1(-)',
                               'GRI-Mech3.0-N',
                               'JetSurF2.0',
                               'CurranPentane', ]
        rmg.kinetics_families = ['default', ]
        rmg.reaction_libraries = [('BurkeH2O2inN2', False),
                                 ('FFCM1(-)', False),
                                 ('JetSurF2.0', False),
                                 ('CurranPentane', False),
                                 ('Glarborg/highP', False),
                                 ('GRI-Mech3.0-N', False), ]

        rmg.kinetics_depositories = ['default', ]
        rmg.adsorption_groups = 'adsorptionPt111'

        rmgpy.rmg.input.rmg = rmg  # put it in this scope so these functions can modify it

        if args.pdep:
            rmgpy.rmg.input.pressure_dependence(
                method='modified strong collision',
                maximum_grain_size=(0.5, 'kcal/mol'),
                minimumNumberOfGrains=250,
                temperatures=(300, 2000, 'K', 8),
                pressures=(0.01, 100, 'atm', 3),
                interpolation=('pdeparrhenius',),
            )

        from rdkit import Chem
        if not Chem.inchi.INCHI_AVAILABLE:
            logging.warning("RDKit installed without InChI support so running without QM calculations!")
        elif args.mopac:
            logging.info("Using MOPAC semiemprical quantum calculations for cyclic species.")
            rmgpy.rmg.input.quantum_mechanics(
                software='mopac',
                method='pm3',
                fileStore=os.path.join(os.path.normpath(os.path.join(rmgpy.get_path(), '..')), 'QMfiles'),  # ToDo: fix this
                scratchDirectory=os.path.join(os.path.normpath(os.path.join(rmgpy.get_path(), '..')), 'QMscratch'),
                onlyCyclics=True,
                maxRadicalNumber=0,
            )

        rmg.load_database()
        logging.info("Loaded database.")

        self.thermo_libraries_to_check.extend(rmg.database.thermo.library_order) # add the RMG libraries
        
        # Should probably look elsewhere, but this is where they tend to be for now...
        directory = os.path.abspath(os.path.join(os.path.split(os.path.abspath(args.thermo))[0], '..'))
        # if that's some subfolder of RMG-models, move up to RMG-models level
        if directory.find('RMG-models'):
            directory = directory[:directory.find('RMG-models')]+'RMG-models'
        logging.info("Looking in {dir} for additional thermo libraries to import".format(dir=directory))
        for root, dirs, files in os.walk(directory):
            for filename in files:
                if not filename == 'ThermoLibrary.py':
                    continue
                path = os.path.join(root, filename)
                logging.info("I think I found a thermo library at {0}".format(path))
                if os.path.abspath(path).startswith(os.path.split(os.path.abspath(args.thermo))[0]):
                    logging.info("But it's the model currently being imported, so not loading.")
                    break
                library = rmgpy.data.thermo.ThermoLibrary()
                library.SKIP_DUPLICATES = True
                library.load(path, rmg.database.thermo.local_context, rmg.database.thermo.global_context)
                library.label = root.split('/')[-2]
                rmg.database.thermo.libraries[library.label] = library
                # Load them (for the check_thermo_libraries method) but don't trust them
                self.thermo_libraries_to_check.append(library.label)
                # rmg.database.thermo.library_order.append(library.label)

        rmg.reaction_model = rmgpy.rmg.model.CoreEdgeReactionModel()
        rmg.reaction_model.kinetics_estimator = 'rate rules'
        rmg.reaction_model.verbose_comments = True
        rmg.initial_species = []
        rmg.reaction_systems = []

        rmgpy.util.make_output_subdirectory(rmg.output_directory, 'pdep')  # deletes contents
        # This is annoying!
        if rmg.pressure_dependence:
            rmg.pressure_dependence.output_file = rmg.output_directory
            rmg.reaction_model.pressure_dependence = rmg.pressure_dependence
        #rmg.reaction_model.reaction_generation_options = rmg.reaction_generation_options
        if rmg.quantum_mechanics:
            rmg.quantum_mechanics.set_default_output_directory(rmg.output_directory)
            rmg.reaction_model.quantum_mechanics = rmg.quantum_mechanics
            rmg.quantum_mechanics.initialize()

        # We need to properly initialize the database so that we can
        # find kinetics without crashing.
        logging.info('Adding rate rules from training set in kinetics families...')
        for family in rmg.database.kinetics.families.values():
            if not family.auto_generated:
                family.add_rules_from_training(thermo_database=rmg.database.thermo)
        logging.info('Filling in rate rules in kinetics families by averaging...')
        for family in rmg.database.kinetics.families.values():
            if not family.auto_generated:
                family.fill_rules_by_averaging_up()

        self.rmg_object = rmg
        return rmg

    def species_match(self, rmg_species, chemkin_species):
        """
        Return True if the species might match, else False.
        
        i.e. if chemkin_species has been identified, it must be the rmg_species,
        but if it hasn't it must at least have the same formula.
        If it matches based only on formula, the match it is added to the self.suggested_matches dictionary.
        If it is blocked, return false.
        """
        chemkin_label = chemkin_species.label
        identified_labels = self.identified_labels
        if chemkin_label in identified_labels:
            return self.species_dict_rmg[chemkin_label] is rmg_species
        elif rmg_species.label in identified_labels:
            return False
        else:
            if self.formula_dict[chemkin_label] == rmg_species.formula:
                if rmg_species in self.blocked_matches.get(chemkin_label, {}):
                    return False
                self.suggested_matches[chemkin_label] = rmg_species
                return True
            else:
                return False

    def reactions_match(self, rmg_reaction, chemkin_reaction, either_direction=True):
        """
        This is based on the rmg.reaction.Reaction.is_isomorphic method
 
        Return ``True`` if rmg_reaction is the same as the chemkin_reaction reaction,
        or ``False`` if they are different.
        If `either_direction=False` then the directions must match.
        """
        species_match = self.species_match
        # Compare reactants to reactants

        # get things we refer to a lot into the local namespace, to reduce lookups
        rmg_reactants = rmg_reaction.reactants
        ck_reactants = chemkin_reaction.reactants
        len_rmg_reactants = len(rmg_reactants)
        len_ck_reactants = len(ck_reactants)

        forward_reactants_match = False
        if len_rmg_reactants == 1 and len_ck_reactants == 1:
            if species_match(rmg_reactants[0], ck_reactants[0]):
                forward_reactants_match = True
        elif len_rmg_reactants == 2 and len_ck_reactants == 2:
            if species_match(rmg_reactants[0], ck_reactants[0]) and species_match(rmg_reactants[1], ck_reactants[1]):
                forward_reactants_match = True
            elif species_match(rmg_reactants[0], ck_reactants[1]) and species_match(rmg_reactants[1], ck_reactants[0]):
                forward_reactants_match = True
        elif len_rmg_reactants == 3 and len_ck_reactants == 3:
            if species_match(rmg_reactants[0], ck_reactants[0]):
                if (species_match(rmg_reactants[1], ck_reactants[1]) and
                    species_match(rmg_reactants[2], ck_reactants[2])):
                    forward_reactants_match = True
                elif(species_match(rmg_reactants[1], ck_reactants[2]) and
                    species_match(rmg_reactants[2], ck_reactants[1])):
                    forward_reactants_match = True
            elif species_match(rmg_reactants[0], ck_reactants[1]):
                if (species_match(rmg_reactants[1], ck_reactants[0]) and
                    species_match(rmg_reactants[2], ck_reactants[2])):
                    forward_reactants_match = True
                elif(species_match(rmg_reactants[1], ck_reactants[2]) and
                    species_match(rmg_reactants[2], ck_reactants[0])):
                    forward_reactants_match = True
            elif species_match(rmg_reactants[0], ck_reactants[2]):
                if (species_match(rmg_reactants[1], ck_reactants[0]) and
                    species_match(rmg_reactants[2], ck_reactants[1])):
                    forward_reactants_match = True
                elif (species_match(rmg_reactants[1], ck_reactants[1]) and
                    species_match(rmg_reactants[2], ck_reactants[0])):
                    forward_reactants_match = True
        elif len_rmg_reactants == len_ck_reactants:
            raise NotImplementedError("Can't check isomorphism of reactions with {0} reactants".format(len_rmg_reactants))

        # Return False now if we can already be sure
        if not forward_reactants_match:
            if not either_direction:
                return False

        rmg_products = rmg_reaction.products
        ck_products = chemkin_reaction.products
        len_rmg_products = len(rmg_products)
        len_ck_products = len(ck_products)
        # Compare products to products
        forward_products_match = False
        if len_rmg_products == 1 and len_ck_products == 1:
            if species_match(rmg_products[0], ck_products[0]):
                forward_products_match = True
        elif len_rmg_products == 2 and len_ck_products == 2:
            if species_match(rmg_products[0], ck_products[0]) and species_match(rmg_products[1], ck_products[1]):
                forward_products_match = True
            elif species_match(rmg_products[0], ck_products[1]) and species_match(rmg_products[1], ck_products[0]):
                forward_products_match = True
        elif len_rmg_products == 3 and len_ck_products == 3:
            if species_match(rmg_products[0], ck_products[0]):
                if (species_match(rmg_products[1], ck_products[1]) and
                    species_match(rmg_products[2], ck_products[2])):
                    forward_products_match = True
                elif (species_match(rmg_products[1], ck_products[2]) and
                    species_match(rmg_products[2], ck_products[1])):
                    forward_products_match = True
            elif species_match(rmg_products[0], ck_products[1]):
                if (species_match(rmg_products[1], ck_products[0]) and
                    species_match(rmg_products[2], ck_products[2])):
                    forward_products_match = True
                elif (species_match(rmg_products[1], ck_products[2]) and
                    species_match(rmg_products[2], ck_products[0])):
                    forward_products_match = True
            elif species_match(rmg_products[0], ck_products[2]):
                if (species_match(rmg_products[1], ck_products[0]) and
                    species_match(rmg_products[2], ck_products[1])):
                    forward_products_match = True
                elif (species_match(rmg_products[1], ck_products[1]) and
                    species_match(rmg_products[2], ck_products[0])):
                    forward_products_match = True
        elif len_rmg_products == len_ck_products:
            raise NotImplementedError("Can't check isomorphism of reactions with {0} products".format(len_rmg_products))

        # Return now, if we can
        if (forward_reactants_match and forward_products_match):
            return True
        if not either_direction:
            return False

        # Compare reactants to products
        reverse_reactants_match = False
        if len_rmg_reactants == 1 and len_ck_products == 1:
            if species_match(rmg_reactants[0], ck_products[0]):
                reverse_reactants_match = True
        elif len_rmg_reactants == 2 and len_ck_products == 2:
            if species_match(rmg_reactants[0], ck_products[0]) and species_match(rmg_reactants[1], ck_products[1]):
                reverse_reactants_match = True
            elif species_match(rmg_reactants[0], ck_products[1]) and species_match(rmg_reactants[1], ck_products[0]):
                reverse_reactants_match = True
        elif len_rmg_reactants == 3 and len_ck_products == 3:
            if (species_match(rmg_reactants[0], ck_products[0]) and
                    species_match(rmg_reactants[1], ck_products[1]) and
                    species_match(rmg_reactants[2], ck_products[2])):
                reverse_reactants_match = True
            elif (species_match(rmg_reactants[0], ck_products[0]) and
                    species_match(rmg_reactants[1], ck_products[2]) and
                    species_match(rmg_reactants[2], ck_products[1])):
                reverse_reactants_match = True
            elif (species_match(rmg_reactants[0], ck_products[1]) and
                    species_match(rmg_reactants[1], ck_products[0]) and
                    species_match(rmg_reactants[2], ck_products[2])):
                reverse_reactants_match = True
            elif (species_match(rmg_reactants[0], ck_products[2]) and
                    species_match(rmg_reactants[1], ck_products[0]) and
                    species_match(rmg_reactants[2], ck_products[1])):
                reverse_reactants_match = True
            elif (species_match(rmg_reactants[0], ck_products[1]) and
                    species_match(rmg_reactants[1], ck_products[2]) and
                    species_match(rmg_reactants[2], ck_products[0])):
                reverse_reactants_match = True
            elif (species_match(rmg_reactants[0], ck_products[2]) and
                    species_match(rmg_reactants[1], ck_products[1]) and
                    species_match(rmg_reactants[2], ck_products[0])):
                reverse_reactants_match = True
        elif len_rmg_reactants == len_ck_products:
            raise NotImplementedError("Can't check isomorphism of reactions with {0} reactants".format(len_rmg_reactants))

        # Should have already returned if matched in forward direction.
        # Return False now if we can be sure it's no match.
        if not reverse_reactants_match:
            return False

        # Compare products to reactants
        reverse_products_match = False
        if len_rmg_products == 1 and len_ck_reactants == 1:
            if species_match(rmg_products[0], ck_reactants[0]):
                reverse_products_match = True
        elif len_rmg_products == 2 and len_ck_reactants == 2:
            if species_match(rmg_products[0], ck_reactants[0]) and species_match(rmg_products[1], ck_reactants[1]):
                reverse_products_match = True
            elif species_match(rmg_products[0], ck_reactants[1]) and species_match(rmg_products[1], ck_reactants[0]):
                reverse_products_match = True
        elif len_rmg_products == 3 and len_ck_reactants == 3:
            if (species_match(rmg_products[0], ck_reactants[0]) and
                    species_match(rmg_products[1], ck_reactants[1]) and
                    species_match(rmg_products[2], ck_reactants[2])):
                reverse_products_match = True
            elif (species_match(rmg_products[0], ck_reactants[0]) and
                    species_match(rmg_products[1], ck_reactants[2]) and
                    species_match(rmg_products[2], ck_reactants[1])):
                reverse_products_match = True
            elif (species_match(rmg_products[0], ck_reactants[1]) and
                    species_match(rmg_products[1], ck_reactants[0]) and
                    species_match(rmg_products[2], ck_reactants[2])):
                reverse_products_match = True
            elif (species_match(rmg_products[0], ck_reactants[2]) and
                    species_match(rmg_products[1], ck_reactants[0]) and
                    species_match(rmg_products[2], ck_reactants[1])):
                reverse_products_match = True
            elif (species_match(rmg_products[0], ck_reactants[1]) and
                    species_match(rmg_products[1], ck_reactants[2]) and
                    species_match(rmg_products[2], ck_reactants[0])):
                reverse_products_match = True
            elif (species_match(rmg_products[0], ck_reactants[2]) and
                    species_match(rmg_products[1], ck_reactants[1]) and
                    species_match(rmg_products[2], ck_reactants[0])):
                reverse_products_match = True
        elif len_rmg_products == len_ck_reactants:
            raise NotImplementedError("Can't check isomorphism of reactions with {0} products".format(len_rmg_products))

        # should have already returned if it matches forwards, or we're not allowed to match backwards
        return  (reverse_reactants_match and reverse_products_match)


    def species_react_according_to_chemkin(self, rmg_species1, rmg_species2):
        """
        Return true if the two species have been identified and are on 
        the same side of at least one chemkin reaction. i.e. we know that
        according to the chemkin file, they react with each other.
        
        If rmg_species1 and rmg_species2 are the same thing, it must react
        with itself to return true. (eg. A + A -> products)
        """
        try:
            set1 = self.chemkin_reactions_dict[rmg_species1.label]
            set2 = self.chemkin_reactions_dict[rmg_species2.label]
        except KeyError:
            # one of the rmg species has not yet been given a label corresponding to a chemkin species
            # therefore it has not been identified
            return False
        for chemkin_reaction in set1.intersection(set2):
            for reacting in [chemkin_reaction.reactants, chemkin_reaction.products]:
                matched_species = list()
                for ck_species in reacting:
                    if ck_species.label in self.identified_labels:
                        matched_species.append(self.species_dict_rmg[ck_species.label])
                if rmg_species1 in matched_species:
                    matched_species.remove(rmg_species1)
                    if rmg_species2 in matched_species:
                        return True
        return False

    def identify_small_molecules(self):
        """Identify anything little that is uniquely determined by its chemical formula"""

        known_formulas = {
             'CH4': 'C',
             'CH3': '[CH3]',
             'CO2': 'O=C=O',
             'H2O': 'O',
             'HO': '[OH]',
             'C2H6': 'CC',
             'C2H5': 'C[CH2]',
             'C2H4': 'C=C',
             'C2H': '[C]#C',
             'O2': '[O][O]',
             'H2': '[H][H]',
             'H2O2': 'OO',
             'O': '[O]',
             'N2': 'N#N',
             'CO': '[C-]#[O+]',
             'HO2': '[O]O',
             'C3H8': 'CCC',
             'CH': '[CH]',
             'Ar': '[Ar]',
             'He': '[He]',
             'H': '[H]',
             'CH4O': 'CO',
             'C': '[C]',
             }
        identified_labels = []

        # use species_list if it is not None or empty, else the formula_dict keys.
        for species_label in [s.label for s in self.species_list or []] or self.formula_dict.keys():
            if species_label in self.identified_labels:
                continue
            formula = self.formula_dict[species_label]
            if formula in known_formulas:
                known_smiles = known_formulas[formula]
                logging.info("I think {0} is {1} based on its formula".format(species_label, known_smiles))
                smiles = known_smiles
            else:
                continue
            self.smiles_dict[species_label] = smiles
            while formula != Molecule(smiles=smiles).get_formula():
                smiles = input("SMILES {0} has formula {1} not required formula {2}. Try again:\n".format(smiles, Molecule(smiles=smiles).get_formula(), formula))
            species = self.species_dict[species_label]
            if smiles == '[C]':  # The SMILES is interpreted as a quintuplet and we can't estimate the thermo
                species.molecule = [Molecule().from_adjacency_list('1 C u0 p2 c0')]
            else:
                species.molecule = [Molecule(smiles=smiles)]
            species.generate_resonance_structures()
            identified_labels.append(species_label)
            self.save_match_to_file(species_label, species)
            
            # Save each identified species to the database
            if self.vote_db_enabled:
                self.vote_db.save_identified_species(
                    job_id=self.job_id,
                    chemkin_label=species_label,
                    rmg_species=species,
                    identified_by=self.identified_by.get(species_label, 'Auto'),
                    identification_method='small_molecule'
                )

        logging.info("Identified {0} species:".format(len(identified_labels)))
        for species_label in identified_labels:
            logging.info("   {0}".format(species_label))

        self.identified_labels.extend(identified_labels)

    def ask_for_match_smiles(self, chemkin_species):
            species_label = chemkin_species.label
            formula = self.formula_dict[species_label]
            print("Species {species} has formula {formula}".format(species=species_label, formula=formula))
            smiles = input('What is its SMILES?\n')
            while formula != Molecule(smiles=smiles).get_formula():
                smiles = input("SMILES {0} has formula {1} not required formula {2}. Try again:\n".format(smiles, Molecule(smiles=smiles).get_formula(), formula))
            self.smiles_dict[species_label] = smiles
            species = self.species_dict[species_label]
            species.molecule = [Molecule(smiles=smiles)]
            species.generate_resonance_structures()
            self.identified_labels.append(species_label)

    def ask_for_match_id(self, species_label, possible_matches):
            """
            Ask user for a match for a given species_label, choosing from the iterable possible_matches
            """
            # print "Species {species} has formula {formula}".format(species=species_label, formula=formula)
            #matches_dict = {species.index: species for species in possible_matches }
            matches_dict = dict((species.index, species) for species in possible_matches)
            possible_indices_str = [str(i) for i in sorted(matches_dict.keys())]
            print("Species {0} could be one of:".format(species_label))
            for index in sorted(matches_dict.keys()):
                rmg_species = matches_dict[index]
                dH = self.get_enthalpy_discrepancy(species_label, rmg_species)
                Nvotes = len(self.votes[species_label][rmg_species])
                all_possible_chemkin_species = [ck for ck, matches in self.votes.items() if rmg_species in matches]
                print("{0:6d} {1:18s} {2:8.1f} kJ/mol  ({3} votes) {4!s}".format(index, rmg_species.label, dH, Nvotes, all_possible_chemkin_species))
            chosen_id = input('What is it? (see voting info above)\n')
            while chosen_id not in possible_indices_str:
                chosen_id = input("That wasn't one of {0}. Try again:\n".format(','.join(possible_indices_str)))

            rmg_species = matches_dict[int(chosen_id)]
            logging.info("Based on user input, matching {0} with {1!s}".format(species_label, rmg_species))
            self.set_match(species_label, rmg_species)
            return species_label, rmg_species

    def edge_reactions_matching(self, chemkin_reaction):
        """A generator giving edge reactions that match the given chemkin reaction"""
        reactions_match = self.reactions_match
        for edge_reaction in self.rmg_object.reaction_model.edge.reactions:
            if reactions_match(edge_reaction, chemkin_reaction):
                yield edge_reaction

    def chemkin_reactions_matching(self, rmg_reaction):
        """A generator giving chemkin reactions that match the given RMG reaction"""
        reactions_match = self.reactions_match
        for chemkin_reaction in self.chemkin_reactions:
            if reactions_match(rmg_reaction, chemkin_reaction):
                yield chemkin_reaction

    # def draw_species(self, rmg_species):
    #     "Draw a species, saved in 'species' directory named after its RMG name (label and id)."
    #     # Draw molecules if necessary
    #     fstr = os.path.join(self.rmg_object.output_directory, 'species',
    #                         '{0!s}.png'.format(rmg_species))
    #     if not os.path.exists(fstr):
    #         MoleculeDrawer().draw(rmg_species.molecule[0], 'png', fstr)

    def draw_species(self, rmg_species):
        "Draw a species, saved in 'species' directory named after its RMG name (label and id)."
        # Draw molecules if necessary
        fstr = os.path.join(self.rmg_object.output_directory, 'species',
                            '{0!s}.png'.format(rmg_species))
        logging.info(f"Attempting to draw species to: {fstr}")
        if not os.path.exists(fstr):
            try:
                MoleculeDrawer().draw(rmg_species.molecule[0], 'png', fstr)
                logging.info(f"Successfully drew species image: {fstr}")
            except Exception as e:
                logging.error(f"Failed to draw {rmg_species}: {e}")
        else:
            logging.info(f"Species image already exists: {fstr}")

    def draw_all_candidate_species(self):
        """Draws all the species that are in self.votes"""
        candidate_species = set()
        for possible_matches in self.votes.values():
            candidate_species.update(possible_matches.keys())
        for rmg_species in candidate_species:
            self.draw_species(rmg_species)

    def move_species_drawing(self, rmg_species):
        "Move a species drawing from 'species' directory to 'species/MATCHED' directory."
        source = os.path.join(self.rmg_object.output_directory, 'species',
                              '{0!s}.png'.format(rmg_species))
        destination = os.path.join(self.rmg_object.output_directory, 'species',
                                   'MATCHED', '{0!s}.png'.format(rmg_species))
        if os.path.exists(source):
            os.renames(source, destination)

    def get_enthalpy_discrepancy(self, chemkin_label, rmg_species):
        """
        Return the difference in enthalpy at 298.15K (or lowest valid T) in kJ/mol
        
        Returns (CHEMKIN file) - (RMG estimate)
        """
        ck_thermo = self.thermo_dict[chemkin_label]
        rmg_thermo = rmg_species.thermo
        temperature = max(298.15, ck_thermo.Tmin.value_si, rmg_thermo.Tmin.value_si)
        ckH = ck_thermo.get_enthalpy(temperature)
        rmgH = rmg_thermo.get_enthalpy(temperature)
        return (ckH - rmgH) / 1000.

    def backfill_enthalpy_discrepancy(self):
        """
        Backfill enthalpy_discrepancy for existing votes that have NULL values.
        
        This is useful when votes were saved before enthalpy calculation was implemented.
        Call this method after loading an existing job to update enthalpy values.
        
        Returns:
            Number of votes updated
        """
        if not self.vote_db:
            logging.warning("No vote database available for backfill")
            return 0
            
        # Build species_dict mapping rmg_species_index -> rmg_species object
        species_dict = {}
        for chemkin_label, possible_matches in self.votes.items():
            for rmg_species in possible_matches.keys():
                rmg_index = getattr(rmg_species, 'index', -1)
                if rmg_index != -1:
                    species_dict[rmg_index] = rmg_species
        
        updated_count = self.vote_db.backfill_enthalpy(
            self.job_id,
            self.get_enthalpy_discrepancy,
            species_dict
        )
        
        logging.info(f"✓ Backfilled enthalpy for {updated_count} votes")
        return updated_count

    def clear_tentative_match(self, chemkin_label, rmg_species):
        """
        Clear all tentative matches from that have either that label or species, 
        eg. because you've confirmed a match.
        """
        for match in self.tentative_matches:
            if match['label'] == chemkin_label or match['species'] == rmg_species:
                self.tentative_matches.remove(match)

    def set_tentative_match(self, chemkin_label, rmg_species, username=None):
        """
        Store a tentative match, waiting for user confirmation.
        
        If it conflicts with an existing tentative match, it instead
        removes that one, and returns false. If you want to add
        the new one, call it again.
        """
        self.draw_species(rmg_species)
        for match in self.tentative_matches:
            if match['label'] == chemkin_label:
                if match['species'] == rmg_species:
                    return True  # it's already there
                else:
                    # something else matches that label! Remove both
                    self.tentative_matches.remove(match)
                    return False
            elif match['species'] == rmg_species:
                # something else matches that rmg_species! Remove both
                self.tentative_matches.remove(match)
                return False
        for (l, s) in self.manual_matches_to_process:
            if l == chemkin_label:
                if s == rmg_species:
                    return True  # it's already matched
                else:
                    # It's matched something else!
                    logging.info("Tentative match conflicts with unprocessed manual match! Ignoring.")
                    return False
            elif s == rmg_species:
                logging.info("Tentative match conflicts with unprocessed manual match! Ignoring.")
                return False
        for l in self.identified_labels:
            s = self.species_dict[l]
            if l == chemkin_label:
                if s == rmg_species:
                    return True  # it's already matched
                else:
                    # It's matched something else!
                    logging.info("Tentative match conflicts with earlier match! Ignoring.")
                    return False
            elif s == rmg_species:
                logging.info("Tentative match conflicts with earlier match! Ignoring.")
                return False

        # haven't already returned? then
        # that tentative match is new, add it
        self.tentative_matches.append({
            'label': chemkin_label,
            'species': rmg_species,
            'enthalpy': self.get_enthalpy_discrepancy(chemkin_label, rmg_species),
            'username': username,
        })
        return True

    def check_thermo_libraries(self):
        """Compares the thermo data of species to be imported
        to all previously identified species in other libraries"""

        formula_to_labels_dict = {}
        for label, formula in self.formula_dict.items():
            if formula not in formula_to_labels_dict:
                formula_to_labels_dict[formula] = [label]
            else:
                formula_to_labels_dict[formula].append(label)

        for library_name in self.thermo_libraries_to_check:
            logging.info("Looking for matches in thermo library "+library_name)
            library = self.rmg_object.database.thermo.libraries[library_name]
            for __, entry in library.entries.items():
                formula = entry.item.get_formula()
                if formula in formula_to_labels_dict:
                    for ck_label in formula_to_labels_dict[formula]:
                        #Skip already identified species
                        if ck_label in self.identified_labels:
                            continue
                        ck_thermo = self.thermo_dict[ck_label]
                        try:
                            match = entry.data.is_identical_to(ck_thermo)  #is_identical_to requires improvement before this should be fully implemented
                        except ValueError:
                            logging.info("Error comparing two thermo entries, skipping entry for chemkin species {0} in the thermo library {1}".format(ck_label, library_name))
                            match = False
                        if match:
                            # Successfully found a tentative match, set the match and report.
                            rmg_species, was_new = self.rmg_object.reaction_model.make_new_species(entry.item, label=entry.label)
                            if was_new:
                                self.draw_species(rmg_species)
                            else:
                                pass
                                # logging.info("Thermo matches {0}, from {1}, but it's already in the model.".format(ck_label, library_name))
                            rmg_species.thermo = generate_thermo_data(rmg_species)

                            logging.info("Thermo match found for chemkin species {0} in thermo library {1}".format(ck_label, library_name))
                            self.set_thermo_match(ck_label, rmg_species, library_name, entry.label)

    def save_blocked_match_to_file(self, ck_label, rmg_species, username=None):
        """
        Save the blocked match to the blocked_matches_file
        """
        if username:
            user_text = "\t! Blocked by {0}".format(username)
            self.identified_by[ck_label] = username
        else:
            user_text = ""

        with open(self.blocked_matches_file, 'a') as f:
            f.write("{name}\t{smi}{user}\n".format(
                name=ck_label,
                smi=rmg_species.molecule[0].to_smiles(),
                user=user_text))
        return True

    def save_match_to_file(self, ck_label, rmg_species, username=None):
        """
        Save the match to the known_species_file
        """
        with open(self.known_species_file) as f:
            for line in f.readlines():
                line = line.split('!')[0]
                if not line.strip():
                    continue
                label, smiles = line.split()
                if label == ck_label:
                    logging.info("Trying to confirm match {0!s} but already matched to {1!s}".format(label, smiles))
                    return False
        if username:
            user_text = "\t! Confirmed by {0}".format(username)
            self.identified_by[ck_label] = username
        else:
            user_text = ""

        with open(self.known_species_file, 'a') as f:
            f.write("{name}\t{smi}{user}\n".format(
                name=ck_label,
                smi=rmg_species.molecule[0].to_smiles(),
                user=user_text))
        return True

    def set_thermo_match(self, chemkin_label, rmg_species, library_name,
                       library_species_name):
        """
        Store a match made by recognizing identical thermo from a library.
        """
        if chemkin_label not in self.thermo_matches:
            self.thermo_matches[chemkin_label] = dict()
        d = self.thermo_matches[chemkin_label]
        if rmg_species not in d: d[rmg_species] = list()
        d[rmg_species].append((library_name, library_species_name))

    def clear_thermo_match(self, chemkin_label, rmg_name=None):
        """
        Clear any thermo matches for that chemkin label,
        or only that rmg_label if supplied.
        """
        if chemkin_label in self.thermo_matches:
            if rmg_name is None:
                del (self.thermo_matches[chemkin_label])
                return
            for rmg_species in self.thermo_matches[chemkin_label].keys():
                if str(rmg_species) == rmg_name:
                    del (self.thermo_matches[chemkin_label][rmg_species])
            if len(self.thermo_matches[chemkin_label]) == 0:
                del (self.thermo_matches[chemkin_label])

    def block_match(self, chemkin_label, rmg_species, username=None):
        """Store a blocked match"""
        for match in self.tentative_matches:
            if match['label'] == chemkin_label:
                if match['species'] == rmg_species:
                    self.tentative_matches.remove(match)
                break
        self.clear_thermo_match(chemkin_label, str(rmg_species))
        self.save_blocked_match_to_file(chemkin_label, rmg_species,
                                    username=username)
        if chemkin_label not in self.blocked_matches:
            self.blocked_matches[chemkin_label] = dict()
        self.blocked_matches[chemkin_label][rmg_species] = username

        self.votes[chemkin_label].pop(rmg_species, None)

        # Save to local database
        if self.vote_db_enabled:
            try:
                self.vote_db.save_blocked_match(
                    job_id=self.job_id,
                    chemkin_label=chemkin_label,
                    rmg_species=rmg_species,
                    blocked_by=username,
                    reason="Manually blocked by user"
                )
                logging.info(f"✓ Saved blocked match to database: {chemkin_label}")
            except Exception as e:
                logging.error(f"Error saving blocked match: {e}")

    def set_match(self, chemkin_label, rmg_species):
        """Store a match, once you've identified it"""
        self.clear_tentative_match(chemkin_label, rmg_species)
        self.identified_labels.append(chemkin_label)
        self.identified_unprocessed_labels.append(chemkin_label)
        self.clear_thermo_match(chemkin_label)

        # For kinetics purposes, we convert the thermo to Wilhoit
        # This allows us to extrapolating H to 298 to find deltaH rxn
        # for ArrheniusEP kinetics,
        # and to 0K so we can do barrier height checks with E0.
        Cp0 = rmg_species.calculate_cp0(), 'J/mol/K'
        CpInf = rmg_species.calculate_cpinf(), 'J/mol/K'
        thermo = self.thermo_dict[chemkin_label]
        # pretend it was valid down to 298 K
        old_lowT = thermo.Tmin.value_si
        if old_lowT > 298.0:
            thermo.select_polynomial(thermo.Tmin.value_si).Tmin.value_si = min(298.0, thermo.Tmin.value_si)
            thermo.Tmin.value_si = min(298.0, thermo.Tmin.value_si)
            thermo.comment += "\n_low T polynomial Tmin changed from {0} to {1} K when importing to RMG".format(old_lowT, 298.0)
        thermo.Cp0 = Cp0
        thermo.CpInf = CpInf
        new_thermo = thermo.to_wilhoit()
        # thermo.select_polynomial(thermo.Tmin.value_si).Tmin.value_si = old_lowT  # put it back
        self.thermo_dict[chemkin_label].E0 = new_thermo.E0

        enthalpy_discrepancy = self.get_enthalpy_discrepancy(chemkin_label, rmg_species)
        logging.info("Storing match: {0} = {1!s}".format(chemkin_label, rmg_species))
        logging.info("  On match, Enthalpies at 298K differ by {0:.1f} kJ/mol".format(enthalpy_discrepancy))
        if abs(enthalpy_discrepancy) > 300:
            logging.warning("Very large thermo difference for identified species {0}".format(chemkin_label))
        
        # Save to local database
        if self.vote_db_enabled:
            try:
                self.vote_db.save_identified_species(
                    job_id=self.job_id,
                    chemkin_label=chemkin_label,
                    rmg_species=rmg_species,
                    identified_by=self.identified_by.get(chemkin_label, 'Auto'),
                    identification_method='vote',
                    enthalpy_discrepancy=enthalpy_discrepancy
                )
                logging.info(f"✓ Saved identified species to database: {chemkin_label}")
                
                # Clear related votes
                cleared = self.vote_db.clear_votes(
                    job_id=self.job_id,
                    chemkin_labels=[chemkin_label],
                    rmg_species_indices=[rmg_species.index]
                )
                logging.info(f"Cleared {cleared} local votes for {chemkin_label}")
            except Exception as e:
                logging.error(f"Error saving identified species: {e}")
        
        display(rmg_species)
        self.move_species_drawing(rmg_species)

        duplicate = False
        if rmg_species.label in self.species_dict_rmg:
            other_species = self.species_dict_rmg[rmg_species.label]
            if other_species is rmg_species:
                logging.warning(("This RMG species has already been matched to the chemkin label"
                                 " {0}").format(other_species.label))
                duplicate = other_species.label
                self.identified_unprocessed_labels.remove(chemkin_label)
            else:
                logging.warning(("Coincidence that RMG made a species with the same label "
                                 "as some other chemkin species: {0}.").format(rmg_species.label))
        if duplicate:
            logging.warning(("Will not rename the RMG species with duplicate chemkin labels;"
                             "leaving as it's first match: {0}").format(rmg_species.label))
        else:
            rmg_species.label = chemkin_label

        self.species_dict_rmg[chemkin_label] = rmg_species

        with open(self.dictionary_file, 'a') as f:
            f.write("{0}\t{1}\t{2:.1f}{3}\n".format(
                        chemkin_label,
                        rmg_species.molecule[0].to_smiles(),
                        enthalpy_discrepancy,
                        '\tDUPLICATE of ' + duplicate if duplicate else ''))

        with open(self.RMGdictionaryFile, 'a') as f:
            f.write("{2}{0}\n{1}\n\n".format(
                         chemkin_label,
                         rmg_species.molecule[0].to_adjacency_list(remove_h=True),
                        '// Warning! Duplicate of ' + duplicate + '\n' if duplicate else ''))

        self.draw_species(rmg_species)

        # Make an entry for the thermo database
        entry = Entry()
        entry.index = len(self.identified_labels)
        entry.label = chemkin_label
        source = os.path.join(self.name, self.args.thermo)

        # Look for the lowest energy resonance isomer that isn't aromatic,
        # because saving aromatic adjacency lists can cause problems downstream.
        for molecule in rmg_species.molecule:
            if not molecule.is_aromatic():
                break
        else:
            logging.warning(("All resonance isomers of {0} are aromatic?! "
                            "Unsafe saving to thermo.py file.").format(chemkin_label))
            return
        entry.item = molecule
        entry.data = thermo
        comment = getattr(thermo, 'comment', '')
        if comment:
            entry.long_desc = comment + '.\n'
        else:
            entry.long_desc = ''
        if duplicate:
            entry.long_desc += ("Duplicate of species {0} (i.e. same molecular structure"
                               " according to RMG)\n").format(duplicate)
        entry.long_desc += '{smiles}\n_imported from {source}.'.format(source=source,
                                                                     smiles=molecule.to_smiles())
        entry.short_desc = comment.split('\n')[0].strip()

        # store in the thermo_library
        self.thermo_library.entries[entry.label] = entry
        if self.vote_db_enabled:
            self.vote_db.update_job_statistics(
                self.job_id,
                identified_species=len(self.identified_labels),
                matched_reactions=len(self.chemkin_reactions) - len(self.chemkin_reactions_unmatched)
            )


    def get_invalidated_reactions_and_remove_votes(self, chemkin_label, rmg_species):
        """
        Remove the votes cast by/for that chemkin_label and rmg_species,
        and return the list of voting reactions that need to be re-checked.
        
        Note: ReactionPlaceholder objects (from restored votes) are filtered out
        since they cannot be rechecked - only real RMG Reaction objects are returned.
        """
        reactions_to_re_check = set()
        # First remove the chemkin_label voting dictionary
        possibles = self.votes.pop(chemkin_label, {})
        for rxns in possibles.values():
            for rxn in rxns:
                # Skip ReactionPlaceholder objects - they can't be rechecked
                if not isinstance(rxn[1], ReactionPlaceholder):
                    reactions_to_re_check.add(rxn[1])
        # Then remove the rmg_species from any voting dictionaries it is in
        for ck, possibles in self.votes.items():
            for rxn in possibles.pop(rmg_species, {}):
                # Skip ReactionPlaceholder objects - they can't be rechecked
                if not isinstance(rxn[1], ReactionPlaceholder):
                    reactions_to_re_check.add(rxn[1])
        return reactions_to_re_check

    def check_reactions_for_matches(self, reactions_to_check):
        """
        Checks the given list of edge reactions for corresponding
        chemkin reactions in the self.chemkin_reactions_unmatched list.
        Updates the votes in self.votes,
        and removes fully matched things from self.chemkin_reactions_unmatched
        also prunes reactions from self.rmg_object.edge if they can't match anything ever.
        """
        chemkin_reactions_unmatched = self.chemkin_reactions_unmatched
        reactions_match = self.reactions_match
        votes = self.votes
        
        # Track reactions checked for periodic auto-save
        reactions_checked_this_call = 0

        reactions_to_prune = set()
        for edge_reaction in reactions_to_check:
            reactions_checked_this_call += 1
            edge_reaction_matches_something = False
            for chemkin_reaction in chemkin_reactions_unmatched[:]:  # iterate over a copy of the list
                self.suggested_matches = {}
                if reactions_match(edge_reaction, chemkin_reaction):
                    edge_reaction_matches_something = True
                    logging.info("Chemkin reaction     {0}\n matches RMG {1} reaction  {2}".format(
                        chemkin_reaction, edge_reaction.family, edge_reaction))
                    if self.suggested_matches:
                        logging.info(" suggesting new species match: {0!r}".format(
                           dict((l, str(s)) for (l, s) in self.suggested_matches.items())))
                    else:
                        logging.info(" suggesting no new species matches.")

                    for chemkin_label, rmg_species in self.suggested_matches.items():
                        if chemkin_label not in votes:
                            votes[chemkin_label] = {rmg_species: set([(chemkin_reaction, edge_reaction)])}
                        else:
                            if rmg_species not in votes[chemkin_label]:
                                votes[chemkin_label][rmg_species] = set([(chemkin_reaction, edge_reaction)])
                            else:
                                votes[chemkin_label][rmg_species].add((chemkin_reaction, edge_reaction))
                        # now votes is a dict of dicts of lists of tuples {'ch3':{<Species CH3>: [ voting_reactions ]}}
            if not edge_reaction_matches_something:
                reactions_to_prune.add(edge_reaction)
        # remove those reactions
        logging.info("Removing {0} edge reactions that didn't match anything.".format(len(reactions_to_prune)))
        prune = self.rmg_object.reaction_model.edge.reactions.remove
        for rxn in reactions_to_prune:
            try:
                prune(rxn)
            except ValueError:
                logging.info("Reaction {0!s} was not in edge! Could not remove it.".format(rxn))

        for chemkin_reaction in self.chemkin_reactions_unmatched[:]:  # iterate over a copy
            if self.reagents_are_all_identified(chemkin_reaction):
                # remove it from the list of useful unmatched reactions.
                self.chemkin_reactions_unmatched.remove(chemkin_reaction)
                self.chemkin_reactions_to_save.append(chemkin_reaction)

        # Update total reaction count
        if hasattr(self, 'reactions_checked_count'):
            self.reactions_checked_count += reactions_checked_this_call
            
            # Periodic auto-save: Save every N reactions
            if self.reactions_checked_count - self.last_save_count >= self.save_interval:
                logging.info(f"Periodic auto-save triggered (checked {self.reactions_checked_count} reactions)")
                self.save_votes_to_storage(force=True)
                self.last_save_count = self.reactions_checked_count
        
        # Always save after each batch for safety
        self.save_votes_to_storage(force=False)

    def save_libraries(self):
        """
        Save the thermo and kinetics libraries in new and old format
        """
        self.save_py_thermo_library()
        self.save_py_kinetics_library()

    def save_py_thermo_library(self):
        "Save an RMG-Py style thermo library"
        library_path = os.path.join(self.output_path, 'RMG-Py-thermo-library')
        make_or_empty_directory(library_path)
        self.thermo_library.save(os.path.join(library_path, 'ThermoLibrary.py'))

    def save_py_kinetics_library(self):
        "Save an RMG-Py style kinetics library"
        library_path = os.path.join(self.output_path, 'RMG-Py-kinetics-library')
        make_or_empty_directory(library_path)
        self.kinetics_library.check_for_duplicates(mark_duplicates=True)
        self.kinetics_library.convert_duplicates_to_multi()
        self.kinetics_library.save(os.path.join(library_path, 'reactions.py'))
        for species in self.species_list:
            if species.molecule:
                species.molecule[0].clear_labeled_atoms()  # don't want '*1' labels in the dictionary
        self.kinetics_library.save_dictionary(os.path.join(library_path, 'dictionary.txt'))

        saved_reactions = [self.kinetics_library.entries[key].item 
                          for key in sorted(self.kinetics_library.entries.keys())
                          ]
        
        with open(os.path.join(library_path, 'reversed_rates.txt'), 'w') as out_file:
            for reaction in saved_reactions:
                out_file.write("Forwards reaction:   {!s}".format(reaction.to_chemkin(species_list=self.species_list)))
                out_file.write("Forwards rate:       {!r}\n".format(reaction.kinetics))
                try:
                    reverse_rate = reaction.generate_reverse_rate_coefficient()
                except (rmgpy.reaction.ReactionError, AttributeError, ValueError):
                    out_file.write("Couldn't reverse reaction rate of type {}\n\n".format(type(reaction.kinetics)))
                except np.linalg.LinAlgError:
                    out_file.write("LinAlgError trying to reverse reaction rate of type {}\n\n".format(type(reaction.kinetics)))
                else:
                    reaction.reactants, reaction.products = reaction.products, reaction.reactants
                    reaction.kinetics, reverse_rate = reverse_rate, reaction.kinetics
                    out_file.write("Reversed reaction:   {!s}".format(reaction.to_chemkin(species_list=self.species_list)))
                    out_file.write("Reversed rate:       {!r}\n\n".format(reaction.kinetics))
                    reaction.reactants, reaction.products = reaction.products, reaction.reactants
                    reaction.kinetics, reverse_rate = reverse_rate, reaction.kinetics
                out_file.write("="*80 + '\n')
            
            
        with open(os.path.join(library_path, 'unidentified_reactions.txt'), 'w') as out_file:
            out_file.write("// Couldn't use these reactions because not yet identified all species\n")
            count = 0
            for index, reaction in enumerate(self.chemkin_reactions):
                if reaction in saved_reactions:
                    continue
                count += 1
                out_file.write('//{0:4d}\n'.format(index + 1))
                out_file.write(rmgpy.chemkin.write_kinetics_entry(reaction,
                                                                species_list=self.species_list,
                                                                verbose=False,
                                                                java_library=False))
                out_file.write('\n')
            out_file.write("// Total {} reactions unidentified\n".format(count))
        
        extra_info_file_path = os.path.join(self.output_path, 'RMG-Py-kinetics-library', 'extra_info.py')
        if 'RMG_MAKE_INFO_FILES' not in os.environ:
            # By default don't write it, because it's slow (and not always helpful)
            with open(extra_info_file_path, 'w') as out_file:
                out_file.write('"To generate extra information file, '
                               'run with RMG_MAKE_INFO_FILES environment variable"')
        else:
            with open(extra_info_file_path, 'w') as out_file:
                out_file.write('"Extra information about kinetics, as a list of dicts"\n\n')
                out_file.write("info = [\n")
            for reaction in saved_reactions:
                self.save_reaction_to_kinetics_info_file(reaction, extra_info_file_path)
            with open(extra_info_file_path, 'a') as out_file:
                out_file.write(']\n\n')

    def reagents_are_all_identified(self, chemkin_reaction, require_molecules=False):
        """
        Determines if a reaction contains only species that have been identified
        by default by comparing against the list in self.identified_labels.
        If you specify require_molecules=True then the test is also that the 
        species all contain a molecule definition, i.e. they are not only in the 
        list of identified labels but also have been processed and given a molecular
        structure.        
        Returns True if all the reagents have been identified, else False.
        """
        for reagents in (chemkin_reaction.reactants, chemkin_reaction.products):
            for reagent in reagents:
                if reagent.label not in self.identified_labels:
                    return False
                if require_molecules and not reagent.molecule:
                    return False
        return True


    def add_reaction_to_kinetics_library(self, chemkin_reaction):
        """
        Add the chemkin reaction (once species are identified) to the reaction_library
        """
        entry = kin_entry()
        #source = self.args.reactions
        #entry.index = len(self.chemkin_reactions) - len(self.chemkin_reactions_unmatched)
        entry.index = self.chemkin_reactions.index(chemkin_reaction) + 1
        entry.item = chemkin_reaction
        entry.label = str(chemkin_reaction)
        entry.data = chemkin_reaction.kinetics
        comment = getattr(chemkin_reaction, 'comment', '')  # This should ideally return the chemkin file comment but currently does not
        if comment:
            entry.long_desc = comment + '.\n'
        else:
            entry.long_desc = ''
        entry.short_desc = 'The chemkin file reaction is {0}'.format(str(chemkin_reaction))

        self.kinetics_library.entries[entry.index] = entry

    def save_reaction_to_kinetics_info_file(self, chemkin_reaction, file_path):
        """
        Output to the kinetics.py information file
        """
        from arkane.output import prettify
        with open(file_path, 'a') as f:
            f.write('{\n')
            f.write(' "reaction": {!r},\n'.format(str(chemkin_reaction)))
            f.write(' "chemkin_kinetics": """\n{!s}""",\n'.format(rmgpy.chemkin.write_kinetics_entry(chemkin_reaction, self.species_list, verbose=False)))
            f.write(' "rmg_py_kinetics": {!s},\n'.format(prettify(repr(chemkin_reaction.kinetics))))
            f.write(' "possible_reaction_families": [')
            reactant_molecules = [s.molecule[0] for s in chemkin_reaction.reactants if s.reactive]
            product_molecules = [s.molecule[0] for s in chemkin_reaction.products if s.reactive]
            f.flush()
            # logging.info("Trying to generate reactions for " + str(chemkin_reaction))
            try:
                generated_reactions = self.rmg_object.database.kinetics.generate_reactions_from_families(reactant_molecules, product_molecules)
            except KineticsError as e:
                f.write('{0!r}'.format('Bug!: ' + str(e)))
                generated_reactions = []
            for reaction in generated_reactions:
                f.write('{0!r}, '.format(reaction.family))
            f.write(' ],\n')
            f.write('},\n\n')
            del generated_reactions
        return True


    def prune_voting(self):
        """
        Return a pruned voting matrix with only significant (unique) votes,
        for making tentative matches.
        
        If the same reaction is voting for several species, remove it.
        If a match has a large enthalpy discrepancy, remove it.
        """
        votes = self.votes
        pruned_votes = {}

        # votes matrix containing sets with only the chemkin reactions, not the corresponding RMG reactions
        ck_votes = dict()
        for chemkin_label, possible_matches in votes.items():
            ck_votes[chemkin_label] = dict(
                    (matching_species, set([r[0] for r in voting_reactions]))
                    for (matching_species, voting_reactions) in possible_matches.items()
                   )

        for chemkin_label, possible_matches in ck_votes.items():
            for rmg_species in list(possible_matches.keys()):
                dH = self.get_enthalpy_discrepancy(chemkin_label, rmg_species)
                if abs(dH) > 150:
                    logging.info(("Removing possible match {0} : {1!s} "
                                  " because enthalpy discrepancy is {2:.1f} kJ/mol"
                                  ).format(chemkin_label, rmg_species, dH))
                    del(possible_matches[rmg_species])

        for chemkin_label, possible_matches in ck_votes.items():
            if len(possible_matches) == 0:
                logging.info("No remaining matches for {0}".format(chemkin_label))
                continue
            if len(possible_matches) == 1:
                pruned_votes[chemkin_label] = possible_matches
                continue
            common_votes = None
            most_votes = 0
            for matching_species, voting_reactions in possible_matches.items():
                most_votes = max(most_votes, len(voting_reactions))
                if common_votes is None:
                    common_votes = set(voting_reactions)  # make a copy!!
                else:
                    common_votes.intersection_update(voting_reactions)
            if len(common_votes) < most_votes:
                logging.info(
                    "Removing {0} voting reactions that are common to all {1} matches for {2}".format(
                        len(common_votes), len(possible_matches), chemkin_label))
                pruned_votes[chemkin_label] = dict(
                    (matching_species, voting_reactions.difference(common_votes))
                    for (matching_species, voting_reactions) in possible_matches.items()
                    if voting_reactions.difference(common_votes))
            else:
                pruned_votes[chemkin_label] = possible_matches
        self.pruned_votes = pruned_votes
        return pruned_votes

    def print_voting(self, votes):
        """
        Log the passed in voting matrix to the console.
        """
        logging.info("Current voting:::")
        chemkin_controversy = dict((label, 0) for label in votes.keys())
        rmg_controversy = {}
        flat_votes = {}
        for chemkin_label, possible_matches in votes.items():
            for matching_species, voting_reactions in possible_matches.items():
                self.draw_species(matching_species)
                flat_votes[(chemkin_label, matching_species)] = voting_reactions
                chemkin_controversy[chemkin_label] += len(voting_reactions)
                rmg_controversy[matching_species] = rmg_controversy.get(matching_species, 0) + len(voting_reactions)

        for chemkin_label in sorted(chemkin_controversy.keys(), key=lambda label:-chemkin_controversy[label]):
            possible_matches = votes[chemkin_label]
            logging.info("{0} matches {1} RMG species:".format(chemkin_label, len(possible_matches)))
            for matching_species in sorted(possible_matches.keys(), key=lambda species:-len(possible_matches[species])) :
                voting_reactions = possible_matches[matching_species]
                logging.info("  {0}  matches  {1!s}  according to {2} reactions:".format(chemkin_label, matching_species, len(voting_reactions)))
                logging.info("  Enthalpies at 298K differ by {0:.1f} kJ/mol".format(self.get_enthalpy_discrepancy(chemkin_label, matching_species)))
                display(matching_species)
                for rxn in voting_reactions:
                    if isinstance(rxn, tuple):
                        logging.info("    {0!s}     //    {1!s}".format(rxn[0], rxn[1]))
                    else:
                        logging.info("    {0!s}".format(rxn))

    def constrain_reaction_families(self):
        """
        Add restraints to the reaction families so they do not produce 
        edge species that cannot possibly be in the chemkin file.
        """
        import rmgpy.data.rmg
        old_is_molecule_forbidden = rmgpy.data.rmg.database.forbidden_structures.is_molecule_forbidden
        chemkin_formulas = set(self.formula_dict.values())

        def new_is_molecule_forbidden(molecule):
            # return True (Forbidden) if we forbid it,
            if molecule.get_formula() not in chemkin_formulas:
                return True
            # otherwise return whatever we would have returned
            return old_is_molecule_forbidden(molecule)

        rmgpy.data.rmg.database.forbidden_structures.is_molecule_forbidden = new_is_molecule_forbidden

    def limit_enlarge(self, new_object):
        """
        Enlarges the rmg reaction model, but only reacts the new species with
        species it is predicted to react with in the chemkin reaction file
        
        Follows a similar procedure to rmg.CoreEdgeReactionModel.enlarge
        """
        rm = self.rmg_object.reaction_model

        import rmgpy.data.rmg
        import itertools
        from rmgpy.rmg.pdep import PDepNetwork
        from rmgpy.kinetics import KineticsData
        from rmgpy.data.kinetics.family import TemplateReaction
        from rmgpy.data.kinetics.depository import DepositoryReaction

        database = rmgpy.data.rmg.database

        obj = new_object

        num_old_core_species = len(rm.core.species)
        num_old_core_reactions = len(rm.core.reactions)
        num_old_edge_species = len(rm.edge.species)
        num_old_edge_reactions = len(rm.edge.reactions)
        reactions_moved_from_edge = []
        new_reaction_list = []
        new_species_list = []

        rm.new_reaction_list = []
        rm.new_species_list = []
        new_reactions = []
        pdep_network = None
        object_was_in_edge = False

        new_species = obj

        object_was_in_edge = new_species in rm.edge.species

        if not new_species.reactive:
            logging.info('NOT generating reactions for unreactive species {0}'.format(new_species))
        else:
            logging.info('Adding species {0} to model core'.format(new_species))

            # Find reactions involving the new species as unimolecular reactant
            # or product (e.g. A <---> products)
            try:
                new_reactions.extend([i for l in rmgpy.rmg.react.react([((new_species,),)]) for i in l])
                # the [i for l in thing for i in l] flattens the list of lists
            except KineticsError as e:
                logging.error(str(e))
                logging.error("Not reacting {0!r} on its own".format(new_species))
                
            # Find reactions involving the new species as bimolecular reactants
            # or products with other core species (e.g. A + B <---> products)
            # This is the primary difference from a standard enlarge, where
            # normally it would react with all things in the core, this just
            # finds reactions in the chemkin file and creates those
            for core_species in rm.core.species:
                if core_species.reactive:
                    if self.species_react_according_to_chemkin(new_species, core_species):
                        try:
                            new_reactions.extend([i for l in rmgpy.rmg.react.react([((new_species, core_species),)]) for i in l])
                        except KineticsError as e:
                            logging.error(str(e))
                            logging.error("Not reacting {0!r} with {1!r}".format(new_species, core_species))
            # Find reactions involving the new species as bimolecular reactants
            # or products with itself (e.g. A + A <---> products)
            # This is also limited to only reactions that occur in the chemkin file.
            if self.species_react_according_to_chemkin(new_species, new_species):
                try:
                    new_reactions.extend([i for l in rmgpy.rmg.react.react([((new_species, new_species.copy(deep=True)),)]) for i in l])
                except KineticsError as e:
                    logging.error(str(e))
                    logging.error("Not reacting {0!r} with itself".format(new_species))

        # Add new species
        reactions_moved_from_edge = rm.add_species_to_core(new_species)

        # Process the new reactions
        # While adding to core/edge/pdep network, this clears atom labels:
        rm.process_new_reactions(new_reactions, new_species, pdep_network)
        # this will call rm.check_for_existing_species to see if it already
        # exists in rm.species_dict and if not there, will add to rm.new_species_list
        # and call .generate_resonance_structures on each Species.

        if object_was_in_edge:
            # moved one species from edge to core
            num_old_edge_species -= 1
            # moved these reactions from edge to core
            num_old_edge_reactions -= len(reactions_moved_from_edge)

        new_species_list.extend(rm.new_species_list)
        new_reaction_list.extend(rm.new_reaction_list)

        # Generate thermodynamics of new species
        logging.info('Generating thermodynamics for new species...')
        for spec in new_species_list:
            try:
                spec.thermo = generate_thermo_data(spec)
            except:
                logging.exception("Error generating thermo for species:\n{0!s}".format(spec.to_adjacency_list()))
                if self.rmg_object.quantum_mechanics:
                    logging.info("Trying again without QM")
                    qm = self.rmg_object.quantum_mechanics # save for later
                    self.rmg_object.quantum_mechanics = None
                    spec.thermo = generate_thermo_data(spec)
                    self.rmg_object.quantum_mechanics = qm # restore original setting
        # Generate kinetics of new reactions
        logging.info('Generating kinetics for new reactions...')
        for reaction in new_reaction_list:
            # If the reaction already has kinetics (e.g. from a library),
            # assume the kinetics are satisfactory
            if reaction.kinetics is None:
                # Set the reaction kinetics
                kinetics, source, entry, is_forward = rm.generate_kinetics(reaction)
                reaction.kinetics = kinetics
                # Flip the reaction direction if the kinetics are defined in the reverse direction
                if not is_forward:
                    reaction.reactants, reaction.products = reaction.products, reaction.reactants
                    reaction.pairs = [(p, r) for r, p in reaction.pairs]
                if rmgpy.rmg.model.get_family_library_object(reaction.family).own_reverse and hasattr(reaction, 'reverse'):
                    if not is_forward:
                        reaction.template = reaction.reverse.template
                    # We're done with the "reverse" attribute, so delete it to save a bit of memory
                    delattr(reaction, 'reverse')

        # For new reactions, convert ArrheniusEP to Arrhenius, and fix barrier heights.
        # rm.new_reaction_list only contains *actually* new reactions, all in the forward direction.
        for reaction in new_reaction_list:
            # convert KineticsData to Arrhenius forms
            if isinstance(reaction.kinetics, KineticsData):
                reaction.kinetics = reaction.kinetics.to_arrhenius()
            #  correct barrier heights of estimated kinetics
            if isinstance(reaction, TemplateReaction) or isinstance(reaction, DepositoryReaction):  # i.e. not LibraryReaction
                reaction.fix_barrier_height()  # also converts ArrheniusEP to Arrhenius.

            if rm.pressure_dependence and reaction.is_unimolecular():
                # If this is going to be run through pressure dependence code,
                # we need to make sure the barrier is positive.
                reaction.fix_barrier_height(force_positive=True)

        # Check new core reactions for Chemkin duplicates
        new_core_reactions = rm.core.reactions[num_old_core_reactions:]
        checked_core_reactions = rm.core.reactions[:num_old_core_reactions]
        from rmgpy.chemkin import mark_duplicate_reaction
        for rxn in new_core_reactions:
            mark_duplicate_reaction(rxn, itertools.chain(checked_core_reactions, rm.output_reaction_list))
            checked_core_reactions.append(rxn)

        rm.log_enlarge_summary(
            new_core_species=rm.core.species[num_old_core_species:],
            new_core_reactions=rm.core.reactions[num_old_core_reactions:],
            reactions_moved_from_edge=reactions_moved_from_edge,
            new_edge_species=rm.edge.species[num_old_edge_species:],
            new_edge_reactions=rm.edge.reactions[num_old_edge_reactions:]
        )

        logging.info('')

    def minimal(self):
        """
        This does a minimal reading of the chemkin files just to detect errors in them.
        """
        args = self.args
        species_file = args.species
        reactions_file = args.reactions or species_file
        thermo_file = args.thermo
        known_species_file = args.known or species_file + '.SMILES.txt'
        self.known_species_file = known_species_file
        self.blocked_matches_file = os.path.splitext(known_species_file)[0] + '-BLOCKED.txt'
        self.output_path = os.path.dirname(os.path.abspath(reactions_file))

        self.load_species(species_file)
        self.load_thermo(thermo_file)
        self.load_known_species(known_species_file)

        for species in self.species_list:
            if species.label not in self.thermo_dict or self.thermo_dict[species.label] is None:
                message = ("Species {sp} in the species file {spf} does not have a valid thermo entry "
                           "in the thermo file {th}").format(sp=species.label, spf=species_file, th=thermo_file)
                logging.error(message)
                raise Exception(message)

        self.load_reactions(reactions_file)


    # NOTE: API sync methods removed
    # Django dashboard syncs data via SSH (see incremental_sync.py)
    # No need for manual sync/pull - Django pulls updates automatically
    
    def export_votes_to_json(self, output_file=None):
        """Export all vote data to JSON file for backup/sharing"""
        if not self.vote_db_enabled:
            logging.warning("Vote database not enabled")
            return False
        
        if output_file is None:
            output_file = f"votes_{self.job_id}.json"
        
        logging.info(f"Exporting votes to {output_file}...")
        try:
            success = self.vote_db.export_to_json(self.job_id, output_file)
            if success:
                logging.info(f"✓ Exported to {output_file}")
                return True
            else:
                logging.error("✗ Export failed")
                return False
        except Exception as e:
            logging.error(f"✗ Export error: {e}")
            return False



    def main(self):
        """This is the main matcher function that does the whole thing"""
        args = self.args
        species_file = args.species
        reactions_file = args.reactions or species_file
        thermo_file = args.thermo
        known_species_file = args.known or species_file + '.SMILES.txt'
        self.known_species_file = known_species_file
        self.blocked_matches_file = os.path.splitext(known_species_file)[0] + '-BLOCKED.txt'
        self.output_path = os.path.dirname(os.path.abspath(reactions_file))
        
        # Initialize counters for periodic auto-save
        self.reactions_checked_count = 0
        self.last_save_count = 0
        self.save_interval = 100  # Save every 100 reactions checked

        self.load_species(species_file)
        self.load_thermo(thermo_file)
        self.load_known_species(known_species_file)
        
        for species in self.species_list:
            if species.label not in self.thermo_dict or self.thermo_dict[species.label] is None:
                message = ("Species {sp} in the species file {spf} does not have a valid thermo entry "
                           "in the thermo file {th}").format(sp=species.label, spf=species_file, th=thermo_file)
                logging.error(message)
                raise Exception(message)

        logging.info("Initializing RMG")
        self.initializeRMG(args)
        rm = self.rmg_object.reaction_model
        self.dictionary_file = os.path.join(args.output_directory, 'MatchedSpeciesDictionary.txt')
        self.RMGdictionaryFile = os.path.join(args.output_directory, 'Original_RMG_dictionary.txt')

        with open(self.dictionary_file, 'w') as f:
            f.write("Species name\tSMILES\t_enthaply discrepancy at 298K\n")
        with open(self.RMGdictionaryFile, 'w') as f:
            f.write("\n")
        try:
            with open('source.txt', encoding='utf-8', errors='replace') as f:
                source = f.read()
                source = str(source.encode('ascii', 'replace'))
                print(source)
        except IOError:
            source = "Unknown source"

        self.thermo_library = rmgpy.data.thermo.ThermoLibrary(
            label=thermo_file.replace('"', ''),
            name=self.name,
            solvent=None,
            short_desc=os.path.abspath(thermo_file).replace('"', ''),
            long_desc=source.strip(),
            )

        self.kinetics_library = rmgpy.data.kinetics.KineticsLibrary(
            label=reactions_file.replace('"', ''),
            name=self.name,
            solvent=None,
            short_desc=os.path.abspath(reactions_file).replace('"', ''),
            long_desc=source.strip(),
            )

        # Show initial sync status
        self.show_sync_status()

        # Load previous votes from storage (resume capability!)
        self.load_votes_from_storage()
        self.load_blocked_matches()

        self.identify_small_molecules()

        self.check_thermo_libraries()

        logging.info("Importing identified species into RMG model")
        # Add identified species to the reaction model complete species list
        new_species_dict = {}
        for species_label in self.identified_labels:
            old_species = self.species_dict[species_label]
            logging.info(species_label)
            rmg_species, was_new = rm.make_new_species(old_species, label=old_species.label)
            if not was_new:
                logging.warning("Species with structure of '{0}' already created with label '{1}'".format(species_label, rmg_species.label))

            new_species_dict[species_label] = rmg_species
            if self.formula_dict[species_label] in ['N2', 'Ar', 'He']:
                rmg_species.reactive = False
                old_species.reactive = False
                # when this occurs in collider lists it's still the old species?
            rmg_species.generate_resonance_structures()
            try:
                rmg_species.thermo = generate_thermo_data(rmg_species)
            except:
                logging.error("Couldn't generate thermo for RMG species {}".format(rmg_species))
                raise
        # Set match using the function to get all the side-effects.
        labels_to_process = self.identified_labels
        self.identified_labels = []
        for chemkin_label in labels_to_process:
            # this adds it back into self.identified_labels
            self.set_match(chemkin_label, new_species_dict[chemkin_label])

        chemkin_formulas = set(self.formula_dict.values())

        self.load_reactions(reactions_file)
        if self.vote_db_enabled:
            self.vote_db.update_job_statistics(
            self.job_id,
            total_species=len(self.species_list),
            total_reactions=len(self.chemkin_reactions)
        )
            
        self.save_reactions_to_storage()
        
        # ... rest of import process ...
        
        # At end, show final status
        # Optionally export backup
        if self.vote_db_enabled:
            self.export_votes_to_json()
            logging.info("Vote database saved. Django will sync via SSH.")
        
        chemkin_reactions_unmatched = self.chemkin_reactions_unmatched
        votes = self.votes

        # Now would be a good time to save identified reactions?
        # All the species in self.identified_labels should have been through generate_resonance_structures and generate_thermo_data
        for chemkin_reaction in chemkin_reactions_unmatched[:]:  # iterate over a copy of the list, so you can modify the list itself
            if self.reagents_are_all_identified(chemkin_reaction):
                chemkin_reactions_unmatched.remove(chemkin_reaction)
                assert self.reagents_are_all_identified(chemkin_reaction, require_molecules=True)
                self.add_reaction_to_kinetics_library(chemkin_reaction)

        self.save_libraries()

        self.prune_inert_species()

        # Let's reduce the number of edge reactions producing things that can't possibly match
        self.constrain_reaction_families()

        # Let's put things in the core by size, smallest first.
        self.identified_unprocessed_labels.sort(key=lambda x: new_species_dict[x].molecular_weight.value_si)
        # We want to put inert things in the core first, so we can do PDep calculations with inert colliders.
        self.identified_unprocessed_labels.sort(key=lambda x: new_species_dict[x].reactive)
        reactions_to_check = set()
        
        # RESTART OPTIMIZATION: Get list of already-processed species from database
        # These species already had limit_enlarge() run, so we skip the expensive computation
        already_processed = getattr(self, 'already_processed_labels', set())
        if already_processed:
            logging.info(f"RESTART OPTIMIZATION: {len(already_processed)} species already processed")
            logging.info(f"  Will skip limit_enlarge() for these species (major speedup!)")
        
        while self.identified_unprocessed_labels:

            label_to_process = self.identified_unprocessed_labels.pop(0)
            
            # RESTART OPTIMIZATION: Skip limit_enlarge() if species was already processed
            if label_to_process in already_processed:
                logging.info(f"SKIPPING already-processed species: {label_to_process}")
                logging.info(f"  (limit_enlarge already completed in previous run)")
                continue
            
            logging.info("Processing species {0}...".format(label_to_process))

            # Add species to RMG core.
            self.limit_enlarge(self.species_dict_rmg[label_to_process])

            # do a partial prune of new reactions that definitely aren't going to be useful
            reactions_to_prune = set()
            for new_species in rm.new_species_list:
                if new_species.molecule[0].get_formula() in chemkin_formulas:
                    # This allows us to extrapolating H to 298 for comparison
                    thermo = new_species.thermo
                    old_lowT = thermo.Tmin.value_si
                    if old_lowT > 298.0:
                        thermo.select_polynomial(thermo.Tmin.value_si).Tmin.value_si = min(298.0, thermo.Tmin.value_si)
                        thermo.Tmin.value_si = min(298.0, thermo.Tmin.value_si)
                        thermo.comment += "\n_extrapolated from Tmin={0} to {1} for comparison.".format(old_lowT, 298.0)
                        logging.warning ("Changing Tmin from {0} to {1} for RMG-generated thermo for {2}".format(old_lowT, 298.0, new_species))
                    continue
                # else it's not useful to us
                # identify any reactions it's involved in
                for rxn in rm.new_reaction_list:
                    if new_species in rxn.reactants or new_species in rxn.products:
                        reactions_to_prune.add(rxn)
            logging.info("Removing {0} edge reactions that aren't useful".format(len(reactions_to_prune)))
            # this should only be library reactions, because we prevented reaction families from making un-helpful things
            # remove those reactions
            for rxn in reactions_to_prune:
                try:
                    rm.edge.reactions.remove(rxn)
                except ValueError:
                    pass  # "It wasn't in the edge. Presumably leaking from a pdep network"
                rm.new_reaction_list.remove(rxn)
            reactions_to_prune.clear()

            logging.info("Adding {0} new RMG reactions to be checked.".format(len(rm.new_reaction_list)))
            reactions_to_check.update(rm.new_reaction_list)
            logging.info("In total will check {0} edge reactions".format(len(reactions_to_check)))
            logging.info("against {0} unmatched chemkin reactions.".format(len(chemkin_reactions_unmatched)))

            if len(self.identified_unprocessed_labels) == 0:
                logging.info("** Running out of things to process!")

            while reactions_to_check:
                self.check_reactions_for_matches(reactions_to_check)
                # Have just checked all those reactions, so clear the reactions_to_check,
                # ready to start adding to it again based on new matches.
                reactions_to_check.clear()

                # self.print_voting(votes)
                pruned_votes = self.prune_voting()
                # self.print_voting(pruned_votes)

                self.draw_all_candidate_species()

                new_matches = []
                for chemkin_label, possible_matches in pruned_votes.items():
                    if len(possible_matches) == 1:
                        matching_species, voting_reactions = list(possible_matches.items())[0]
                        logging.info("\n_only one suggested match for {0}: {1!s}".format(chemkin_label, matching_species))
                        display(matching_species)
                        logging.info("With {0} unique voting reactions:".format(len(voting_reactions)))
                        for reaction in voting_reactions:
                            logging.info("  {0!s}".format(reaction))
                        all_possible_chemkin_species = [ck for ck, matches in pruned_votes.items() if matching_species in matches]
                        if len(all_possible_chemkin_species) == 1:
                            logging.info("Only one chemkin species has this match (after pruning).")
                            self.set_tentative_match(chemkin_label, matching_species)
                            #new_matches.append((chemkin_label, matching_species))
                        else:
                            logging.info("Other Chemkin species that also match {0} (after pruning) are {1!r}".format(matching_species.label, all_possible_chemkin_species))
                            logging.info("Will not make match at this time.")

                for chemkin_label, matching_species in new_matches:
                    invalidated_reactions = self.get_invalidated_reactions_and_remove_votes(
                        chemkin_label, matching_species)
                    reactions_to_check.update(invalidated_reactions)
                logging.info("After making {0} matches, will have to re-check {1} edge reactions".format(len(new_matches), len(reactions_to_check)))

            if self.vote_db_enabled:
                self.update_reaction_status_in_storage()
            # self.save_votes_to_storage(force=False)

            logging.info("Finished processing species {0}!".format(
                            label_to_process))
            
            # MARK SPECIES AS PROCESSED in database
            # This enables restart optimization - on next run, this species will be skipped
            if self.vote_db_enabled:
                try:
                    self.vote_db.mark_species_processed(self.job_id, label_to_process)
                    logging.debug(f"Marked {label_to_process} as processed in database")
                except Exception as e:
                    logging.warning(f"Failed to mark {label_to_process} as processed: {e}")
            
            logging.info("Have now identified {0} of {1} species ({2:.1%}).".format(
                            len(self.identified_labels),
                            len(self.species_list),
                            float(len(self.identified_labels)) / len(self.species_list)))
            logging.info("And fully identified {0} of {1} reactions ({2:.1%}).".format(
                            len(self.chemkin_reactions) - len(self.chemkin_reactions_unmatched),
                            len(self.chemkin_reactions),
                            1 - float(len(self.chemkin_reactions_unmatched)) / len(self.chemkin_reactions)))
            logging.info("Still to process {0} matches: {1!r}".format(
                            len(self.identified_unprocessed_labels),
                            self.identified_unprocessed_labels))

            logging.info("Saving chemkin files")
            rmgpy.chemkin.save_chemkin(rm,
                               os.path.join(self.rmg_object.output_directory, 'identified_chemkin.txt'),
                               os.path.join(self.rmg_object.output_directory, 'identified_chemkin_verbose.txt'),
                               os.path.join(self.rmg_object.output_directory, 'identified_RMG_dictionary.txt'))

            while len(self.identified_unprocessed_labels) == 0:
                if not self.manual_matches_to_process :
                    logging.info("Updating exported library files...")
                    self.save_libraries()

                    if self.args.quit_when_exhausted:
                        logging.warning("--quit_when_exhausted option detected."
                                        " Now exiting without waiting for input.")
                        break
                    logging.info(("Waiting for input from the web front end..."
                                 " (port {0})").format(self.args.port))
                while not self.manual_matches_to_process:
                    time.sleep(1)

                while self.manual_matches_to_process:
                    chemkin_label, matching_species = self.manual_matches_to_process.pop(0)
                    logging.info("There is a manual match to process: {0} is {1!s}".format(
                                    chemkin_label, matching_species))
                    if chemkin_label in self.identified_labels:
                        assert self.species_dict_rmg[chemkin_label] == matching_species, \
                            "Manual match disagrees with an automatic match!"
                        continue  # don't match something that's already matched.
                    self.set_match(chemkin_label, matching_species)
                    invalidated_reactions = self.get_invalidated_reactions_and_remove_votes(
                                                        chemkin_label, matching_species)
                    reactions_to_check.update(invalidated_reactions)
                    logging.info(("After making that match, "
                        "will have to re-check {0} edge reactions").format(len(reactions_to_check)))

                #After processing all matches, now is a good time to save reactions.
                couldnt_save = []
                while self.chemkin_reactions_to_save:
                    chemkin_reaction = self.chemkin_reactions_to_save.pop(0)
                    if self.reagents_are_all_identified(chemkin_reaction, require_molecules=True):
                        self.add_reaction_to_kinetics_library(chemkin_reaction)
                    else:
                        couldnt_save.append(chemkin_reaction)
                self.chemkin_reactions_to_save = couldnt_save  # try again later!

            terminal_input_enabled = False
            if (len(self.identified_unprocessed_labels) == 0
            and self.votes
            and terminal_input_enabled):
                self.print_voting(pruned_votes)
                logging.info("Run out of options. Asking for help!")
                species_label = input('Which label would you like to identify? (see voting info above)\n')
                while True:
                    if species_label not in self.formula_dict:
                        print("That's not a known species label")
                    elif species_label in self.identified_labels:
                        print("That's already been identified")
                    elif species_label not in votes:
                        print("We have no candidate matches for that label.")
                    else:  # label is valid, break out of while loop.
                        break
                    species_label = input("Try again:\n")
                possible_matches = list(votes[species_label].keys())
                chemkin_label, matching_species = self.ask_for_match_id(species_label, possible_matches)
                invalidated_reactions = self.get_invalidated_reactions_and_remove_votes(chemkin_label, matching_species)
                reactions_to_check.update(invalidated_reactions)
                logging.info("After making that match, will have to re-check {0} edge reactions".format(len(reactions_to_check)))

        # POST-LOOP WAITING: Handle case where all processed species were skipped
        # but there are still unidentified species that need user votes
        unidentified_count = len(self.species_list) - len(self.identified_labels)
        if unidentified_count > 0 and self.votes:
            logging.info("\n" + "="*60)
            logging.info(f"WAITING FOR USER INPUT: {unidentified_count} species still unidentified")
            logging.info(f"  {len(self.votes)} species have candidate matches needing votes")
            logging.info(f"  Web interface available at http://0.0.0.0:{self.args.port}")
            logging.info("="*60)
            
            # Draw all candidate species images for the web interface
            # This is needed because on restart, the main loop's draw_all_candidate_species() may be skipped
            logging.info("Drawing candidate species images for web interface...")
            self.draw_all_candidate_species()
            
            if not self.args.quit_when_exhausted:
                self.save_libraries()
                logging.info("Waiting for input from the web front end... (port {0})".format(self.args.port))
                
                # Wait for manual matches from web interface
                while unidentified_count > 0:
                    while not self.manual_matches_to_process:
                        time.sleep(1)
                    
                    # Process manual matches
                    while self.manual_matches_to_process:
                        chemkin_label, matching_species = self.manual_matches_to_process.pop(0)
                        logging.info("There is a manual match to process: {0} is {1!s}".format(
                                        chemkin_label, matching_species))
                        if chemkin_label in self.identified_labels:
                            assert self.species_dict_rmg[chemkin_label] == matching_species, \
                                "Manual match disagrees with an automatic match!"
                            continue  # don't match something that's already matched.
                        self.set_match(chemkin_label, matching_species)
                        
                        # Add to identified_unprocessed_labels for processing
                        if chemkin_label not in self.identified_unprocessed_labels:
                            self.identified_unprocessed_labels.append(chemkin_label)
                        
                        invalidated_reactions = self.get_invalidated_reactions_and_remove_votes(
                                                            chemkin_label, matching_species)
                        logging.info("After making that match, will have to re-check {0} edge reactions".format(
                                        len(invalidated_reactions)))
                    
                    # Process any newly identified species
                    while self.identified_unprocessed_labels:
                        label_to_process = self.identified_unprocessed_labels.pop(0)
                        if label_to_process in already_processed:
                            logging.info(f"SKIPPING already-processed species: {label_to_process}")
                            continue
                        
                        logging.info("Processing species {0}...".format(label_to_process))
                        self.limit_enlarge(self.species_dict_rmg[label_to_process])
                        
                        # Mark as processed
                        if self.vote_db_enabled:
                            try:
                                self.vote_db.mark_species_processed(self.job_id, label_to_process)
                            except Exception as e:
                                logging.warning(f"Failed to mark {label_to_process} as processed: {e}")
                        
                        already_processed.add(label_to_process)
                    
                    # Update count and save progress
                    unidentified_count = len(self.species_list) - len(self.identified_labels)
                    self.save_votes_to_storage(force=False)
                    logging.info(f"Progress: {len(self.identified_labels)}/{len(self.species_list)} species identified")
                    
                    if unidentified_count > 0:
                        logging.info(f"Still waiting for votes on {unidentified_count} unidentified species...")
            else:
                logging.warning("--quit_when_exhausted option detected. Exiting without waiting for input.")

        self.save_reactions_to_storage()
        print("Finished reading")
        counter = 0
        for species in self.species_list:
            counter += 1
            print(counter, species,)
            if species.label not in self.species_dict_rmg:
                print("")
                continue  # don't save unidentified species
            print("\t IDENTIFIED")
        print("done")
        
        # Final save and export at end of main()
        logging.info("\n" + "="*60)
        logging.info("MAIN LOOP COMPLETED - Saving final votes...")
        logging.info("="*60)
        self.save_votes_to_storage(force=True)
        self.export_votes_to_json()
        self.show_sync_status()
        logging.info("✓ Final votes saved successfully")

        if hasattr(self, 'vote_db') and self.vote_db:
            self.vote_db.close()

    def _img(self, species):
        """Get the html tag for the image of a species"""
        images_path = 'img'  # to serve via cherry_py
        #images_path = 'file://'+os.path.abspath(os.path.join(self.args.output_directory,'species')) # to get from disk
        return "<img src='{path}/{file!s}.png' title='{title}'>".format(file=urllib.parse.quote(str(species)), path=images_path, title=str(species))

    @cherrypy.expose
    def index(self):
        location = os.path.abspath(self.args.reactions or self.args.species)
        thermo_location = os.path.abspath(self.args.thermo)
        name = self.name
        output = [self.html_head() , """
<script>
function also_update(json) {
$('#identified_count').html("("+json.confirmed+(json.unprocessed?"; "+json.unprocessed+" unprocessed":"")+")");
$('#tentative_count').html("("+json.tentative+")");
$('#unmatchedreactions_count').html("("+json.unmatchedreactions+")");
$('#unconfirmedspecies_count').html("("+json.unconfirmed+")");
$('#thermomatches_count').html("("+json.thermomatches+")");
}
</script>
<h1>Mechanism importer: """ + name + """</h1>
<ul>
<li><a href="species.html">All species.</a> (Sorted by <a href="species.html?sort=name">name</a> or <a href="species.html?sort=formula">formula</a>.)</li>
<li><a href="identified.html">Identified species.</a> <span id="identified_count"></span></li>
<li><a href="tentative.html">Tentative Matches.</a> <span id="tentative_count"></span></li>
<li><a href="votes.html">Voting reactions list view.</a></li>
<li><a href="votes2.html">Voting reactions table view.</a></li>
<li><a href="autoconfirm.html">Autoconfirm table.</a></li>
<li><a href="unmatchedreactions.html">Unmatched reactions.</a> <span id="unmatchedreactions_count"></span></li>
<li><a href="unconfirmedspecies.html">Unconfirmed species.</a> <span id="unconfirmedspecies_count"></span></li>
<li><a href="blocked.html">Blocked matches.</a></li>
<li><a href="thermomatches.html">Unconfirmed thermodynamics matches.</a> <span id="thermomatches_count"></span></li>
<li><a href="thermomatchesmodel.html">Unconfirmed thermodynamics matches (select by model)</a></li>
<li><a href="thermolibraries.html">Loaded thermodynamics libraries.</a></li>
<li><a href="ThermoLibrary.py">Download thermo library.</a></li>
</ul>
        """]
        
        output.append("""Your name: <a href="setname.html">{0}</a><br/>""".format(self.get_username()))
        output.append("""Model: <a href="chemkin.inp">{0}</a><br/>""".format(location))
        output.append("""Thermo: <a href="therm.dat">{0}</a><br/>""".format(thermo_location))
        output.append(self.html_tail)
        return "\n".join(output)

    @cherrypy.expose
    def blocked_html(self):
        img = self._img
        blocked_matches = self.blocked_matches
        output = [self.html_head()]

        count = sum([len(blocks) for label, blocks in blocked_matches.items()])
        output.append('<h1>{0} Blocked Matches</h1><table style="width:500px">'.format(count))

        blocked_labels = sorted(blocked_matches.keys())
        for ck_label in blocked_labels:
            blocks = blocked_matches[ck_label]
            for rmg_species, username in blocks.items():
                output.append("<tr><td>{label}</td><td>{img}</td><td>{user}</td></tr>".format(
                                img=img(rmg_species),
                                label=ck_label,
                                user=(username or '-')
                                ))
        output.append('</table>')
        output.append(self.html_tail)
        return '\n'.join(output)
        
    @cherrypy.expose
    def identified_html(self):
        img = self._img
        def safe_img(label):
            spec = self.species_dict_rmg.get(label)
            if spec is None:
                return "<span class='badmatch'>[missing]</span>"
            return img(spec)
        return (self.html_head() + 
        f"""<h1>{len(self.identified_labels)} Identified Species</h1>
        <form action="/deletemistakes.html" method="get">
        <table style="width:500px"><tr>
        <th>#</th><th>Label</th><th>Molecule</th><th>Identified by</th><th>Delete?</th></tr>""" +
         "<tr>".join([
                        """<td>{number}</td><td>{label}</td><td>{img}</td><td>{user}</td>
                        <td><input type="checkbox" name="errors" value="{label}" /></td>
                        """.format(
                                img=safe_img(lab),
                                label=lab,
                                number=n + 1,
                                user = self.identified_by.get(lab,"-"),
                                ) for n, lab in enumerate(self.identified_labels)
                            ]) +
                """</tr>
                <tr><td colspan=4></td><td><input type="submit" value="Delete"></td></tr>
                </table></form>""" + self.html_tail)
    
    @cherrypy.expose
    def deletemistakes_html(self, errors=None):
        """
        Process the form from identified_html to remove mistaktes
        """
        if not errors:
            return "Nothing to delete"
        output = [self.html_head(), "<h1> Deleting errors</h1>", "<ol>"]
        for name in errors:
            output.append(f'<li>{name}</li>')
        
        out = []
        with open(self.known_species_file) as infile:
            for line in infile:
                try:
                    label = line.strip().split()[0]
                    if label in errors:
                        out.append(f"! Deleted by {self.get_username()}: "+line)
                        continue
                except:
                    pass
                out.append(line)
        
        with open(self.known_species_file,'w') as outfile:
            outfile.writelines(out)
                
        output.extend(['</ol>',
            "<a href='killjob.html'><button type='button'>Kill job</button></a>",
            self.html_tail])
        return ('\n'.join(output))

    @cherrypy.expose
    def killjob_html(self):
        logging.warning("Job killed by user request")
        cherrypy.engine.exit()
        os._exit(0)


    @cherrypy.expose
    def thermomatches_html(self):
        img = self._img
        output = [self.html_head(), '<h1>{0} Thermochemistry Matches</h1>'.format(len(self.thermo_matches))]
        output.append('<table style="width:800px; border-collapse:collapse;">')
        for chemkin_label, rmg_specs_dict in self.thermo_matches.items():
            label = chemkin_label
            if len(rmg_specs_dict) > 1:
                label = "<span class='badmatch'>{0}</span>".format(label)
            def format_spec(name):
                "Makes 'name' green if it matches 'label'"
                if name.upper() == chemkin_label.upper():
                    return "<span class='goodmatch'>{0}</span>".format(name)
                else:
                    return "<span class='badmatch'>{0}</span>".format(name)

            first_of_this_name = True
            for rmg_spec, libraries in rmg_specs_dict.items():
                libs = '<br>'.join(["{spec} ({lib})".format(spec=format_spec(spec), lib=lib) for (lib, spec) in libraries])
                output.append('<tr style="border-top: 1px solid black;">' if first_of_this_name else '<tr>')
                first_of_this_name = False
                output.append("<td>{label}</td>".format(label=label))
                output.append("<td>{img}</td>".format(img=img(rmg_spec)))
                output.append('<td style="border-top: 1px solid black;">{libs}</td>'.format(libs=libs))
                if chemkin_label in self.votes :
                    output.append("<td><a href='votes2.html#{0}'>check votes</a></td>".format(urllib.parse.quote(chemkin_label)))
                else:
                    output.append("<td>No votes yet.</td>")
                output.append("<td><a href='confirmthermomatch.html?ck_label={ckl}&rmg_name={rmgl}'>confirm</a></td>".format(ckl=urllib.parse.quote(chemkin_label), rmgl=urllib.parse.quote(str(rmg_spec))))
                output.append("<td><a href='clearthermomatch.html?ck_label={ckl}&rmg_name={rmgl}'>clear</a></td>".format(ckl=urllib.parse.quote(chemkin_label), rmgl=urllib.parse.quote(str(rmg_spec))))
                output.append("</tr>")
        output.extend(['</table>', self.html_tail])
        return ('\n'.join(output))

    @cherrypy.expose
    def thermomatchesmodel_html(self, model=None, confirm=None):
        if model not in self.rmg_object.database.thermo.libraries:
            output = [self.html_head(), '<h1>Select model to find Thermochemistry Matches</h1>']
            output.extend(['<form action="/thermomatchesmodel.html" method="get">', '<select name="model">'])
            for library in self.rmg_object.database.thermo.libraries.keys():
                output.append('  <option value="{lib}">{lib}</option>'.format(lib=library))
            output.extend([' </select>', '<input type="submit" />', '</form>'])
            return ('\n'.join(output)) # stop here
        img = self._img
        output = [self.html_head(), '<h1>Thermochemistry Matches with model {}</h1>'.format(model)]
        output.append('<table style="width:800px; border-collapse:collapse;">')
        to_confirm = []
        for chemkin_label, rmg_specs_dict in self.thermo_matches.items():
            for rmg_spec, libraries in rmg_specs_dict.items():
                for library_name, species_name in libraries:
                    if library_name == model and species_name == chemkin_label:
                        break
                else: # didn't break
                    continue # to next thermo match
                output.append('<tr style="border-top: 1px solid black;">')
                output.append("<td>{label}</td>".format(label=chemkin_label))
                output.append("<td>{img}</td>".format(img=img(rmg_spec)))
                if chemkin_label in self.votes :
                    output.append("<td><a href='votes2.html#{0}'>check votes</a></td>".format(urllib.parse.quote(chemkin_label)))
                else:
                    output.append("<td>No votes yet.</td>")

                if confirm == 'all':
                    to_confirm.append((chemkin_label, rmg_spec))
                    output.append("<td>Confirmed!</td>")
                else:
                    output.append("<td><a href='confirmthermomatch.html?ck_label={ckl}&rmg_name={rmgl}'>confirm</a></td>".format(ckl=urllib.parse.quote(chemkin_label), rmgl=urllib.parse.quote(str(rmg_spec))))
                    output.append("<td><a href='clearthermomatch.html?ck_label={ckl}&rmg_name={rmgl}'>clear</a></td>".format(ckl=urllib.parse.quote(chemkin_label), rmgl=urllib.parse.quote(str(rmg_spec))))
                output.append("</tr>")
        output.extend(['</table>', self.html_tail])
        
        if confirm == 'all':
            for chemkin_label, rmg_spec in to_confirm:
                self.clear_thermo_match(chemkin_label, None)
                self.manual_matches_to_process.append((str(chemkin_label), rmg_spec))
                self.clear_tentative_match(chemkin_label, None)
                self.save_match_to_file(chemkin_label, rmg_spec, username=self.get_username()+' (because it matches thermo/name in {})'.format(model))
        else:
            output.append("<a href='thermomatchesmodel.html?model={model}&confirm=all'><button>Confirm all</button></a>".format(model=model))

        return ('\n'.join(output))

    @cherrypy.expose
    def tentative_html(self):
        img = self._img
        output = [
            self.html_head(),
            '<h1>{0} Tentative Matches</h1><table style="width:800px">'.format(
                len(self.tentative_matches))
        ]
        output.append(
            "<tr><th>Name</th><th>Molecule</th><th>&Delta;H&deg;<sub>f</sub>(298K)</th><th>Matching Thermo</th></tr>")
        for match in self.tentative_matches:
            chemkin_label = match['label']
            rmg_spec = match['species']
            deltaH = match['enthalpy']
            username = match['username']
            output.append(
                "<tr><td>{label}</td><td>{img}</td><td title='{Hsource}'>{delH:.1f} kJ/mol</td>".format(
                    img=img(rmg_spec),
                    label=chemkin_label,
                    delH=deltaH,
                    Hsource=rmg_spec.thermo.comment))
            output.append("<td>")
            try:
                for library_name, library_species_name in self.thermo_matches[chemkin_label][rmg_spec]:
                    output.append("<span title='{spec}' class='{match}'>{lib}</span><br>".format(
                                        lib=library_name,
                                        spec=library_species_name,
                                        match=('goodmatch' if library_species_name.upper() == chemkin_label.upper() else 'badmatch'),
                                        ))
            except KeyError:
                output.append('-')
            output.append("</td>")
            output.append(
                "<td><a href='confirm.html?ck_label={ckl}&rmg_label={rmgl}'>confirm</a></td>".format(
                    ckl=urllib.parse.quote(chemkin_label),
                    rmgl=urllib.parse.quote(str(rmg_spec))))
            output.append(
                "<td><a href='edit.html?ck_label={ckl}&smiles={smi}'>edit</a></td>".format(
                    ckl=urllib.parse.quote(chemkin_label),
                    smi=urllib.parse.quote(rmg_spec.molecule[0].to_smiles())))
            output.append(
                "<td><a href='clear.html?ck_label={ckl}'>clear</a></td>".format(
                    ckl=urllib.parse.quote(chemkin_label)))
            output.append(
                "<td><a href='block.html?ck_label={ckl}&rmg_label={rmgl}'>block</a></td>".format(
                    ckl=urllib.parse.quote(chemkin_label),
                    rmgl=urllib.parse.quote(str(rmg_spec))))
            output.append(
                "<td><a href='votes2.html#{label}'>check {num} votes</a></td>".format(
                    label=urllib.parse.quote(chemkin_label),
                    num=len(self.votes[chemkin_label].get(rmg_spec, [])))
                if chemkin_label in self.votes else "<td>No votes yet.</td>")
            if username:
                output.append("<td>Proposed by {0}</td>".format(username))
            else:
                output.append("<td></td>")
            output.append("</tr>")
        output.extend(['</table>', self.html_tail])
        return ('\n'.join(output))

    def get_username(self):
        try:
            username = cherrypy.request.cookie['username'].value.strip()
            username = str(username.encode('ascii', 'ignore'),'ascii','replace')
            username = re.sub(r'\s+', ' ', username)
            username = re.sub(r'[^A-Za-z ]+', '_', username)
            return username
        except KeyError:
            return "Anonymous"

    @cherrypy.expose
    def setname_html(self, Name=None):
        """Save the user's name"""
        if Name:
            username = str(Name)
            cookie = cherrypy.response.cookie
            cookie['username'] = username
            cookie['username']['path'] = '/'
            cookie['username']['max-age'] = 3600 * 24 * 30
            cookie['username']['version'] = 1
            raise cherrypy.HTTPRedirect("/")
        else:
            username = self.get_username()
            output = [self.html_head()]
            output.append("<h1>Edit your name</h1>")
            output.append("""
                <form action="setname.html" method="get">
                <input type=text name="Name" value="{name}">
                <input type=submit value="Save">
                </form>
                """.format(name=username))
            output.append(self.html_tail)
            return '\n'.join(output)

    @cherrypy.expose
    def chemkin_inp(self):
        """The raw chemkin input file"""
        return serve_file(os.path.abspath(self.args.reactions or self.args.species),
                              content_type='text/plain')

    @cherrypy.expose
    def therm_dat(self):
        """The raw thermo data file"""
        return serve_file(os.path.abspath(self.args.thermo),
                              content_type='text/plain')

    @cherrypy.expose
    def unconfirmedspecies_html(self):
        output = [
            self.html_head(),
            '<h1>{0} Unconfirmed species</h1><table style="width:500px">'.format(
                len(self.species_list) - len(self.identified_labels) -
                len(self.manual_matches_to_process))
        ]
        for label in [s.label for s in self.species_list]:
            if label in self.identified_labels:
                continue
            for pair in self.manual_matches_to_process:
                if pair[0] == label:
                    continue
            output.append("<tr><td>{label}</td>".format(label=label))
            output.append(
                "<td><a href='propose.html?ck_label={ckl}'>propose match</a></td></tr>".format(
                    ckl=urllib.parse.quote(label), ))
        output.extend(['</table>', self.html_tail])
        return ('\n'.join(output))

    @cherrypy.expose
    def thermolibraries_html(self):
        "Show a list of the loaded thermo libraries"
        output = [self.html_head(), '<h1>Thermo libraries loaded</h1><ul>']

        for library_name in self.thermo_libraries_to_check:
            library = self.rmg_object.database.thermo.libraries[library_name]
            library_length = len(library.entries)
            output.append('<li>{name} ({num})</li>'.format(name=library_name, num=library_length))
        output.append('</ul>'+self.html_tail)
        return ('\n'.join(output))

    @cherrypy.expose
    def species_html(self, sort="ck"):
        img = self._img
        output = [
            self.html_head(),
            '<h1>All {0} Species</h1><table>'.format(len(self.species_list))
        ]
        tentative_dict = dict((match['label'],
                              (match['species'], match['enthalpy']))
                             for match in self.tentative_matches)
        manual_dict = dict(
            (chemkin_label, rmg_spec)
            for (chemkin_label, rmg_spec) in self.manual_matches_to_process)

        labels = [s.label for s in self.species_list]
        if sort == 'name':
            labels.sort()
            output.append('Sorted by name. Sort by <a href="/species.html">chemkin file</a> or <a href="?sort=formula">formula</a>.')
        elif sort == 'formula':
            labels.sort(key=lambda l: self.formula_dict[l])
            output.append('Sorted by formula. Sort by <a href="/species.html">chemkin file</a> or <a href="?sort=name">name</a>.')
        else:
            output.append('Sorted by chemkin file. Sort by <a href="?sort=name">name</a> or <a href="?sort=formula">formula</a>.')
        for chemkin_label in labels:
            if (chemkin_label in self.identified_labels) or (chemkin_label in manual_dict):
                try:
                    rmg_spec = self.species_dict_rmg[chemkin_label]
                    pending = False
                except KeyError:
                    rmg_spec = manual_dict[chemkin_label]
                    pending = True
                deltaH = self.get_enthalpy_discrepancy(chemkin_label, rmg_spec)
                output.append(
                    ("<tr><td class='confirmed'>{label}</td>"
                     "<td class='centered'>{img}</td>"
                     "<td>{smi}</td>"
                     "<td title='{Hsource}'>{delH:.1f} kJ/mol</td>").format(
                        img=img(rmg_spec),
                        label=chemkin_label,
                        delH=deltaH,
                        Hsource=rmg_spec.thermo.comment,
                        smi='<br/>'.join((m.to_smiles() for m in rmg_spec.molecule))
                    ))
                if chemkin_label in self.identified_unprocessed_labels:
                    output.append("<td>Identified, waiting to react.</td>")
                elif pending:
                    output.append("<td>Identified, pending processing.</td>")
                else:
                    output.append("<td>Identified, reacted, in model.</td>")
            elif chemkin_label in tentative_dict:
                rmg_spec, deltaH = tentative_dict[chemkin_label]
                output.append(
                    ("<tr><td class='tentative'>{label}</td>"
                     "<td class='centered'>{img}</td>"
                     "<td>{smi}</td>"
                     "<td title='{Hsource}'>{delH:.1f} kJ/mol</td>").format(
                        img=img(rmg_spec),
                        label=chemkin_label,
                        delH=deltaH,
                        Hsource=rmg_spec.thermo.comment,
                        smi='<br/>'.join((m.to_smiles() for m in rmg_spec.molecule))
                    ))
                output.append(
                    "<td>Tentative match. <a href='confirm.html?ck_label={ckl}&rmg_label={rmgl}'>confirm</a> / ".format(
                        ckl=urllib.parse.quote(chemkin_label),
                        rmgl=urllib.parse.quote(str(rmg_spec))))
                votes = "/ <a href='votes2.html#{0}'>check votes</a>".format(
                    urllib.parse.quote(chemkin_label)) if chemkin_label in self.votes else "No votes yet. "
                output.append(
                    "<a href='edit.html?ck_label={ckl}&smiles={smi}'>edit</a> {votes}</td></tr>".format(
                        ckl=urllib.parse.quote(chemkin_label),
                        smi=urllib.parse.quote(rmg_spec.molecule[0].to_smiles()),
                        votes=votes))
            else:
                output.append(
                    ("<tr><td class='unknown'>{label}</td>"
                     "<td class='centered'>?</td>").format(
                        label=chemkin_label))
                output.append("""
            <form action="edit.html" method="get"><td>
            <input type=hidden name="ck_label" value="{lab}">
            <input type=text name="smiles"></td>
            <td><input type=submit></td>
            </form>
            """.format(lab=chemkin_label))
                votes = "<a href='votes2.html#{0}'>check votes</a> / ".format(urllib.parse.quote(chemkin_label)) if chemkin_label in self.votes else "No votes yet. "
                output.append("<td>Unknown species. {votes} <a href='propose.html?ck_label={ckl}'>propose match</a></td></tr>".format(ckl=urllib.parse.quote(chemkin_label), votes=votes))
        output.extend(['</table>', self.html_tail])
        return ('\n'.join(output))

    @cherrypy.expose
    def identified_json(self):
        return json.dumps(self.identified_labels)

    @cherrypy.expose
    def unmatchedreactions_html(self):
        img = self._img
        output = [
            self.html_head(),
            '<h1>{0} Unmatched Reactions</h1><table style="width:500px"><tr>'.format(
                len(self.chemkin_reactions_unmatched))
        ]
        for i, reaction in enumerate(self.chemkin_reactions_unmatched):
            reaction_string = []
            for token in str(reaction).split():
                if token in ['+', '<=>', '=>']:
                    pass
                elif token in self.species_dict_rmg:
                    token = img(self.species_dict_rmg[token])
                elif token in self.species_dict:
                    token = "<a href='propose.html?ck_label={escaped}' class='unid'>{plain}</a>".format(
                        escaped=urllib.parse.quote(token),
                        plain=token)
                else:
                    token = "<span class='unid'>{0}</span>".format(token)
                reaction_string.append(token)
            reaction_string = ' '.join(reaction_string)
            output.append(
                "<tr><td>{number}</td><td style='white-space: nowrap;'>{rxn}</td></tr>".format(
                    number=i + 1,
                    rxn=reaction_string))
        output.append(self.html_tail)
        return ('\n'.join(output))

    @cherrypy.expose
    def ThermoLibrary_py(self):
        """The thermo database in py format"""
        return serve_file(os.path.join(self.output_path, 'RMG-Py-thermo-library', 'ThermoLibrary.py'),
                          content_type='application/octet-stream')

    @cherrypy.expose
    def autoconfirm_html(self):
        """Make (hopefully) non-controversial matches automatically"""
        output = [
            self.html_head(),
            '<h1>Autoconfirm suggestions</h1><table style="width:500px">',
            '<tr><th>Label</th>',
            '<th>Reactions matched</th>'
            '<th>Thermo matches</th>',
            '<th>Count</th>',
            '<th>Name matches</th>',
            '</tr>'
            ]
        votes = self.votes.copy()
        for chemkin_label in sorted(votes.keys(), key=lambda label:len(votes[label])):
            possible_matches = votes[chemkin_label]
            output.append('\n<tr><td>{}</td>'.format(chemkin_label))
            if len(possible_matches) != 1:
                output.append('<td>{} possible matches.'.format(len(possible_matches)))
                output.append('Not confirming</td></tr>')
                continue
            auto_confirm = True  # for now...
            chemkin_reactions = self.chemkin_reactions_dict[chemkin_label]
            for matching_species, voting_reactions in possible_matches.items():
                pass  # we know at this point there is only one item
            fraction_matched = float(len(voting_reactions)) / len(chemkin_reactions)
            output.append('<td>{} of {} = {:.0f}%</td>'.format(len(voting_reactions), len(chemkin_reactions), fraction_matched * 100))
            if fraction_matched < 0.5:
                auto_confirm = False
            try:
                thermo_matches = []
                output.append('<td>')
                for library_name, library_species_name in self.thermo_matches[chemkin_label][matching_species]:
                    names_match = ( library_species_name.upper() == chemkin_label.upper() )
                    thermo_matches.append(int(names_match))
                    output.append("<span title='{spec}' class='{match}'>{lib}</span>".format(
                                        lib=library_name,
                                        spec=library_species_name,
                                        match=('goodmatch' if names_match else 'badmatch'),
                                        ))
                output.append("</td>")
            except KeyError:
                output.append("None</td>")
                auto_confirm = False
            output.append('<td>{} libraries'.format(len(thermo_matches)))
            if len(thermo_matches) < 2:
                output.append(" is insufficient")
                auto_confirm = False
            if thermo_matches:
                fraction_matched = float(sum(thermo_matches)) / len(thermo_matches)
            else:
                fraction_matched = 0
            output.append("</td><td>{0:.0f}% name matches ".format(fraction_matched * 100))
            if fraction_matched < 0.5:
                output.append(" is insufficient")
                auto_confirm = False

            output.append("</td><td>{}".format(self._img(matching_species)))
            output.append("</td><td><a href='match.html?ck_label={ckl}&rmg_label={rmgl}' class='confirm'>confirm</a></td>".format(
                        ckl=urllib.parse.quote(chemkin_label),
                        rmgl=urllib.parse.quote(str(matching_species))))
            if auto_confirm:
                output.append('<td>Autoconfirm passes!</td>')
            output.append('</tr>')
        output.append("</table>")
        output.append(self.html_tail)
        return '\n'.join(output)
    
    @cherrypy.expose
    def votes2_html(self):
        votes = self.votes.copy()
        img = self._img
        chemkin_controversy = dict((label, 0) for label in votes.keys())
        rmg_controversy = {}
        flat_votes = {}

        labels_waiting_to_process = [item[0] for item in self.manual_matches_to_process]
        species_waiting_to_process = [item[1] for item in self.manual_matches_to_process]
        # to turn reactions into pictures
        searcher = re.compile('(\S+\(\d+\))\s')
        def replacer(match):
            return self._img(match.group(1))

        user_proposed_matches = {}
        for match in self.tentative_matches:
            label = match['label']
            species = match['species']
            user = match['username']
            user_proposed_matches[label] = (species, user)

        for chemkin_label, possible_matches in votes.items():
            for matching_species, voting_reactions in possible_matches.items():
                flat_votes[(chemkin_label, matching_species)] = voting_reactions
                chemkin_controversy[chemkin_label] += len(voting_reactions)
                rmg_controversy[matching_species] = rmg_controversy.get(matching_species, 0) + len(voting_reactions)
        output = [self.html_head()]
        output.append("<h1>Votes Tables</h1>")
        for chemkin_label in sorted(chemkin_controversy.keys(), key=lambda label:-chemkin_controversy[label]):
            output.append("<hr id='{0}' />".format(chemkin_label))
            if chemkin_label in labels_waiting_to_process:
                output.append("<h2>{0} has just been identified but not yet processed.</h2>".format(chemkin_label))
                continue
            possible_matches = votes[chemkin_label]
            output.append("<h2>{0} matches {1} RMG species</h2>".format(chemkin_label, len(possible_matches)))
            chemkin_reactions = self.chemkin_reactions_dict[chemkin_label]

            thermo_comment = self.thermo_dict[chemkin_label].comment
            if thermo_comment:
                output.append("""<table><tr><td>Thermo comment:</td>
                <td style='font-size: small;'>{}</td>
                </tr></table>""".format(thermo_comment))

            my_voting_chemkin_reactions = dict()
            
            sorted_matching_species_list = sorted(possible_matches.keys(), key=lambda species:-len(possible_matches[species])+0.001*abs(self.get_enthalpy_discrepancy(chemkin_label, species)))
            
            for s in species_waiting_to_process:
                if s in sorted_matching_species_list:
                    # structure already matched, so remove from possible matches
                    sorted_matching_species_list.remove(s)
            
            # Add thermo matches to the start of the table even if no voting reactions
            for thermo_match in self.thermo_matches.get(chemkin_label, []):
                if thermo_match not in sorted_matching_species_list:
                    sorted_matching_species_list.insert(0, thermo_match)
            # Add user-proposed matches to the start
            if chemkin_label in user_proposed_matches:
                species, proposer = user_proposed_matches[chemkin_label]
                if species not in sorted_matching_species_list:
                    sorted_matching_species_list.insert(0, species)

            # Helper function to normalize reaction strings for comparison
            def normalize_rxn_str(s):
                """
                Normalize a reaction string for comparison.
                Removes species indices and normalizes whitespace.
                Returns a canonical form that handles forward/reverse direction.
                """
                if not isinstance(s, str):
                    s = str(s)
                # Remove species indices like (23), (145), etc. that may appear in RMG species
                s = re.sub(r'\(\d+\)', '', s)
                # Normalize whitespace: collapse multiple spaces, strip
                s = re.sub(r'\s+', ' ', s.strip())
                return s
            
            def reactions_equivalent(rxn_str1, rxn_str2):
                """
                Check if two reaction strings represent the same reaction,
                even if written in opposite directions.
                """
                s1 = normalize_rxn_str(rxn_str1)
                s2 = normalize_rxn_str(rxn_str2)
                
                # Direct match
                if s1 == s2:
                    return True
                
                # Try to parse and compare both directions
                # Handle both <=> and => arrows
                for arrow in [' <=> ', ' => ']:
                    if arrow in s1:
                        parts1 = s1.split(arrow)
                        if len(parts1) == 2:
                            # Also check the reverse
                            s1_reverse = parts1[1] + arrow + parts1[0]
                            if s1_reverse == s2:
                                return True
                    if arrow in s2:
                        parts2 = s2.split(arrow)
                        if len(parts2) == 2:
                            s2_reverse = parts2[1] + arrow + parts2[0]
                            if s1 == s2_reverse:
                                return True
                
                return False

            # DEBUG: Log first few reactions to understand format
            debug_logged = False
            debug_match_count = 0
            debug_total_comparisons = 0
            
            for chemkin_reaction in chemkin_reactions:
                this_reaction_votes_for = dict()
                my_voting_chemkin_reactions[chemkin_reaction] = this_reaction_votes_for
                
                for matching_species, voting_reactions in possible_matches.items():
                    for (chemkin_rxn, rmg_rxn) in voting_reactions:
                        debug_total_comparisons += 1
                        
                        # Compare using reactions_equivalent which handles direction changes
                        if reactions_equivalent(chemkin_reaction, chemkin_rxn):
                            debug_match_count += 1
                            if matching_species in this_reaction_votes_for:
                                this_reaction_votes_for[matching_species].append(rmg_rxn)
                            else:
                                this_reaction_votes_for[matching_species] = [rmg_rxn]
                        elif not debug_logged:
                            # Debug: log first mismatch for each chemkin_label
                            logging.debug(f"votes2 mismatch for {chemkin_label}:")
                            logging.debug(f"  Current (from dict): |{normalize_rxn_str(chemkin_reaction)}|")
                            logging.debug(f"  Stored (from votes): |{normalize_rxn_str(chemkin_rxn)}|")
                            # ADD: Show raw strings before normalization
                            logging.debug(f"  Raw current: |{chemkin_reaction}|")
                            logging.debug(f"  Raw stored:  |{chemkin_rxn}|")
                            # ADD: Show what matching_species this vote was for
                            logging.debug(f"  This vote was for species: {matching_species}")
                            debug_logged = True
            
            # Log summary for debugging
            if chemkin_label and debug_total_comparisons > 0:
                logging.info(f"votes2 for {chemkin_label}: {debug_match_count}/{debug_total_comparisons} comparisons matched")

            output.append("<table>")
            output.append("<tr><td>Structure</td>")
            for matching_species in sorted_matching_species_list:
                output.append("<td>{img}</td>".format(img=img(matching_species)))
            output.append("</tr>")

            output.append("<tr><td>SMILES</td>")
            for matching_species in sorted_matching_species_list:
                output.append("<td style='font-size: small; white-space: nowrap;'>{smiles}</td>".format(smiles='<br />'.join(
                    [m.to_smiles() for m in matching_species.molecule])))
            output.append("</tr>")

            output.append("<tr><td>Action</td>")
            for matching_species in sorted_matching_species_list:
                output.append(
                    """<td><a href='match.html?ck_label={ckl}&rmg_label={rmgl}' class='confirm'>confirm</a>
                <a href='block.html?ck_label={ckl}&rmg_label={rmgl}' class='block'>block</a></td>""".format(
                        ckl=urllib.parse.quote(chemkin_label),
                        rmgl=urllib.parse.quote(str(matching_species))))
            output.append("</tr>")

            if chemkin_label in user_proposed_matches:
                proposed_species, proposer = user_proposed_matches[chemkin_label]
                output.append("<tr><td>Proposed by...</td>")
                for matching_species in sorted_matching_species_list:
                    if matching_species == proposed_species:
                        output.append("<td class='goodmatch'>{0}</td>".format(proposer))
                    else:
                        output.append("<td></td>")
                output.append("</tr>")

            output.append("<tr><td>&Delta;H(298K)</td>")
            for matching_species in sorted_matching_species_list:
                output.append(
                    "<td><span title='{Hsource}'>{0:.1f} kJ/mol</span></td>".format(
                        self.get_enthalpy_discrepancy(chemkin_label,
                                                    matching_species),
                        Hsource=matching_species.thermo.comment))
            output.append("</tr>")

            output.append("<tr><td></td>")
            for matching_species in sorted_matching_species_list:
                output.append("<td style='font-size: small; width: 150px';>")
                try:
                    for library_name, library_species_name in self.thermo_matches[chemkin_label][matching_species]:
                        output.append("<span title='{spec}' class='{match}'>{lib}</span>".format(
                                            lib=library_name,
                                            spec=library_species_name,
                                            match=('goodmatch' if library_species_name.upper() == chemkin_label.upper() else 'badmatch'),
                                            ))
                    output.append("have the same thermo.</td>")
                except KeyError:
                    output.append("</td>")
            output.append("</tr>")

            # Count matched votes per candidate (votes that correspond to actual chemkin reactions)
            matched_vote_counts = {}
            for matching_species in sorted_matching_species_list:
                count = 0
                for chemkin_reaction in chemkin_reactions:
                    if matching_species in my_voting_chemkin_reactions[chemkin_reaction]:
                        count += len(my_voting_chemkin_reactions[chemkin_reaction][matching_species])
                matched_vote_counts[matching_species] = count

            output.append("<tr><td>{num} Reactions</td>".format(num=len(chemkin_reactions)))
            for matching_species in sorted_matching_species_list:
                try:
                    total_votes = len(possible_matches[matching_species])
                    matched_votes = matched_vote_counts.get(matching_species, 0)
                    if matched_votes == total_votes:
                        output.append("<td>{n}</td>".format(n=total_votes))
                    else:
                        # Show matched/total if they differ
                        output.append("<td title='{total} total votes, {matched} matched to reactions'>{matched}/{total}</td>".format(
                            matched=matched_votes, total=total_votes))
                except KeyError:
                    output.append("<td>{n}</td>".format(n=0))
            output.append("</tr>")
                
            for chemkin_reaction in sorted(chemkin_reactions, key=lambda rxn:-len(my_voting_chemkin_reactions[rxn])):
                reaction_string = []
                for token in str(chemkin_reaction).split():
                    if token in ['+', '<=>', '=>']:
                        pass
                    elif token in self.species_dict_rmg:
                        token = img(self.species_dict_rmg[token])
                    elif token == chemkin_label:
                        token = "<span class='unid'>{0}</span>".format(token)
                    elif token in votes:
                        token = "<a href='#{0}' class='species'>{0}</a>".format(token)
                    reaction_string.append(token)
                reaction_string = ' '.join(reaction_string)
                
                output.append("<tr><td style='white-space: nowrap;'>{rxn!s}</td>".format(rxn=reaction_string))
                this_reaction_votes_for = my_voting_chemkin_reactions[chemkin_reaction]
                for matching_species in sorted_matching_species_list :
                    if matching_species in this_reaction_votes_for:
                        rmg_rxns = this_reaction_votes_for[matching_species]
                        output.append("<td style='font-size: small;'>{family}</td>".format(
                            family='<br/>'.join([getattr(r, 'family', '-') for r in rmg_rxns])))
                    else:
                        output.append("<td>-</td>")
                output.append("</tr>")
            output.append("</table>")

        output.append(self.html_tail)
        return '\n'.join(output)

    @cherrypy.expose
    def votes_html(self):
        votes = self.votes.copy()
        img = self._img
        chemkin_controversy = dict((label, 0) for label in votes.keys())
        rmg_controversy = {}
        flat_votes = {}

        labels_waiting_to_process = [item[0] for item in self.manual_matches_to_process]
        species_waiting_to_process = [item[1] for item in self.manual_matches_to_process]
        # to turn reactions into pictures
        searcher = re.compile('(\S+\(\d+\))\s')

        def replacer(match):
            return self._img(match.group(1))

        for chemkin_label, possible_matches in votes.items():
            for matching_species, voting_reactions in possible_matches.items():
                flat_votes[(chemkin_label, matching_species)] = voting_reactions
                chemkin_controversy[chemkin_label] += len(voting_reactions)
                rmg_controversy[matching_species] = rmg_controversy.get(matching_species, 0) + len(voting_reactions)
        output = [self.html_head()]
        output.append("<h1>Votes</h1>")
        for chemkin_label in sorted(chemkin_controversy.keys(), key=lambda label:-chemkin_controversy[label]):
            output.append("<hr id='{0}' />".format(chemkin_label))
            if chemkin_label in labels_waiting_to_process:
                output.append("<h2>{0} has just been identified but not yet processed.</h2>".format(chemkin_label))
                continue
            possible_matches = votes[chemkin_label]
            output.append("<h2>{0} matches {1} RMG species</h2>".format(chemkin_label, len(possible_matches)))
            
            for matching_species in sorted(possible_matches.keys(), key=lambda species:-len(possible_matches[species])) :
                if matching_species in species_waiting_to_process:
                    output.append("{img} which has just been identified but not yet processed.<br>".format(img=img(matching_species)))
                    continue
                voting_reactions = possible_matches[matching_species]
                output.append(
                    "<a href='match.html?ck_label={ckl}&rmg_label={rmgl}'>{img}</a>  according to {n} reactions. ".format(
                        ckl=urllib.parse.quote(chemkin_label),
                        rmgl=urllib.parse.quote(str(matching_species)),
                        img=img(matching_species),
                        n=len(voting_reactions)))
                output.append(
                    "  Enthalpies at 298K differ by <span title='{Hsource}'>{0:.1f} kJ/mol</span><br>".format(
                        self.get_enthalpy_discrepancy(chemkin_label,
                                                    matching_species),
                        Hsource=matching_species.thermo.comment))
                try:
                    for library_name, library_species_name in self.thermo_matches[chemkin_label][matching_species]:
                        output.append("<span title='{spec}' class='{match}'>{lib}</span>, ".format(
                                            lib=library_name,
                                            spec=library_species_name,
                                            match=('goodmatch' if library_species_name.upper() == chemkin_label.upper() else 'badmatch'),
                                            ))
                    output.append("have the same thermo.<br>")
                except KeyError:
                    pass
                output.append('<table  style="width:800px">')
                for n, rxn in enumerate(voting_reactions):
                    if isinstance(rxn, tuple):
                        chemkinrxn = str(rxn[0])
                        rmgrxn = str(rxn[1])
                        rmg_rxn_pics = searcher.sub(replacer, rmgrxn + ' ')
                        family = getattr(rxn[1], 'family', '-')
                        output.append("<tr><td>{0}</td><td style='white-space: nowrap;'> {1!s}   </td><td>  {2!s} </td><td style='text-align: right; font-size: small; white-space: nowrap;'>{3!s}</td></tr>".format(n + 1, chemkinrxn, rmg_rxn_pics, family))
                    else:
                        output.append("<tr><td>{0}</td><td> {1!s}</td></tr>".format(n + 1, rxn))
                output.append("</table>")
        output.append(self.html_tail)
        return '\n'.join(output)

    @cherrypy.expose
    def edit_html(self, ck_label=None, smiles=None):
        smiles = str(smiles)
        proposal = Molecule(smiles=str(smiles))
        species, isnew = self.rmg_object.reaction_model.make_new_species(proposal)
        species.generate_resonance_structures()
        self.draw_species(species)
        if isnew:
            species.thermo = generate_thermo_data(species)

        # get a list of names from Cactus
        url = "http://cactus.nci.nih.gov/chemical/structure/{0}/names".format(urllib.parse.quote(smiles))
        try:
            f = urllib.request.urlopen(url, timeout=4)
            response = f.read()
        except (urllib.error.URLError, socket.timeout) as e:
            print("Couldn't identify {0}. NCI resolver responded {1} to request for {2}".format(smiles, e, url))
            response = "Unknown"

        output = [self.html_head()]
        output.append("<h1>Edit {0}</h1>".format(ck_label))
        output.append("""
            <form action="edit.html" method="get">
            <input type=hidden name="ck_label" value="{lab}">
            <input type=text name="smiles" value="{smi}">
            <input type=submit label="Edit">
            </form>
            """.format(lab=ck_label, smi=smiles))
        username = self.get_username()
        if self.formula_dict[ck_label] == species.molecule[0].get_formula():
            if not self.set_tentative_match(ck_label, species, username=username):
                # first attempt removed the old tentative match
                # second attempt should add the new!
                self.set_tentative_match(ck_label, species, username=username)
            output.append("Return to <a href='tentative.html'>Tentative matches</a> to confirm.")
        else:
            output.append(
                '<p><b>Invalid match!</b></p>Species "{lab}" has formula {f1}<br/>\n but SMILES "{smi}" has formula {f2}'.format(
                    lab=ck_label,
                    f1=self.formula_dict[ck_label],
                    smi=smiles,
                    f2=species.molecule[0].get_formula()))
        output.append("<div style='margin: 2em;'>{img}</div>".format(
            img=self._img(species)))
        output.append(
            "Thermo difference at 298K: {dh:.1f} kJ/mol<br/><br/>".format(
                dh=self.get_enthalpy_discrepancy(ck_label, species)))
        output.append("Names:")
        for name in response.splitlines():
            output.append("<li>{name}</li>".format(name=name))

        output.append(self.html_tail)

        return '\n'.join(output)

    @cherrypy.expose
    def propose_html(self, ck_label=None):
        output = [self.html_head()]
        output.append("<h1>Propose {0}</h1>".format(ck_label))
        output.append("""
            <form action="edit.html" method="get">
            <input type=hidden name="ck_label" value="{lab}">
            <input type=text name="smiles">
            <input type=submit>
            </form>
            """.format(lab=ck_label))
        output.append(self.html_tail)
        return '\n'.join(output)

    @cherrypy.expose
    def block_html(self, ck_label=None, rmg_label=None):
        #rmg_name = re.match('^(.*)\(\d+\)$',rmg_label).group(1)
        chemical_formula = self.formula_dict[ck_label]
        for rmg_species in self.rmg_object.reaction_model.species_dict[chemical_formula]:
            if str(rmg_species) == rmg_label:
                break
        else:
            raise KeyError("Couldn't find RMG species with formula {0} and name {1}".format(chemical_formula, rmg_label))

        if ck_label not in self.votes:
            logging.warning("Blocking a match that had no votes for anything: {0} is {1} with SMILES {2}".format(ck_label, rmg_label, rmg_species.molecule[0].to_smiles()))
        elif rmg_species not in self.votes[ck_label] :
            logging.warning("Blocking a match that had no votes for this match: {0} is {1} with SMILES {2}".format(ck_label, rmg_label, rmg_species.molecule[0].to_smiles()))
        assert str(rmg_species) == rmg_label, "Didn't find the right RMG species!"

        self.block_match(ck_label, rmg_species, username=self.get_username())

        # referer = cherrypy.request.headers.get("Referer", "tentative.html")
        # raise cherrypy.HTTPRedirect(referer)
        raise cherrypy.HTTPRedirect(get_relative_referer("tentative.html"))

    @cherrypy.expose
    def confirm_html(self, ck_label=None, rmg_label=None):
        #rmg_name = re.match('^(.*)\(\d+\)$',rmg_label).group(1)
        chemical_formula = self.formula_dict[ck_label]
        for rmg_species in self.rmg_object.reaction_model.species_dict[chemical_formula]:
            if str(rmg_species) == rmg_label:
                break
        else:
            raise KeyError("Couldn't find RMG species with formula {0} and name {1}".format(chemical_formula, rmg_label))

        if ck_label not in self.votes:
            logging.warning(
                "Confirming a match that had no votes for anything: {0} is {1} with SMILES {2}".format(
                    ck_label, rmg_label, rmg_species.molecule[0].to_smiles()))
        elif rmg_species not in self.votes[ck_label]:
            logging.warning(
                "Confirming a match that had no votes for this match: {0} is {1} with SMILES {2}".format(
                    ck_label, rmg_label, rmg_species.molecule[0].to_smiles()))

        assert str(rmg_species) == rmg_label, "Didn't find the right RMG species!"

        for match in self.tentative_matches:
            if match['label'] == ck_label:
                if str(match['species']) != rmg_label:
                    raise cherrypy.HTTPError(message="Trying to confirm something that wasn't a tentative match!")
                self.manual_matches_to_process.append((str(ck_label), rmg_species))
                self.tentative_matches.remove(match)
                break
        else:
            raise cherrypy.HTTPError(message="Trying to confirm something that has no tentative matches!")
        self.save_match_to_file(ck_label, rmg_species, username=self.get_username())

        # referer = cherrypy.request.headers.get("Referer", "tentative.html")
        # raise cherrypy.HTTPRedirect(referer)
        raise cherrypy.HTTPRedirect(get_relative_referer("tentative.html"))

    @cherrypy.expose
    def clearthermomatch_html(self, ck_label=None, rmg_name=None):
        "Clear the specified thermo match"
        if ck_label:
            self.clear_thermo_match(ck_label, rmg_name)
        # referer = cherrypy.request.headers.get("Referer", "thermomatches.html")
        # raise cherrypy.HTTPRedirect(referer)
        raise cherrypy.HTTPRedirect(get_relative_referer("thermomatches.html"))

    @cherrypy.expose
    def confirmthermomatch_html(self, ck_label=None, rmg_name=None):
        for rmg_species in self.thermo_matches[ck_label].keys():
            if str(rmg_species) == rmg_name:
                break
        else:
            return "Trying to confirm something that wasn't a thermo match"
        self.clear_thermo_match(ck_label, None)
        self.manual_matches_to_process.append((str(ck_label), rmg_species))
        self.clear_tentative_match(ck_label, None)
        self.save_match_to_file(ck_label, rmg_species, username=self.get_username())
        # referer = cherrypy.request.headers.get("Referer", "thermomatches.html")
        # raise cherrypy.HTTPRedirect(referer)
        raise cherrypy.HTTPRedirect(get_relative_referer("thermomatches.html"))

    @cherrypy.expose
    def clear_html(self, ck_label=None):
        logging.info("Clearing the tentative match for {0} at user's request".format(ck_label))
        self.clear_tentative_match(ck_label, None)
        raise cherrypy.HTTPRedirect("tentative.html")

    @cherrypy.expose
    def match_html(self, ck_label=None, rmg_label=None):
        if ck_label not in self.votes:
            return "ck_label not valid"
        for rmg_species in self.votes[ck_label].keys():
            if str(rmg_species) == rmg_label:
                self.manual_matches_to_process.append((str(ck_label), rmg_species))
                break
        else:
            # Maybe it was just a thermo match with no votes?
            self.confirmthermomatch_html(ck_label, rmg_label)
            # If that didn't raise a HTTPRedirect, then it wasn't a thermo match either
            return "rmg_label not a candidate for that ck_label"

        self.save_match_to_file(ck_label, rmg_species, username=self.get_username())
        ## Wait for it to be processed:
        #while self.manual_matches_to_process:
        #    time.sleep(1)
        # referer = cherrypy.request.headers.get("Referer", "votes2.html")
        # raise cherrypy.HTTPRedirect(referer)
        raise cherrypy.HTTPRedirect(get_relative_referer("votes2.html"))

    @cherrypy.expose
    def progress_json(self):
        total = len(self.species_list)
        identified = len(self.identified_labels) + len(self.manual_matches_to_process)
        unprocessed = len(self.identified_unprocessed_labels) + len(self.manual_matches_to_process)
        tentative = len(self.tentative_matches)
        unmatchedreactions = len(self.chemkin_reactions_unmatched)
        totalreactions = len(self.chemkin_reactions)
        thermomatches = len(self.thermo_matches)
        answer = {
            'processed': identified - unprocessed,
            'unprocessed': unprocessed,
            'confirmed': identified,
            'tentative': tentative,
            'unidentified': total - identified - tentative,
            'unconfirmed': total - identified,
            'total': total,
            'unmatchedreactions': unmatchedreactions,
            'totalreactions': totalreactions,
            'thermomatches': thermomatches,
        }
        
        # Write progress.json to disk for SSH-based dashboard sync
        try:
            progress_file = os.path.join(self.args.output_directory, '..', 'progress.json')
            with open(progress_file, 'w') as f:
                json.dump(answer, f)
        except Exception as e:
            logging.debug(f"Could not write progress.json to disk: {e}")
        
        # Persist progress to vote database for Django dashboard sync
        if self.vote_db_enabled and self.vote_db:
            try:
                self.vote_db.update_progress_from_importer(self.job_id, answer)
            except Exception as e:
                logging.warning(f"Could not save progress to database: {e}")
        
        return json.dumps(answer)

    @cherrypy.expose
    def votes_api_json(self):
        """
        Export detailed voting data as JSON for dashboard consumption.
        Returns complete voting information including candidates, reactions, and thermodynamics.
        """
        species_data = []
        
        # Process unidentified species with voting data
        for ck_label, candidates in self.votes.items():
            candidates_list = []
            
            for rmg_species, reaction_pairs in candidates.items():
                votes_list = []
                unique_votes = set()
                
                # Process voting reactions
                for ck_rxn, rmg_rxn in reaction_pairs:
                    reaction_str = str(ck_rxn)
                    is_unique = reaction_str not in unique_votes
                    unique_votes.add(reaction_str)
                    
                    votes_list.append({
                        'chemkin_reaction': str(ck_rxn),
                        'rmg_reaction': str(rmg_rxn),
                        'family': getattr(rmg_rxn, 'family', ''),
                        'is_unique': is_unique
                    })
                
                # Get enthalpy discrepancy
                try:
                    enthalpy_diff = self.get_enthalpy_discrepancy(ck_label, rmg_species)
                except:
                    enthalpy_diff = None
                
                # Get SMILES
                try:
                    smiles = rmg_species.molecule[0].to_smiles()
                except:
                    smiles = ''
                
                # Get formula
                try:
                    formula = rmg_species.molecule[0].get_formula()
                except:
                    formula = ''
                
                candidates_list.append({
                    'smiles': smiles,
                    'label': str(rmg_species),
                    'formula': formula,
                    'enthalpy_discrepancy': enthalpy_diff,
                    'votes': votes_list,
                    'vote_count': len(votes_list),
                    'unique_vote_count': len(unique_votes)
                })
            
            # Sort candidates by vote count (descending)
            candidates_list.sort(key=lambda x: (-x['unique_vote_count'], -x['vote_count']))
            
            species_data.append({
                'chemkin_label': ck_label,
                'formula': self.formula_dict.get(ck_label, ''),
                'candidates': candidates_list,
                'status': 'unidentified'
            })
        
        # Add tentative matches
        # tentative_matches is a list, not a dict - iterate directly
        if isinstance(self.tentative_matches, list):
            for match_item in self.tentative_matches:
                # Extract chemkin_label and rmg_species from tuple or dict
                if isinstance(match_item, tuple) and len(match_item) == 2:
                    ck_label, rmg_species = match_item
                elif isinstance(match_item, dict):
                    ck_label = match_item.get('label')
                    rmg_species = match_item.get('species')
                else:
                    continue  # Skip invalid format
                
                try:
                    smiles = rmg_species.molecule[0].to_smiles()
                except:
                    smiles = ''
                
                try:
                    formula = rmg_species.molecule[0].get_formula()
                except:
                    formula = self.formula_dict.get(ck_label, '')
                
                species_data.append({
                    'chemkin_label': ck_label,
                    'formula': formula,
                    'candidates': [{
                        'smiles': smiles,
                        'label': str(rmg_species),
                        'formula': formula,
                        'enthalpy_discrepancy': None,
                        'votes': [],
                    'vote_count': 0,
                    'unique_vote_count': 0
                }],
                'status': 'tentative'
            })
        
        # Add identified species
        identified_species = []
        for ck_label in self.identified_labels:
            if ck_label in self.species_dict_rmg:
                rmg_species = self.species_dict_rmg[ck_label]
                try:
                    smiles = rmg_species.molecule[0].to_smiles()
                except:
                    smiles = ''
                
                try:
                    formula = rmg_species.molecule[0].get_formula()
                except:
                    formula = self.formula_dict.get(ck_label, '')
                
                identified_species.append({
                    'chemkin_label': ck_label,
                    'formula': formula,
                    'smiles': smiles,
                    'rmg_label': str(rmg_species)
                })
        
        # Get blocked matches
        blocked_matches = []
        for ck_label, blocked_species_list in self.blocked_matches.items():
            for rmg_species in blocked_species_list:
                try:
                    smiles = rmg_species.molecule[0].to_smiles()
                except:
                    smiles = ''
                
                blocked_matches.append({
                    'chemkin_label': ck_label,
                    'rmg_label': str(rmg_species),
                    'smiles': smiles
                })
        
        # Compile response
        response = {
            'job_name': self.name,
            'species': species_data,
            'identified': identified_species,
            'blocked': blocked_matches,
            'statistics': json.loads(self.progress_json())
        }
        
        return json.dumps(response)

    def html_head(self):
        location = os.path.abspath(self.args.reactions or self.args.species)
        name = self.name
        return """
<html>
<head>
<script src="https://ajax.googleapis.com/ajax/libs/jquery/1.9.1/jquery.min.js"></script>
<script>
function also_update(json) {
 // replace this with another <script> block on a specific page if you want it to do something
 }
var last_alert = """ + str(min(len(self.identified_labels) - len(self.identified_unprocessed_labels), 5)) + """;
var progress_updates = 0;
function update_stats() {
    $.getJSON( "progress.json", function( json ) {
            var total = json.total;
            console.log('Updating stats.. Unidentified now ' + json.unidentified );
            $('#processed').html(json.processed).width(100*json.processed/total+'%');
            $('#unprocessed').html(json.unprocessed+json.processed).width(100*json.unprocessed/total+'%');
            $('#tentative').html(json.unprocessed+json.processed+json.tentative).width(100*json.tentative/total+'%');
            $('#unidentified').html(total).width(100*json.unidentified/total+'%');
            also_update(json); // any other update scripts for specific pages
            if ((json.processed>last_alert) && (json.unprocessed==0) && (progress_updates > 0)) {
                $("title").text("Input needed! Please confirm a match.");
                last_alert = json.processed;
            }
            repeater = setTimeout(update_stats, 10000); // do again in 10 seconds
            progress_updates++;
        }).fail(function( jqxhr, text_status, error ) {
              var err = text_status + ', ' + error;
              console.log( "Request Failed: " + err);
        });
}
$( document ).ready(function() {
    update_stats();
});
</script>
<style>
#processed {background-color: #7777ff;}
#unprocessed {background-color: #9999ff;}
#tentative {background-color: #bbbbff;}
#unidentified {background-color: #eeeeff;}
td.bar { text-align: right; overflow: hidden}
.goodmatch {color: green;}
.badmatch {color: red;}
.unid {color: #00DE1A;}
a.unid {text-decoration: none;}
a.unid:hover {text-decoration: underline;}
a.confirm {background-color: green; padding-left: 1em; padding-right: 1em; color: white; text-decoration: none;}
a.block {background-color: red; padding-left: 1em; padding-right: 1em; color: white; text-decoration: none;}
td.confirmed {border-left: 5px solid green;}
td.tentative {border-left: 5px solid orange;}
td.unknown {border-left: 5px solid red;}
h1, h2 {font-family: Helvetica, sans-serif;}
td.centered {text-align: center;}
</style>
<title>""" + name + """</title>
</head>

<body>
<div style="position: fixed; width: 100%; ">
    <table width=98% style="table-layout:fixed;"><tr>
        <td class="bar" id="processed"></td>
        <td class="bar" id="unprocessed"></td>
        <td class="bar" id="tentative"></td>
        <td class="bar" id="unidentified"></td>
    </tr>
    </table>
</div>
<div style="height: 1em; padding-top: 0.5em;"><br>
<script language="JavaScript">
document.write('<a href="//' + window.location.hostname + ':8000/importer/" >Dashboard</a> > ' );
</script>
<a href="./">""" + name + """</a>&nbsp</div>
    """

    html_tail = """
    </body></html>
    """


def run_cherry_py_server(args):
    import cherrypy
    cherrypy.server.socket_host = '0.0.0.0'
    cherrypy.server.socket_port = args.port
    cherrypy.config.update({
        'environment': 'production',
        'log.error_file': os.path.join(args.output_directory,
                                       'CherryPyError.log'),
        'log.access_file': '',
        'log.screen': False
    })

    conf = {
        '/img': {
            'tools.staticdir.on': True,
            'tools.staticdir.dir': os.path.join(args.output_directory,
                                                'species'),
        }
    }
    cherrypy.log.access_log.propagate = False
    cherrypy.quickstart(mm, '/', config=conf)


if __name__ == '__main__':

    # Parse the command-line arguments (requires the argparse module)
    args = parse_command_line_arguments()
    os.path.exists(args.output_directory) or os.makedirs(args.output_directory)

    # Initialize the logging system (resets the RMG.log file)
    level = logging.INFO
    if args.debug: level = 0
    elif args.verbose: level = logging.DEBUG
    elif args.quiet: level = logging.WARNING
    initialize_log(level, os.path.join(args.output_directory, 'RMG.log'))


    mm = ModelMatcher(args)

#     t = threading.Thread(target=mm.main)
#     t.daemon = False
#     t.start()

    if args.minimal:
        mm.minimal()
        print("Done")
        exit()

    t2 = threading.Thread(target=run_cherry_py_server, args=(args, ))
    t2.daemon = True
    t2.start()

    #import webbrowser
    print('http://127.0.0.1:{0:d}'.format(args.port))
    #webbrowser.open('http://127.0.0.1:{0:d}'.format(args.port))
    try:
        mm.main()
    except KeyboardInterrupt:
        logging.info("\n" + "="*60)
        logging.info("INTERRUPTED BY USER - Saving votes before exit...")
        logging.info("="*60)
        mm.save_votes_to_storage(force=True)
        mm.export_votes_to_json()
        mm.show_sync_status()
        logging.info("✓ Votes saved successfully. Safe to exit.")
        raise
    except Exception as e:
        logging.error("\n" + "="*60)
        logging.error(f"CRASH DETECTED - Saving votes before exit...")
        logging.error(f"Error: {e}")
        logging.error("="*60)
        mm.save_votes_to_storage(force=True)
        mm.export_votes_to_json()
        mm.show_sync_status()
        logging.info("✓ Votes saved after crash. Data preserved.")
        raise
    finally:
        cherrypy.engine.exit()
