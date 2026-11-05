"""
Tests for how importChemkin.py uses the identifications in its vote database on a restart.
SMILES.txt is the record of a model's identifications; one that only the vote database has comes
back as a tentative match to confirm, not as an identification.
"""
import os
import sys
from unittest.mock import MagicMock

import pytest

pytest.importorskip('cherrypy')  # importChemkin.py serves its web pages with CherryPy

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import importChemkin as ic  # noqa: E402

from rmgpy.molecule import Molecule  # noqa: E402
from rmgpy.species import Species  # noqa: E402


def matcher(identified=()):
    """A ModelMatcher with only what restoring from the vote database uses."""
    mm = ic.ModelMatcher.__new__(ic.ModelMatcher)
    mm.tentative_matches = []
    mm.manual_matches_to_process = []
    mm.identified_labels = list(identified)
    mm.identified_by = {}
    mm.species_dict = {label: Species(label=label) for label in identified}
    mm.species_dict_rmg = {}
    mm.formula_dict = {'CH4': 'CH4', 'N2': 'N2'}
    mm.draw_species = lambda species: None
    mm.get_enthalpy_discrepancy = lambda label, species: 0.0
    mm.vote_db_enabled = True
    mm.job_id = 'job'
    mm.vote_db = MagicMock()
    made = {}

    def make_new_species(molecule):
        species = made.setdefault(molecule.to_smiles(), Species(molecule=[molecule], label=molecule.to_smiles()))
        return species, False

    mm.rmg_object = MagicMock()
    mm.rmg_object.reaction_model.make_new_species.side_effect = make_new_species
    return mm


def test_an_identification_only_in_the_vote_database_is_offered_not_applied():
    mm = matcher(identified=['N2'])  # N2 is in SMILES.txt
    mm.vote_db.load_identified_species_full.return_value = [
        {'chemkin_label': 'N2', 'smiles': 'N#N', 'identified_by': 'Auto'},
        {'chemkin_label': 'CH4', 'smiles': 'C', 'identified_by': 'lekia'},
    ]
    mm.vote_db.load_blocked_matches.return_value = []
    mm.load_votes_from_storage()

    assert mm.identified_labels == ['N2']
    assert 'CH4' not in mm.species_dict_rmg
    assert [m['label'] for m in mm.tentative_matches] == ['CH4']
    assert mm.tentative_matches[0]['username'] == 'lekia, from the vote database (not in SMILES.txt)'
    assert mm.tentative_matches[0]['species'].molecule[0].is_isomorphic(Molecule(smiles='C'))


def test_a_vote_database_identification_with_the_wrong_formula_is_not_offered():
    mm = matcher()
    mm.vote_db.load_identified_species_full.return_value = [
        {'chemkin_label': 'CH4', 'smiles': 'CC', 'identified_by': 'Auto'},
    ]
    mm.vote_db.load_blocked_matches.return_value = []
    mm.load_votes_from_storage()
    assert mm.tentative_matches == []
    assert mm.identified_labels == []
