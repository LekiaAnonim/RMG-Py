"""
Tests for how importChemkin.py uses the identifications in its vote database on a restart.
Every confirmation is written to SMILES.txt when it is made, so an identification that only the
vote database has is a line SMILES.txt lost: it is applied, and written back to SMILES.txt with a
comment that says it was restored.
"""
import os
import sys
from unittest.mock import MagicMock

import pytest

pytest.importorskip('cherrypy')  # importChemkin.py serves its web pages with CherryPy

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import importChemkin as ic  # noqa: E402

from rmgpy.species import Species  # noqa: E402


def matcher(tmp_path, smiles_txt, identified=()):
    """A ModelMatcher with only what restoring from the vote database uses."""
    mm = ic.ModelMatcher.__new__(ic.ModelMatcher)
    mm.known_species_file = str(tmp_path / 'SMILES.txt')
    with open(mm.known_species_file, 'w') as f:
        f.write(smiles_txt)
    mm.tentative_matches = []
    mm.manual_matches_to_process = []
    mm.identified_labels = list(identified)
    mm.identified_by = {}
    mm.species_dict = {label: Species(label=label) for label in identified}
    mm.species_dict_rmg = {}
    mm.formula_dict = {'CH4': 'CH4', 'N2': 'N2', 'H2O': 'H2O'}
    mm.vote_db_enabled = True
    mm.job_id = 'job'
    mm.vote_db = MagicMock()
    mm.vote_db.load_blocked_matches.return_value = []
    made = {}

    def make_new_species(molecule):
        species = made.setdefault(molecule.to_smiles(), Species(molecule=[molecule], label=molecule.to_smiles()))
        return species, False

    mm.rmg_object = MagicMock()
    mm.rmg_object.reaction_model.make_new_species.side_effect = make_new_species
    return mm


def test_an_identification_only_in_the_vote_database_is_written_back_to_smiles_txt(tmp_path):
    mm = matcher(tmp_path, 'N2\tN#N\n', identified=['N2'])  # N2 is in SMILES.txt
    mm.vote_db.load_identified_species_full.return_value = [
        {'chemkin_label': 'N2', 'smiles': 'N#N', 'identified_by': 'Auto'},
        {'chemkin_label': 'CH4', 'smiles': 'C', 'identified_by': 'lekia'},
        {'chemkin_label': 'H2O', 'smiles': 'O', 'identified_by': 'Auto'},
    ]
    mm.load_votes_from_storage()

    assert mm.identified_labels == ['N2', 'CH4', 'H2O']  # applied, as before
    assert mm.species_dict_rmg['CH4'].label == 'CH4'
    assert mm.identified_by['CH4'] == 'lekia'
    assert open(mm.known_species_file).read() == (
        'N2\tN#N\n'  # not written twice
        'CH4\tC\t! Confirmed by lekia (restored from the vote database)\n'
        'H2O\tO\t! Restored from the vote database\n')
    assert mm.tentative_matches == []


def test_the_written_lines_read_back_as_the_same_identifications(tmp_path):
    mm = matcher(tmp_path, '')
    mm.vote_db.load_identified_species_full.return_value = [
        {'chemkin_label': 'CH4', 'smiles': 'C', 'identified_by': 'lekia'}]
    mm.load_votes_from_storage()
    lines = [line.split('!')[0].split() for line in open(mm.known_species_file) if line.strip()]
    assert lines == [['CH4', 'C']]


def test_a_vote_database_identification_with_the_wrong_formula_is_not_restored(tmp_path):
    mm = matcher(tmp_path, '')
    mm.vote_db.load_identified_species_full.return_value = [
        {'chemkin_label': 'CH4', 'smiles': 'CC', 'identified_by': 'Auto'},
    ]
    mm.load_votes_from_storage()
    assert mm.identified_labels == []
    assert open(mm.known_species_file).read() == ''


def test_a_missing_smiles_txt_is_made(tmp_path):
    mm = matcher(tmp_path, '')
    os.remove(mm.known_species_file)
    mm.vote_db.load_identified_species_full.return_value = [
        {'chemkin_label': 'CH4', 'smiles': 'C', 'identified_by': 'lekia'}]
    mm.load_votes_from_storage()
    assert 'CH4\tC' in open(mm.known_species_file).read()


def test_a_name_with_a_line_break_in_the_vote_database_is_restored_on_one_line(tmp_path):
    # ANL-Brown, Nov 2026: "Prosper\n" was read from "! Confirmed by Prosper" in SMILES.txt and
    # kept in the vote database; the restored comment took two lines, and the second,
    # " (restored from the vote database)", stopped the import at the next confirmation.
    mm = matcher(tmp_path, '')
    mm.vote_db.load_identified_species_full.return_value = [
        {'chemkin_label': 'CH4', 'smiles': 'C', 'identified_by': 'Prosper\n'}]
    mm.load_votes_from_storage()
    assert open(mm.known_species_file).read() == (
        'CH4\tC\t! Confirmed by Prosper (restored from the vote database)\n')
    assert mm.identified_by['CH4'] == 'Prosper'
