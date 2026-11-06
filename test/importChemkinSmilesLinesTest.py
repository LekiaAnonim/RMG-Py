"""
Tests for how importChemkin.py adds lines to SMILES.txt and SMILES-BLOCKED.txt while it runs.

A confirmation was added after the last line of the file even when that line had no line break,
so the two joined ("C6H6 c1ccccc1AR [Ar]"), and the next confirmation stopped the import with
"ValueError: too many values to unpack (expected 2)" (ANL-Brown, Nov 2026).
"""
import logging
import os
import sys

import pytest

pytest.importorskip('cherrypy')  # importChemkin.py serves its web pages with CherryPy

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import importChemkin as ic  # noqa: E402

from rmgpy.molecule import Molecule  # noqa: E402
from rmgpy.species import Species  # noqa: E402


def matcher(tmp_path, smiles_txt=None, blocked_txt=None):
    mm = ic.ModelMatcher.__new__(ic.ModelMatcher)
    mm.known_species_file = str(tmp_path / 'SMILES.txt')
    mm.blocked_matches_file = str(tmp_path / 'SMILES-BLOCKED.txt')
    mm.identified_by = {}
    for path, text in ((mm.known_species_file, smiles_txt), (mm.blocked_matches_file, blocked_txt)):
        if text is not None:
            with open(path, 'w', newline='') as f:
                f.write(text)
    return mm


def species(smiles):
    return Species(molecule=[Molecule(smiles=smiles)])


def read(path):
    with open(path, newline='') as f:
        return f.read()


def test_a_confirmation_after_a_last_line_without_a_line_break_gets_its_own_line(tmp_path):
    mm = matcher(tmp_path, 'C2H2\tC#C\nC6H6\tc1ccccc1')
    assert mm.save_match_to_file('AR', species('[Ar]'))
    assert mm.save_match_to_file('H2O', species('O'))  # stopped here before
    assert read(mm.known_species_file) == 'C2H2\tC#C\nC6H6\tc1ccccc1\nAR\t[Ar]\nH2O\tO\n'


def test_a_file_with_a_line_break_at_the_end_is_not_given_a_blank_line(tmp_path):
    mm = matcher(tmp_path, 'C2H2\tC#C\r\n')
    assert mm.save_match_to_file('AR', species('[Ar]'), username='lekia')
    assert read(mm.known_species_file) == 'C2H2\tC#C\r\nAR\t[Ar]\t! Confirmed by lekia\n'
    assert mm.identified_by == {'AR': 'lekia'}


def test_a_line_that_is_not_two_words_is_reported_and_does_not_stop_the_import(tmp_path, caplog):
    mm = matcher(tmp_path, 'C2H2\tC#C\nC6H6\tc1ccccc1AR\t[Ar]\n')
    with caplog.at_level(logging.WARNING):
        assert mm.save_match_to_file('H2O', species('O'))
    assert 'Line 2 of {0} is not NAME SMILES (3 words): C6H6\tc1ccccc1AR\t[Ar]'.format(
        mm.known_species_file) in caplog.text
    assert read(mm.known_species_file).endswith('H2O\tO\n')


def test_a_name_already_in_the_file_is_not_written_again(tmp_path):
    mm = matcher(tmp_path, 'AR\t[Ar]\t! Confirmed by lekia\nC6H6\tc1ccccc1 extra\n')
    assert not mm.save_match_to_file('AR', species('[Ar]'))
    assert not mm.save_match_to_file('C6H6', species('c1ccccc1'))  # first word of a bad line
    assert read(mm.known_species_file) == 'AR\t[Ar]\t! Confirmed by lekia\nC6H6\tc1ccccc1 extra\n'


def test_a_missing_smiles_txt_is_made(tmp_path):
    mm = matcher(tmp_path)
    assert mm.save_match_to_file('AR', species('[Ar]'))
    assert read(mm.known_species_file) == 'AR\t[Ar]\n'


def test_a_blocked_match_after_a_last_line_without_a_line_break_gets_its_own_line(tmp_path):
    mm = matcher(tmp_path, blocked_txt='CH4\tCC\t! Blocked by lekia')
    assert mm.save_blocked_match_to_file('H2O', species('OO'), username='lekia')
    assert read(mm.blocked_matches_file) == 'CH4\tCC\t! Blocked by lekia\nH2O\tOO\t! Blocked by lekia\n'
