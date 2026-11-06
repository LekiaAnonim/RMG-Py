"""
Tests for how importChemkin.py reads and adds lines of SMILES.txt and SMILES-BLOCKED.txt while
it runs.

On ANL-Brown (Nov 2026) the next confirmation stopped the import with "ValueError: too many
values to unpack (expected 2)": a restored comment had a line break in it, so its end became a
line of its own, " (restored from the vote database)". The name "Prosper\n" came from reading
"! Confirmed by Prosper" in SMILES.txt with its line break. A last line without a line break
would have done the same, by joining the next line to it ("C6H6 c1ccccc1AR [Ar]").
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


def test_a_confirmed_by_name_is_read_without_its_line_break(tmp_path):
    mm = matcher(tmp_path, 'CH2CC\t[C]#C[CH2]\t! Confirmed by Prosper\nAR\t[Ar]\t! Confirmed by lekia. Checked\r\n')
    mm.formula_dict = {}  # only the reading of the file is tested
    mm.identified_labels = []
    mm.load_known_species(mm.known_species_file)
    assert mm.identified_by == {'CH2CC': 'Prosper', 'AR': 'lekia'}


def test_names_and_comments_are_written_on_one_line(tmp_path):
    mm = matcher(tmp_path, '', blocked_txt='')
    assert mm.save_match_to_file('AR', species('[Ar]'), username='Prosper\n')
    assert mm.save_match_to_file('H2O', species('O'), comment='Confirmed by Prosper\n (restored)')
    assert mm.save_blocked_match_to_file('CH4', species('CC'), username=' Prosper\r\n')
    assert read(mm.known_species_file) == 'AR\t[Ar]\t! Confirmed by Prosper\nH2O\tO\t! Confirmed by Prosper (restored)\n'
    assert read(mm.blocked_matches_file) == 'CH4\tCC\t! Blocked by Prosper\n'
    assert mm.identified_by == {'AR': 'Prosper', 'CH4': 'Prosper'}
