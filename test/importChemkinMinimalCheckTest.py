"""
Tests for importChemkin.py --minimal (Check files on Prometheus): before it reads the reactions
it names the species whose formula gives their structure, as the full import does, without
writing them to SMILES.txt, and it lists every collider that still has no structure.
"""
import argparse
import os
import sys

import pytest

pytest.importorskip('cherrypy')  # importChemkin.py serves its web pages with CherryPy

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import importChemkin as ic  # noqa: E402

MECHANISM = """ELEMENTS
H C F HE
END
SPECIES
H2 H HE CH3F CH2F2 CH2F
END
REACTIONS
H2+M=H+H+M    4.577E19 -1.4 1.044E5
    HE/0.83/ CH3F/6.0/  ! a comment
CH3F(+M)=CH2F+H(+M)    1.0E14 0.0 1.0E5
    LOW/ 1.0E16 0.0 9.0E4 /
    TROE/ 0.5 100.0 1000.0 /
    CH2F2/6.0/ CH3F/6.0/
END
"""

FORMULAS = {'H2': {'H': 2}, 'H': {'H': 1}, 'HE': {'HE': 1}, 'CH3F': {'C': 1, 'H': 3, 'F': 1},
            'CH2F2': {'C': 1, 'H': 2, 'F': 2}, 'CH2F': {'C': 1, 'H': 2, 'F': 1}}


def thermo_entry(label, elements):
    """A CHEMKIN NASA entry with this formula (the coefficients don't matter here)."""
    composition = ''.join('{0:<2}{1:>3}'.format(element, count) for element, count in elements.items())
    first = '{0:<18}{1:6}{2:<20}G{3:10.3f}{4:10.3f}{5:8.2f}{6:6}1'.format(
        label, '', composition, 300.0, 5000.0, 1000.0, '')
    coefficients = ['{0:15.8E}'.format(c) for c in [3.5, 0, 0, 0, 0, -1000.0, 3.0] * 2]
    return '\n'.join([
        first,
        '{0:<79}2'.format(''.join(coefficients[0:5])),
        '{0:<79}3'.format(''.join(coefficients[5:10])),
        '{0:<79}4'.format(''.join(coefficients[10:14])),
    ])


def model(tmp_path, monkeypatch, smiles_txt):
    """A ModelMatcher for a small model in tmp_path, with this SMILES.txt."""
    (tmp_path / 'mechanism.txt').write_text(MECHANISM)
    (tmp_path / 'thermo.txt').write_text('THERMO\n   300.000  1000.000  5000.000\n' + '\n'.join(
        thermo_entry(label, elements) for label, elements in FORMULAS.items()) + '\nEND\n')
    (tmp_path / 'SMILES.txt').write_text(smiles_txt)
    monkeypatch.setenv('VOTE_DB_PATH', str(tmp_path / 'votes.db'))
    args = argparse.Namespace(species=str(tmp_path / 'mechanism.txt'), reactions=str(tmp_path / 'mechanism.txt'),
                              thermo=str(tmp_path / 'thermo.txt'), known=str(tmp_path / 'SMILES.txt'))
    return ic.ModelMatcher(args)


def test_every_collider_without_a_structure_is_listed(tmp_path, monkeypatch):
    mm = model(tmp_path, monkeypatch, '')
    with pytest.raises(ic.CollidersWithoutStructure) as raised:
        mm.minimal()
    message = str(raised.value)
    lines = message.splitlines()
    assert lines[0].startswith('2 species are third-body colliders but have no structure')
    assert 'SMILES.txt' in lines[0]
    assert lines[1].split() == ['CH3F', 'CH3F', 'collider', 'in', 'H2+M=H+H+M', 'and', '1', 'more']
    assert lines[2].split() == ['CH2F2', 'CH2F2', 'collider', 'in', 'CH3F(+M)=CH2F+H(+M)']
    assert len(lines) == 3  # HE is named by its formula, as in the full import; CH2F is no collider
    assert (tmp_path / 'SMILES.txt').read_text() == ''  # the check writes nothing


def test_the_check_names_small_molecules_without_writing_smiles_txt(tmp_path, monkeypatch):
    smiles_txt = 'CH3F\tCF\nCH2F2\tFCF\nCH2F\t[CH2]F\n'
    mm = model(tmp_path, monkeypatch, smiles_txt)
    mm.minimal()  # reads the reactions
    assert len(mm.chemkin_reactions) == 2
    assert set(mm.identified_labels) == set(FORMULAS)
    assert (tmp_path / 'SMILES.txt').read_text() == smiles_txt  # H2, H and HE are not written to it
