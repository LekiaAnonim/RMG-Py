"""
Tests for how importChemkin.py writes structures to SMILES.txt and reads them back, and for how it
keeps identified species from being dropped by RMG's forbidden structures.
"""
import os
import sys

import pytest

pytest.importorskip('cherrypy')  # importChemkin.py serves its web pages with CherryPy

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import importChemkin as ic  # noqa: E402

from rmgpy.data.base import Entry, ForbiddenStructures  # noqa: E402
from rmgpy.molecule import Molecule  # noqa: E402
from rmgpy.molecule.group import Group  # noqa: E402
from rmgpy.species import Species  # noqa: E402

SINGLET_CH3CF = """multiplicity 1
1 C u0 p0 c0 {2,S} {3,S} {4,S} {5,S}
2 C u0 p1 c0 {1,S} {6,S}
3 H u0 p0 c0 {1,S}
4 H u0 p0 c0 {1,S}
5 H u0 p0 c0 {1,S}
6 F u0 p3 c0 {2,S}"""
TRIPLET_CH3CF = """multiplicity 3
1 C u0 p0 c0 {2,S} {3,S} {4,S} {5,S}
2 C u2 p0 c0 {1,S} {6,S}
3 H u0 p0 c0 {1,S}
4 H u0 p0 c0 {1,S}
5 H u0 p0 c0 {1,S}
6 F u0 p3 c0 {2,S}"""
SINGLET_CYCLOPROPENYLIDENE = """multiplicity 1
1 C u0 p1 c0 {2,S} {3,S}
2 C u0 p0 c0 {1,S} {3,D} {4,S}
3 C u0 p0 c0 {1,S} {2,D} {5,S}
4 H u0 p0 c0 {2,S}
5 H u0 p0 c0 {3,S}"""
SINGLET_PROPADIENYLIDENE = """multiplicity 1
1 C u0 p0 c0 {2,D} {4,S} {5,S}
2 C u0 p0 c0 {1,D} {3,D}
3 C u0 p1 c0 {2,D}
4 H u0 p0 c0 {1,S}
5 H u0 p0 c0 {1,S}"""
DOUBLET_C3H = """multiplicity 2
1 C u1 p0 c0 {2,D} {4,S}
2 C u0 p0 c0 {1,D} {3,D}
3 C u0 p1 c0 {2,D}
4 H u0 p0 c0 {1,S}"""


def round_trip(adjlist):
    """What known_smiles_for writes for a molecule, and whether it reads back as the same molecule."""
    molecule = Molecule().from_adjacency_list(adjlist)
    written = ic.known_smiles_for(molecule)
    back = ic.molecule_from_known_smiles(written)
    return written, back.multiplicity == molecule.multiplicity and back.is_isomorphic(molecule)


class TestSmilesTxtNotation:

    @pytest.mark.parametrize('adjlist, spin', [(SINGLET_CH3CF, 'singlet'), (SINGLET_CYCLOPROPENYLIDENE, 'singlet'),
                                               (SINGLET_PROPADIENYLIDENE, 'singlet'), (DOUBLET_C3H, 'doublet')])
    def test_the_spin_is_written_when_a_smiles_cannot_say_it(self, adjlist, spin):
        written, same = round_trip(adjlist)
        assert written.startswith(spin)
        assert same

    def test_plain_smiles_are_written_when_they_read_back(self):
        for smiles in ('O=CO', '[CH2]C=C', 'C#C[CH2]', '[C]#C[O]'):
            molecule = Molecule(smiles=smiles)
            assert ic.known_smiles_for(molecule) == molecule.to_smiles()
        written, same = round_trip(TRIPLET_CH3CF)  # C[C]F reads back as the triplet
        assert written == 'C[C]F' and same

    def test_special_names_come_first(self):
        singlet_ch2 = Molecule().from_adjacency_list(ic.SPECIAL_SMILES['singlet[CH2]'])
        assert ic.known_smiles_for(singlet_ch2) == 'singlet[CH2]'
        vinylidene = Molecule().from_adjacency_list(ic.SPECIAL_SMILES['singletC=[C]'])
        assert ic.known_smiles_for(vinylidene) == 'singletC=[C]'

    def test_reading_a_stated_spin(self):
        singlet = ic.molecule_from_known_smiles('singletC[C]F')
        carbene = [a for a in singlet.atoms if a.is_carbon() and a.lone_pairs == 1]
        assert singlet.multiplicity == 1 and len(carbene) == 1 and carbene[0].radical_electrons == 0
        assert ic.molecule_from_known_smiles('tripletC[C]F').multiplicity == 3
        assert ic.molecule_from_known_smiles('doublet[C]=C=[CH]').get_formula() == 'C3H'
        with pytest.raises(ValueError):
            ic.molecule_from_known_smiles('singlet[CH2]C=C')  # an allyl radical can't be a singlet

    def test_states_spin(self):
        assert ic.states_spin('singletC[C]F') and ic.states_spin('triplet[CH]O') and ic.states_spin('doublet[C]F')
        assert not ic.states_spin('C[C]F') and not ic.states_spin('[CH]O')


class TestForbiddenStructures:

    @pytest.fixture
    def forbidden(self):
        """RMG's Carbene_S_triplet: a triplet carbene carbon bonded to anything but hydrogen."""
        structures = ForbiddenStructures()
        structures.entries['Carbene_S_triplet'] = Entry(
            label='Carbene_S_triplet', item=Group().from_adjacency_list("1 C u2 p0 {2,S}\n2 R!H u0 {1,S}"))
        return structures

    def test_an_allowed_resonance_structure_goes_first(self, forbidden):
        species = Species(molecule=[Molecule(smiles='[CH]C#C')])  # triplet propargylene
        species.generate_resonance_structures()
        assert forbidden.is_molecule_forbidden(species.molecule[0])
        moved = ic.put_allowed_form_first(species, forbidden)
        assert moved is species.molecule[0]
        assert not forbidden.is_molecule_forbidden(species.molecule[0])
        assert ic.put_allowed_form_first(species, forbidden) is None  # nothing left to do

    def test_a_structure_forbidden_in_every_form(self, forbidden):
        species = Species(molecule=[Molecule().from_adjacency_list(TRIPLET_CH3CF)])
        species.generate_resonance_structures()
        assert ic.put_allowed_form_first(species, forbidden) is None
        assert ic.forbidden_entry(forbidden, species.molecule[0]) == 'Carbene_S_triplet'
        singlet = Molecule().from_adjacency_list(SINGLET_CH3CF)
        assert ic.forbidden_entry(forbidden, singlet) is None
