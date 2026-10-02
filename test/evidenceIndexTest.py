"""
Tests for evidence_index.py (the importer's index of RMG-database and earlier-import libraries).
"""
import os
import sys
import time

import pytest

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import evidence_index as ei  # noqa: E402

from rmgpy.molecule import Molecule  # noqa: E402
from rmgpy.kinetics import Arrhenius  # noqa: E402

CH4_NASA = """NASA(
        polynomials = [
            NASAPolynomial(coeffs=[5.14987613E+00,-1.36709788E-02,4.91800599E-05,-4.84743026E-08,1.66693956E-11,-1.02466476E+04,-4.64130376E+00], Tmin=(200,'K'), Tmax=(1000,'K')),
            NASAPolynomial(coeffs=[7.48514950E-02,1.33909467E-02,-5.73285809E-06,1.22292535E-09,-1.01815230E-13,-9.46834459E+03,1.84373180E+01], Tmin=(1000,'K'), Tmax=(3500,'K')),
        ],
        Tmin = (200,'K'),
        Tmax = (3500,'K'),
    )"""

DICTIONARY = """H
multiplicity 2
1 H u1 p0 c0

CH3
multiplicity 2
1 C u1 p0 c0 {2,S} {3,S} {4,S}
2 H u0 p0 c0 {1,S}
3 H u0 p0 c0 {1,S}
4 H u0 p0 c0 {1,S}

CH4
1 C u0 p0 c0 {2,S} {3,S} {4,S} {5,S}
2 H u0 p0 c0 {1,S}
3 H u0 p0 c0 {1,S}
4 H u0 p0 c0 {1,S}
5 H u0 p0 c0 {1,S}

C2H4
1 C u0 p0 c0 {2,D} {3,S} {4,S}
2 C u0 p0 c0 {1,D} {5,S} {6,S}
3 H u0 p0 c0 {1,S}
4 H u0 p0 c0 {1,S}
5 H u0 p0 c0 {2,S}
6 H u0 p0 c0 {2,S}

C2H5
multiplicity 2
1 C u0 p0 c0 {2,S} {3,S} {4,S} {5,S}
2 C u1 p0 c0 {1,S} {6,S} {7,S}
3 H u0 p0 c0 {1,S}
4 H u0 p0 c0 {1,S}
5 H u0 p0 c0 {1,S}
6 H u0 p0 c0 {2,S}
7 H u0 p0 c0 {2,S}
"""

REACTIONS = """name = "test"
entry(
    index = 1,
    label = "CH3 + H <=> CH4",
    kinetics = Arrhenius(A=(1e+14,'cm^3/(mol*s)'), n=0, Ea=(0,'kcal/mol'), T0=(1,'K')),
)
entry(
    index = 2,
    label = "C2H4 + H <=> C2H5",
    kinetics = Arrhenius(A=(5e+12,'cm^3/(mol*s)'), n=0, Ea=(2,'kcal/mol'), T0=(1,'K')),
)
"""

THERMO = '''name = "test"
entry(
    index = 1,
    label = "CH4",
    molecule =
"""
1 C u0 p0 c0 {2,S} {3,S} {4,S} {5,S}
2 H u0 p0 c0 {1,S}
3 H u0 p0 c0 {1,S}
4 H u0 p0 c0 {1,S}
5 H u0 p0 c0 {1,S}
""",
    thermo = %s,
)
''' % CH4_NASA


def write_model(models, name, reactions=REACTIONS):
    model = os.path.join(models, name)
    os.makedirs(os.path.join(model, 'RMG-Py-kinetics-library'))
    os.makedirs(os.path.join(model, 'RMG-Py-thermo-library'))
    with open(os.path.join(model, 'RMG-Py-kinetics-library', 'dictionary.txt'), 'w') as f:
        f.write(DICTIONARY)
    with open(os.path.join(model, 'RMG-Py-kinetics-library', 'reactions.py'), 'w') as f:
        f.write(reactions)
    with open(os.path.join(model, 'RMG-Py-thermo-library', 'ThermoLibrary.py'), 'w') as f:
        f.write(THERMO)
    return model


def key(smiles, multiplicity=None):
    molecule = Molecule(smiles=smiles)
    if multiplicity:
        molecule.multiplicity = multiplicity
    return ei.structure_key(molecule)


class TestStructureKeys:

    def test_spin_states_differ(self):
        singlet = Molecule().from_adjacency_list("multiplicity 1\n1 C u0 p1 c0 {2,S} {3,S}\n2 H u0 p0 c0 {1,S}\n3 H u0 p0 c0 {1,S}")
        triplet = Molecule().from_adjacency_list("multiplicity 3\n1 C u2 p0 c0 {2,S} {3,S}\n2 H u0 p0 c0 {1,S}\n3 H u0 p0 c0 {1,S}")
        assert ei.structure_key(singlet) != ei.structure_key(triplet)

    def test_resonance_forms_match(self):
        assert key('[CH2]C=C') == key('C=C[CH2]')
        assert key('[CH2]C=O') == key('C=C[O]')

    def test_isomers_differ(self):
        assert key('CC[CH2]') != key('C[CH]C')


class TestMatching:

    def test_signature_ignores_order(self):
        assert ei.formula_signature(['H', 'CH3'], ['CH4']) == ei.formula_signature(['CH3', 'H'], ['CH4'])

    def test_side_assignments(self):
        library = [('k_ch3', 'CH3'), ('k_h', 'H')]
        assert ei.side_assignments([(None, 'CH3'), ('k_h', 'H')], library) == {('k_ch3', 'k_h')}
        assert ei.side_assignments([(None, 'CH3'), ('k_other', 'H')], library) == set()  # known species differs
        assert ei.side_assignments([(None, 'C2H5'), ('k_h', 'H')], library) == set()  # formula differs

    def test_match_in_either_direction(self):
        lib_r, lib_p = [('k_ch3', 'CH3'), ('k_h', 'H')], [('k_ch4', 'CH4')]
        forward = ei.match_library_reaction([(None, 'CH3'), ('k_h', 'H')], [('k_ch4', 'CH4')], lib_r, lib_p)
        backward = ei.match_library_reaction([('k_ch4', 'CH4')], [(None, 'CH3'), ('k_h', 'H')], lib_r, lib_p)
        assert forward == [(True, ('k_ch3', 'k_h'), ('k_ch4',))]
        assert backward == [(False, ('k_ch4',), ('k_ch3', 'k_h'))]

    def test_thermo_rule_matches_rmg(self):
        from rmgpy.thermo import NASA, NASAPolynomial  # noqa: F401 (used by eval)
        thermo = eval(CH4_NASA)
        values = ei.thermo_values(thermo)
        assert ei.thermo_values_match(values, values)
        assert ei.thermo_values_match(values, [v * 1.04 for v in values])
        assert not ei.thermo_values_match(values, [v * 1.10 for v in values])

    def test_rate_fingerprint(self):
        k = Arrhenius(A=(1e14, 'cm^3/(mol*s)'), n=0, Ea=(0, 'kcal/mol'), T0=(1, 'K'))
        k2 = Arrhenius(A=(1.01e14, 'cm^3/(mol*s)'), n=0, Ea=(0, 'kcal/mol'), T0=(1, 'K'))
        k3 = Arrhenius(A=(2e14, 'cm^3/(mol*s)'), n=0, Ea=(0, 'kcal/mol'), T0=(1, 'K'))
        assert ei.rates_match(ei.rate_fingerprint(k), ei.rate_fingerprint(k2))
        assert not ei.rates_match(ei.rate_fingerprint(k), ei.rate_fingerprint(k3))


@pytest.fixture
def index(tmp_path):
    models = str(tmp_path / 'RMG-models')
    earlier = write_model(models, os.path.join('Journal', 'Earlier'))
    current = write_model(models, os.path.join('Journal', 'Current'),
                          reactions=REACTIONS.replace('C2H4 + H <=> C2H5', 'C2H4 + H <=> C2H5 + H + H'))
    path = str(tmp_path / 'index.sqlite')
    assert ei.build(path, None, models) == 2
    return path, models, earlier, current


def rate(A):
    """Rate fingerprint of k = A cm^3/(mol*s), as in the test libraries."""
    return ei.rate_fingerprint(Arrhenius(A=(A, 'cm^3/(mol*s)'), n=0, Ea=(0, 'kcal/mol'), T0=(1, 'K')))


class TestIndex:

    def test_build_and_lookups(self, index):
        path, models, earlier, current = index
        idx = ei.EvidenceIndex(path)
        kinds, (n_thermo, n_reactions, n_structures) = idx.counts()
        assert kinds == {'imported': 2} and n_thermo == 2 and n_reactions == 4 and n_structures == 5
        assert len(idx.thermo_for_formulas(['CH4'])) == 2
        assert len(idx.reactions_for_signatures([ei.formula_signature(['CH3', 'H'], ['CH4'])])) == 2

    def test_library_votes_leave_out_the_model_being_imported(self, index):
        path, models, earlier, current = index
        idx = ei.EvidenceIndex(path, exclude_paths=[current])
        k_h, k_ch4 = key('[H]'), key('C')
        rate = ei.rate_fingerprint(Arrhenius(A=(1e14, 'cm^3/(mol*s)'), n=0, Ea=(0, 'kcal/mol'), T0=(1, 'K')))
        # CHEMKIN reaction 7: X + H = CH4, with X unidentified; and the same reaction written backwards
        chemkin = [(7, [('X', None, 'CH3'), ('H', k_h, 'H')], [('CH4', k_ch4, 'CH4')], rate),
                   (8, [('CH4', k_ch4, 'CH4')], [('X', None, 'CH3'), ('H', k_h, 'H')], None)]
        votes = ei.library_votes(idx, chemkin)
        assert set(votes) == {'X'}
        assert set(votes['X']) == {key('[CH3]')}
        evidence = votes['X'][key('[CH3]')]
        assert set(evidence) == {7}  # 8 has one source and no comparable rate, so it doesn't vote
        assert evidence[7][0]['sources'] == ['Journal/Earlier']  # not Journal/Current
        assert evidence[7][0]['rate_match'] is True
        evidence = ei.library_votes(idx, chemkin, min_sources=1)['X'][key('[CH3]')]
        assert set(evidence) == {7, 8}
        assert evidence[8][0]['rate_match'] is False  # written backwards, so not comparable

    def test_one_source_with_other_rates_needs_a_second_source(self, index):
        path, models, earlier, current = index
        k_h, k_ch4 = key('[H]'), key('C')
        other_rate = ei.rate_fingerprint(Arrhenius(A=(3e14, 'cm^3/(mol*s)'), n=0, Ea=(0, 'kcal/mol'), T0=(1, 'K')))
        chemkin = [(7, [('X', None, 'CH3'), ('H', k_h, 'H')], [('CH4', k_ch4, 'CH4')], other_rate)]
        assert ei.library_votes(ei.EvidenceIndex(path, exclude_paths=[current]), chemkin) == {}
        write_model(models, os.path.join('Journal', 'Another'))
        assert ei.build(path, None, models, refresh=True) == 1
        votes = ei.library_votes(ei.EvidenceIndex(path, exclude_paths=[current]), chemkin)
        evidence = votes['X'][key('[CH3]')][7]
        assert evidence[0]['sources'] == ['Journal/Another', 'Journal/Earlier']
        assert evidence[0]['rate_match'] is False

    def test_refresh_rereads_only_changed_sources(self, index):
        path, models, earlier, current = index
        assert ei.build(path, None, models, refresh=True) == 0
        time.sleep(1.1)  # fingerprints use whole-second modification times
        with open(os.path.join(earlier, 'RMG-Py-kinetics-library', 'reactions.py'), 'a') as f:
            f.write('\n')
        assert ei.build(path, None, models, refresh=True) == 1
        assert ei.EvidenceIndex(path).counts()[1][1] == 4


class TestCopiedChemistry:
    """Species identified by chemistry the mechanism copied, with nothing identified first."""

    # a CHEMKIN reaction with made-up labels: MET + HYD <=> METHANE
    SPECIES = {'MET': ('CH3', None), 'HYD': ('H', None), 'METHANE': ('CH4', None)}

    @staticmethod
    def reaction(A=1e14):
        return [(1, ['MET', 'HYD'], ['METHANE'], rate(A))]

    def test_a_copied_reaction_pairs_species_by_formula(self, index):
        path, models, earlier, current = index
        idx = ei.EvidenceIndex(path, exclude_paths=[current])
        found = ei.copied_chemistry(idx, self.SPECIES, self.reaction())
        assert [c.key for c in found['MET']] == [key('[CH3]')]
        assert found['MET'][0].reactions == {1: {'Journal/Earlier'}}  # not Journal/Current
        assert found['MET'][0].score == 1
        assert found['HYD'][0].key == key('[H]') and found['METHANE'][0].key == key('C')
        assert ei.copied_chemistry(idx, self.SPECIES, self.reaction(A=3e14)) == {}  # other rate constants

    def test_a_reaction_copied_into_many_sources_counts_once(self, index):
        path, models, earlier, current = index
        write_model(models, os.path.join('Journal', 'Another'))
        assert ei.build(path, None, models, refresh=True) == 1
        found = ei.copied_chemistry(ei.EvidenceIndex(path, exclude_paths=[current]), self.SPECIES, self.reaction())
        assert found['MET'][0].reactions == {1: {'Journal/Another', 'Journal/Earlier'}}
        assert found['MET'][0].score == 1

    def test_identified_species_constrain_the_pairing(self, index):
        path, models, earlier, current = index
        idx = ei.EvidenceIndex(path, exclude_paths=[current])
        found = ei.copied_chemistry(idx, self.SPECIES, self.reaction(), identified={'HYD': key('[H]')})
        assert set(found) == {'MET', 'METHANE'}
        # a library reaction that pairs an identified species with another structure isn't counted
        assert ei.copied_chemistry(idx, self.SPECIES, self.reaction(), identified={'HYD': key('[CH3]')}) == {}
        found = ei.copied_chemistry(idx, self.SPECIES, self.reaction(), blocked={('MET', key('[CH3]'))})
        assert 'MET' not in found

    def test_copied_thermo_and_names_add_to_the_score(self, index):
        from rmgpy.thermo import NASA, NASAPolynomial  # noqa: F401 (used by eval)
        path, models, earlier, current = index
        idx = ei.EvidenceIndex(path, exclude_paths=[current])
        values = ei.thermo_values(eval(CH4_NASA))
        species = {'CH4': ('CH4', values), 'ch3': ('CH3', None), 'X': ('H', None)}
        found = ei.copied_chemistry(idx, species, [(1, ['ch3', 'X'], ['CH4'], rate(1e14))])
        methane = found['CH4'][0]
        assert methane.thermo == {'Journal/Earlier'} and methane.names == {'Journal/Earlier'}
        assert methane.score == 1 + ei.THERMO_WEIGHT + ei.LABEL_WEIGHT
        assert found['ch3'][0].names == {'Journal/Earlier'}  # names are compared ignoring case
        assert set(ei.confident_proposals(found)) == {'CH4', 'ch3'}  # X has only the reaction
        species['CH4'] = ('CH4', [v * 1.02 for v in values])
        assert not ei.copied_chemistry(idx, species, [])['CH4'][0].thermo  # similar thermo isn't copied

    def test_confident_proposals(self):
        def candidate(key, n):
            c = ei.CopiedCandidate(key)
            c.reactions = {i: {'Source'} for i in range(n)}
            return c
        found = {'A': [candidate('k1', 3)],
                 'B': [candidate('k2', 3), candidate('k3', 3)],  # a tie
                 'C': [candidate('k4', 2)],                      # too little evidence
                 'D': [candidate('k5', 4)], 'E': [candidate('k5', 3)]}  # one structure, two labels
        assert set(ei.confident_proposals(found)) == {'A'}
