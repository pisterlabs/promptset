from unittest import TestCase
import logging
import os
import tempfile
import shutil
from CBGM import textual_flow
from CBGM import test_db
from CBGM.shared import INIT
from CBGM.genealogical_coherence import ParentCombination
from CBGM.pre_genealogical_coherence import Coherence
from CBGM.test_logging import default_logging

default_logging()
logger = logging.getLogger(__name__)

TEST_DATA = """# -*- coding: utf-8 -*-
# This is a made up data set, purely for testing.

from CBGM.populate_db import Reading, LacunaReading, AllBut
from CBGM.shared import UNCL, INIT, OL_PARENT

all_mss = set(['P75', '01', '02', '03', '05', '07', '011', '013', '017', '019', '021', '022', '028', '030', '031', '032', '034', '036', '037', '038', '044', '045', '047', '063', '091', '0141', '0211'])

struct = {
    '21': {
        '2': [
            Reading('a', 'ηθελον', AllBut('01', '091'), INIT),
            Reading('b', 'ηλθον', ['01'], 'a'),
            LacunaReading(['091'])],
        '6-8': [
            Reading('a', 'λαβειν αυτον', AllBut('05', '032', '044', '091'), UNCL),
            Reading('b', 'αυτον λαβειν', ['05', '032', '044'], UNCL),
            LacunaReading(['091'])],
        '20-24': [
            Reading('a', 'το πλοιον εγενετο', ['01', '07', '013', '017', '021', '028', '030', '031', '034', '036', '037', '038', '045', '047', '063', '0211'], UNCL),
            Reading('b', 'εγενετο το πλοιον', ['P75', '02', '03', '011', '019', '022', '032', '044', '0141'], UNCL),
            Reading('c', 'το πλοιον εγενηθη', ['05'], UNCL),
            LacunaReading(['091'])],
        '28-30': [
            Reading('a', 'της γης', AllBut('01', '0211', '091', 'P75'), UNCL),
            Reading('b', 'την γην', ['01', '0211'], UNCL),
            LacunaReading(['091', 'P75'])],
        '36': [
            Reading('a', 'υπηγον', AllBut('01', '091'), INIT),
            Reading('b', 'υπηντησεν', ['01'], 'a'),
            LacunaReading(['091'])],
    },
    '22': {
        '3': [
            Reading('a', '', AllBut('091', '0211'), INIT),
            Reading('b', 'τε', ['0211'], 'a'),
            LacunaReading(['091'])],
        '10': [
            Reading('a', 'ο', AllBut('038', '045', '091'), UNCL),
            Reading('b', '', ['038', '045'], UNCL),
            LacunaReading(['091'])],
        '12': [
            Reading('a', 'εστηκως', AllBut('01', '091'), INIT),
            Reading('b', 'εστως', ['01'], 'a'),
            LacunaReading(['091'])],
        '20': [
            Reading('a', 'ιδων', ['07', '011', '013', '017', '021', '028', '030', '031', '034', '036', '044', '045', '047', '063', '0141'], UNCL),
            Reading('b', 'ειδον', ['P75', '02', '03', '038'], UNCL),
            Reading('c', 'ειδεν', ['01', '05'], UNCL),
            Reading('d', 'ειδως', ['0211'], UNCL),
            LacunaReading(['091', '019', '022', '032', '037'])],

        '40-52': [
            Reading('a', 'εκεινο εις ο ενεβησαν οι μαθηται αυτου', AllBut('P75', '02', '03', '019', '022', '032', '044', '063', '091'), 'b'),
            Reading('b', '', ['P75', '02', '03', '019', '022', '032', '044', '063'], INIT),
            LacunaReading(['091'])],
        '40': [
            Reading('a', 'εκεινο', AllBut('P75', '02', '03', '05', '019', '022', '032', '044', '063', '0211', '091'), 'b'),
            Reading('b', '', ['05', '0211'], OL_PARENT),
            LacunaReading(['P75', '02', '03', '019', '022', '032', '044', '063', '091'])],
        '42': [
            Reading('a', 'εις', AllBut('P75', '02', '03', '019', '022', '032', '044', '034', '063', '091'), UNCL),
            Reading('b', '', ['034'], UNCL),
            LacunaReading(['P75', '02', '03', '019', '022', '032', '044', '063', '091'])],
        '46': [
            Reading('a', 'ενεβησαν', AllBut('P75', '02', '03', '019', '022', '032', '044', '063', '091', '037', '047'), OL_PARENT),
            Reading('b', 'ανεβησαν', ['037', '047'], 'a'),
            LacunaReading(['P75', '02', '03', '019', '022', '032', '044', '063', '091'])],
        '52': [
            Reading('a', 'αυτου', AllBut('P75', '01', '02', '03', '05', '019', '022', '032', '044', '063', '091'), OL_PARENT),
            Reading('b', 'του ιησου', ['01'], 'a'),
            Reading('c', 'ιησου', ['05'], 'a&b'),  # split parentage of both a and b
            LacunaReading(['P75', '02', '03', '019', '022', '032', '044', '063', '091'])],

        '60': [
            Reading('a', 'συνεισηλθε', AllBut('01'), INIT),
            Reading('b', 'συνεληλυθι', ['01'], 'a')],
        '61': [
            Reading('a', '', AllBut('02'), INIT),
            Reading('b', 'ο ιησους', ['02'], 'a')],
        '62-66': [
            Reading('a', 'τοις μαθηταις αυτου', AllBut('01'), 'b'),
            Reading('b', 'αυτοις', ['01'], INIT)],
        '68-70': [
            Reading('a', 'ο ιησους', AllBut('034'), INIT),
            Reading('b', '', ['034'], 'a')],
        '76': [
            Reading('a', 'πλοιαριον', ['07', '011', '013', '021', '028', '030', '034', '036', '037', '038', '045', '047', '063'], UNCL),
            Reading('b', 'πλοιον', ['P75', '01', '02', '03', '05', '017', '019', '022', '032', '044', '091', '0141', '0211'], UNCL),
            LacunaReading(['031'])],
        '80': [
            Reading('a', 'μονοι', AllBut('05', '047'), INIT),
            Reading('b', 'μονον', ['05', '047'], 'a')],
        '88': [
            Reading('a', 'απηλθον', AllBut('01', '038', '031'), UNCL),
            Reading('b', 'εισηλθον', ['038'], UNCL),
            Reading('c', '', ['01'], UNCL),
            LacunaReading(['031'])],
    },
    '23': {
        '1': [
            Reading('a', '', AllBut('022', '031'), INIT),
            Reading('b', 'και', ['022'], 'a'),
            LacunaReading(['031'])],
        '3': [
            Reading('a', '', ['P75', '03', '019', '091'], INIT),
            Reading('b', 'δε', AllBut('P75', '01', '03', '05', '019', '091', '031'), 'a'),
            LacunaReading(['031', '01', '05'])],
        '4-10': [
            Reading('a', 'ηλθεν πλοιαρια εκ τιβεριαδος', ['02', '07', '011', '013', '028', '030', '031', '034', '037', '038', '045', '063', '0211'], UNCL),
            Reading('b', 'ηλθεν πλοια εκ τιβεριαδος', ['P75'], UNCL),
            Reading('c', 'ηλθεν πλοια εκ της τιβεριαδος', ['03', '032'], UNCL),
            Reading('d', 'ηλθον πλοιαρια εκ τιβεριαδος', ['021', '047', '036', '091'], UNCL),
            Reading('e', 'ηλθον πλοιαρια εκ της τιβεριαδος', ['022'], UNCL),
            Reading('f', 'πλοιαρια ηλθον εκ τιβεριαδος', ['017'], UNCL),
            Reading('g', 'πλοια ηλθεν εκ τιβεριαδος', ['044'], UNCL),
            Reading('h', 'πλοιαρια εκ τιβεριαδος ηλθον', ['019'], UNCL),
            Reading('i', 'πλοια εκ τιβεριαδος ηλθεν', ['0141'], UNCL),
            LacunaReading(['01', '05'])],
        '2-10': [
            Reading('a', 'αλλα ηλθεν πλοιαρια', AllBut('01', '05'), UNCL),
            Reading('b', 'επελθοντων ουν των πλοιων', ['01'], UNCL),
            Reading('c', 'αλλων πλοιαρειων ελθοντων', ['05'], UNCL)],

       '12-16': [
            Reading('a', 'εγγυς του τοπου', AllBut('032', '01'), UNCL),
            Reading('b', 'εγγυς ουσης', ['01'], UNCL),
            Reading('c', '', ['032'], UNCL)],
        '20-22': [
            Reading('a', 'εφαγον τον', AllBut('01'), UNCL),
            Reading('b', 'και εφαγον', ['01'], UNCL)],
        '26-30,Fruit/5-6': [
            Reading('a', 'ευχαριστησαντος του κυριου', AllBut('05', '091', '02'), UNCL),
            Reading('b', 'ευχαριστησαντος του θεου', ['02'], 'a'),
            Reading('c', '', ['05', '091'], UNCL)],
    },
    '24': {
        '6': [
            Reading('a', 'ειδεν', AllBut('01', '013', '030'), UNCL),
            Reading('b', 'ειπεν', ['013'], 'a'),
            Reading('c', 'εγνω', ['030'], UNCL),
            LacunaReading(['01'])],
        '2-10': [
            Reading('a', 'οτε ουν ειδεν ο οχλος', AllBut('01'), UNCL),
            Reading('b', 'και ιδοντες', ['01'], UNCL)],
        '14': [
            Reading('a', 'ιησους', AllBut('038', '031', '063', '01', '013'), INIT),
            Reading('b', 'ο ιησους', ['038', '01'], 'a'),
            Reading('c', '', ['013'], 'a'),
            LacunaReading(['031', '063'])],
        '14-20': [
            Reading('a', 'ο ιησους ουκ εστιν εκει', AllBut('01', '031'), UNCL),
            Reading('b', 'ουκ ην εκει ο ιησους', ['01'], UNCL),
            LacunaReading(['031'])],
        '28': [
            Reading('a', 'αυτου', AllBut('01', '091', '063'), UNCL),
            Reading('b', '', ['01'], UNCL),
            LacunaReading(['091', '063'])],

        '30': [
            Reading('a', 'ενεβησαν', AllBut('P75', '01', '019', '05', '091', '063'), UNCL),
            Reading('b', 'ανεβησαν', ['P75', '01', '019'], UNCL),
            LacunaReading(['091', '063', '05'])],
        '31': [
            Reading('a', '', AllBut('030', '036', '063', '05', '0211', '091', '063'), INIT),
            Reading('b', 'και', ['030', '036', '0211'], 'a'),
            LacunaReading(['091', '063', '05'])],
        '32': [
            Reading('a', 'αυτοι', AllBut('05', '01', '028', '091', '063'), UNCL),
            Reading('c', '', ['01', '028'], UNCL),
            LacunaReading(['091', '063', '05'])],
        '30-32': [
            Reading('a', 'ενεβησαν αυτοι', AllBut('0211', '05', '091', '063'), UNCL),
            Reading('b', 'τοτε και αυτοι ενεβησαν', ['0211'], UNCL),
            LacunaReading(['091', '063', '05'])],
        '36': [
            Reading('a', 'τα', AllBut('05', '01', '091'), UNCL),
            Reading('b', 'το', ['01'], UNCL),
            LacunaReading(['091', '05'])],
        '38': [
            Reading('a', 'πλοια', AllBut('01', 'P75', '03', '05', '019', '022', '032', '044', '091', '063'), UNCL),
            Reading('b', 'πλοιαρια', ['P75', '03', '05', '019', '022', '032', '044'], UNCL),
            Reading('c', 'πλοιον', ['01'], UNCL),
            LacunaReading(['091', '063'])],
        '30-38': [
            Reading('a', 'ενεβησαν αυτοι εις τα πλοια', AllBut('05', '091', '063'), UNCL),
            Reading('b', 'ελαβον εαυτοις πλοιαρια', ['05'], UNCL),
            LacunaReading(['091', '063'])],

        '50-52': [
            Reading('a', 'τον ιησουν', AllBut('017', '091'), INIT),
            Reading('b', 'αυτον', ['017'], 'a'),
            LacunaReading(['091'])],
    }
}
"""


class TestTextualFlow(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_db = test_db.TestDatabase(TEST_DATA)
        cls.tmpdir = tempfile.mkdtemp(__name__)
        Coherence.CACHE_BASEDIR = cls.tmpdir

    @classmethod
    def tearDownClass(cls):
        cls.test_db.cleanup()
        shutil.rmtree(cls.tmpdir)

    def test_get_parents_P75(self):
        """
        Test the get_parents method for P75
        """
        ret = textual_flow.get_parents('23/3', 'P75', 'a', INIT, connectivity=["499"], db_file=self.test_db.db_file, min_strength=0)
        exp = {'499': [ParentCombination('A', 1, 93.33333333333333, 1, 1, 0, False)]}
        self.assertEqual(exp, ret)

    def test_get_parents_0211(self):
        """
        Test the get_parents method for 0211
        """
        ret = textual_flow.get_parents('22/3', '0211', 'b', 'a', connectivity=["499"], db_file=self.test_db.db_file, min_strength=0)
        exp = {'499': [ParentCombination('07', 2, 82.92682926829268, 2, 2, 1, False)]}
        self.assertEqual(exp, ret)

    def test_textual_flow(self):
        """
        Check the high-level textual_flow method works for simple inputs
        """
        ret = textual_flow.textual_flow(self.test_db.db_file, variant_units=['22/3'], connectivity=["499"],
                                        path=self.tmpdir)
        expected_path = os.path.join(self.tmpdir, 'c499/textual_flow_22_3_c499')
        self.assertEqual({'499': expected_path}, ret)
        with open("%s.dot" % expected_path) as f:
            dotdata = f.read()
        # The innards of graphviz dot files don't come out in a predictable order. But we can check that various
        # features are correct:
        start = 'strict digraph  {\n\tnode [label="\\N"];\n\tsubgraph cluster_legend {\n\t\tgraph [style=rounded];'
        self.assertEqual(start, dotdata[:len(start)])

        node_044 = '044\t [color="#b43f3f",\n\t\tfillcolor="#FF8A8A",\n\t\tlabel="044 (a)",\n\t\tstyle=filled];'
        self.assertIn(node_044, dotdata)

        edge_044_0141 = '044 -> 0141\t [color="#b43f3f",\n\t\tlabel="1 (89.2)",\n\t\tstyle=dotted];'
        self.assertIn(edge_044_0141, dotdata)

    def test_box_readings(self):
        """
        Check the high-level textual_flow method works with box_readings=True
        """
        ret = textual_flow.textual_flow(self.test_db.db_file, variant_units=['22/3'], connectivity=["499"],
                                        path=self.tmpdir, box_readings=True)
        expected_path = os.path.join(self.tmpdir, 'c499/textual_flow_22_3_c499')
        self.assertEqual({'499': expected_path}, ret)
        with open("%s_a.dot" % expected_path) as f:
            dotdata = f.read()

        # The innards of graphviz dot files don't come out in a predictable order. But we can check that various
        # features are correct:
        start = 'strict digraph  {\n\tnode [label="\\N"];\n\tsubgraph cluster_'  # might be legend or subgraph first
        self.assertEqual(start, dotdata[:len(start)])

        node_044 = '044\t\t [color="#b43f3f",\n\t\t\tfillcolor="#FF8A8A",\n\t\t\tlabel="044 (a)",\n\t\t\tstyle=filled];'
        self.assertIn(node_044, dotdata)

        edge_044_0141 = '044 -> 0141\t\t [color="#b43f3f",\n\t\t\tlabel="1 (89.2)",\n\t\t\tstyle=dotted];'
        self.assertIn(edge_044_0141, dotdata)

        self.assertIn("subgraph cluster_reading {", dotdata)