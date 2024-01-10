from unittest import TestCase
import logging
import tempfile
import shutil
from CBGM.genealogical_coherence import GenealogicalCoherence
from CBGM.pre_genealogical_coherence import Coherence
from CBGM import test_db
from CBGM.test_logging import default_logging

default_logging()
logger = logging.getLogger(__name__)

TEST_DATA = """# -*- coding: utf-8 -*-
# This is a made up data set, purely for testing.

from CBGM.populate_db import Reading, LacunaReading, AllBut
from CBGM.shared import UNCL, INIT, OL_PARENT

all_mss = set(['B', 'C', 'D', 'E'])

struct = {
    '21': {
        '2': [
            Reading('a', 'ηθελον', AllBut('B'), INIT),
            Reading('b', 'ηλθον', ['B'], 'a')],
        '6-8': [
            Reading('a', 'λαβειν αυτον', AllBut('C', 'D'), UNCL),
            Reading('b', 'αυτον λαβειν', ['C'], UNCL),
            LacunaReading(['D'])],
    },
    '22': {
        '3': [
            Reading('a', '', AllBut('C'), INIT),
            Reading('b', 'τε', ['C'], 'a')],
        '20': [
            Reading('a', 'ιδων', ['B'], 'b'),
            Reading('b', 'ειδον', ['C'], 'c'),
            Reading('c', 'ειδεν', ['D'], INIT),
            Reading('d', 'ειδως', ['E'], 'c')],
    },
    '23': {
        '1': [
            Reading('a', '', AllBut('C', 'B'), INIT),
            Reading('b', 'και', ['B'], 'a'),
            LacunaReading(['C'])],
        '4-10': [
            Reading('a', 'ηλθεν πλοιαρια εκ τιβεριαδος', ['B'], UNCL),
            Reading('b', 'ηλθεν πλοια εκ τιβεριαδος', ['C'], UNCL),
            Reading('c', 'ηλθεν πλοια εκ της τιβεριαδος', ['D', 'E'], UNCL)],
    }
}
"""


B_ROWS = [{'W2': 'E', 'NR': 1, 'EQ': 2, 'PASS': 6, 'W1<W2': 2, 'W1>W2': 0, 'UNCL': 1, 'NOREL': 1, 'D': '',
           'PERC1': 33.333333333333336, '_NR': 1, '_RANK': 1},
          {'W2': 'A', 'NR': 2, 'EQ': 1, 'PASS': 4, 'W1<W2': 2, 'W1>W2': 0, 'UNCL': 0, 'NOREL': 1, 'D': '',
           'PERC1': 25.0, '_NR': 2, '_RANK': 2},
          {'W2': 'D', 'NR': 3, 'EQ': 1, 'PASS': 5, 'W1<W2': 2, 'W1>W2': 0, 'UNCL': 1, 'NOREL': 1, 'D': '',
           'PERC1': 20.0, '_NR': 3, '_RANK': 3},
          {'W2': 'C', 'NR': 4, 'EQ': 0, 'PASS': 5, 'W1<W2': 2, 'W1>W2': 1, 'UNCL': 2, 'NOREL': 0, 'D': '',
           'PERC1': 0.0, '_NR': 4, '_RANK': 4}]



B_ROWS_AT_23_4_10 = [
    {'W2': 'E', 'NR': 1, 'EQ': 2, 'PASS': 6, 'W1<W2': 2, 'W1>W2': 0, 'UNCL': 1, 'NOREL': 1, 'D': '',
     'PERC1': 33.333333333333336, '_NR': 1, '_RANK': 1, 'READING': 'c', 'TEXT': 'ηλθεν πλοια εκ της τιβεριαδος'},
    {'W2': 'A', 'NR': 2, 'EQ': 1, 'PASS': 4, 'W1<W2': 2, 'W1>W2': 0, 'UNCL': 0, 'NOREL': 1, 'D': '',
     'PERC1': 25.0, '_NR': 2, '_RANK': 2, 'READING': None, 'TEXT': None},
    {'W2': 'D', 'NR': 3, 'EQ': 1, 'PASS': 5, 'W1<W2': 2, 'W1>W2': 0, 'UNCL': 1, 'NOREL': 1, 'D': '',
     'PERC1': 20.0, '_NR': 3, '_RANK': 3, 'READING': 'c', 'TEXT': 'ηλθεν πλοια εκ της τιβεριαδος'},
    {'W2': 'C', 'NR': 4, 'EQ': 0, 'PASS': 5, 'W1<W2': 2, 'W1>W2': 1, 'UNCL': 2, 'NOREL': 0, 'D': '',
     'PERC1': 0.0, '_NR': 4, '_RANK': 4, 'READING': 'b', 'TEXT': 'ηλθεν πλοια εκ τιβεριαδος'}]


C_TABLE = """  W2    	   NR    	    D    	  PERC1  	   EQ    	  PASS   	  W1<W2  	  W1>W2  	  UNCL   	  NOREL 
   A    	    1    	         	 33.333 	    1    	    3    	    2    	         	         	        
   D    	    2    	         	 25.000 	    1    	    4    	    2    	         	    1    	        
   E    	    3    	         	 20.000 	    1    	    5    	    1    	         	    2    	    1   """


class TestGenealogicalCoherence(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_db = test_db.TestDatabase(TEST_DATA)
        cls.tmpdir = tempfile.mkdtemp(__name__)
        Coherence.CACHE_BASEDIR = cls.tmpdir

    @classmethod
    def tearDownClass(cls):
        cls.test_db.cleanup()
        shutil.rmtree(cls.tmpdir)

    def test_generate(self):
        """
        Test that the generate function produces rows, and they
        contain the right data for w1=B.
        """
        coh = GenealogicalCoherence(self.test_db.db_file, 'B')
        self.assertFalse(coh.rows)
        coh.generate()
        self.assertEqual(B_ROWS, coh.rows)

    def test_set_variant_unit(self):
        """
        Check that the rows get correctly changed when we set the variant unit
        """
        coh = GenealogicalCoherence(self.test_db.db_file, 'B')
        coh.generate()
        self.assertEqual(B_ROWS, coh.rows)
        coh.set_variant_unit('23/4-10')
        self.assertEqual(B_ROWS_AT_23_4_10, coh.rows)

    def test_tab_delim_table(self):
        """
        Check that the human-readable table is correct
        """
        coh = GenealogicalCoherence(self.test_db.db_file, 'C')
        tab = coh.tab_delim_table()
        self.assertEqual(tab, C_TABLE)

    def test_potential_ancestors(self):
        """
        Check the lists of potential ancestors are right
        """
        coh = GenealogicalCoherence(self.test_db.db_file, 'C')
        anc = coh.potential_ancestors()
        self.assertEqual(anc, ['A', 'D', 'E'])

        coh = GenealogicalCoherence(self.test_db.db_file, 'E')
        anc = coh.potential_ancestors()
        self.assertEqual(anc, ['D', 'A'])

        coh = GenealogicalCoherence(self.test_db.db_file, 'D')
        anc = coh.potential_ancestors()
        self.assertEqual(anc, [])

    def test_parent_combinations(self):
        """
        Check the possible parent combinations are correct
        """
        coh = GenealogicalCoherence(self.test_db.db_file, 'E')
        coh.set_variant_unit('22/20')
        comb = coh.parent_combinations('b', 'c')

        self.assertEqual(len(comb), 2)
        self.assertEqual(len(comb[0]), 1)
        self.assertEqual(len(comb[1]), 1)

        self.assertEqual(comb[0][0].parent, 'D')
        self.assertEqual(comb[0][0].rank, 1)
        self.assertEqual(comb[0][0].perc, 80)
        self.assertEqual(comb[0][0].gen, 2)
        self.assertEqual(comb[0][0].prior, 1)
        self.assertEqual(comb[0][0].posterior, 0)
        self.assertEqual(comb[0][0].strength, 1)

        self.assertEqual(comb[1][0].parent, 'A')
        self.assertEqual(comb[1][0].rank, 2)
        self.assertEqual(comb[1][0].perc, 75)
        self.assertEqual(comb[1][0].gen, 2)
        self.assertEqual(comb[1][0].prior, 1)
        self.assertEqual(comb[1][0].posterior, 0)
        self.assertEqual(comb[1][0].strength, 1)
