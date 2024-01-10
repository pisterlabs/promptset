from coherence.wn import WordNetEvaluator
from unittest import TestCase
from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic
from nltk.corpus import reuters
from topic.topic import Topic


class TestTopicEvaluator(TestCase):
    def setUp(self):
        self.te = WordNetEvaluator()

    # def test_sim_words(self):
    #     self.assertEqual(self.te.sim_words("dog", "dog", self.te.path), 1.0)
    #     topic = Topic()
    #     topic.words_dist = [("cat", "0.1")] * 2
    #     topic.words_dist.extend([("walk", "0.2")]*3)
    #     self.assertEqual(self.te.evaluate_write(topic, 3, "wup"), (3.0, 1.0, 1.0, [1.0, 1.0, 1.0]))
    #     self.assertTrue(isinstance(self.te.evaluate_write(topic, 4, "lch")[0], float))
    #
    # def test_sim_words_ic(self):
    #     reuters_ic = wn.ic(reuters, False, 0.0)
    #     # self.assertTrue(isinstance(self.te.sim_words_ic("dog", "cat", reuters_ic, self.te.res), float))
    #     topic = Topic()
    #     topic.words_dist =  [("xxxxxxxxxxxxxxxxxxxxxxxxx.", 0.3), ("cat",0.2), ("rabbit", 0.15), ("table", 0.35)]
    #     ofile = open("test_ic.txt","w")
    #     rsum, rmean, rmedian, rlist = self.te.evaluate_ic_write(topic, 4, reuters_ic, "lin", ofile)
    #     self.assertTrue(isinstance(rsum, float))
        # self.assertTrue(isinstance(rmean, float))
        # self.assertTrue(isinstance(rmedian, float))
        # self.assertTrue(isinstance(rlist, list))
        # self.assertTrue(len(rlist) == 3)

    def test_hso(self):
        self.assertEqual(1, 1)
        a = wn.synsets("dog")[0]
        b = wn.synsets("dog")[0]
        self.assertEqual(self.te.hso(a, b), 6)

        c = wn.synset('domestic_cat.n.01')
        # (dog.n.01)-(domestic_animal.n.01)-(domestic_cat.n.01)  path = 2   turns =1
        self.assertEqual(self.te.hso(a,c), 6-2-1)

        d = wn.synset("domestic_animal.n.01")
        # (dog.n.01)-(domestic_animal.n.01) pth =1, turns =0
        self.assertEqual(self.te.hso(a,d), 6-1)

        e = wn.synset('kitty.n.04')
        # (dog.n.01)-(domestic_animal.n.01)-(domestic_cat.n.01)- ('kitty.n.04')  path = 3   turns =1
        self.assertEqual(self.te.hso(a, e), 6 - 3 -1 )

        f = wn.synset('cat.n.01')
        # (dog.n.01)-(canine.n.02)-(carnivore.n.01)- ('feline.n.01')- (cat.n.01) path =4  turns =1
        self.assertEqual(self.te.hso(a, f), 6-4-1)

        e = wn.synset("canine.n.02")  # path =1  turns = 0
        print self.te.hso(e, a)




