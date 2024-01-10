from unittest import TestCase

from coherence.umass import TopicCoherence


class TestTopicCoherence(TestCase):
    def setUp(self):
        self.corpus = [[(1, 3), (2, 3), (3, 5)], [(1, 2), (3, 4), (6, 2), (7, 1)], [(2, 3), (3, 1), (7, 4)]]
        self.dictc = []
        for doc in self.corpus:
            self.dictc.append(dict(doc))
        self.tc = TopicCoherence()

    def test_word_doc_frequency(self):
        self.assertEqual(self.tc.word_doc_freq(1, self.dictc), 2)
        self.assertEqual(self.tc.word_doc_freq(4, self.dictc), 0)
        self.assertEqual(self.tc.word_doc_freq(6, self.dictc), 1)

    def test_words_doc_cofreq(self):
        self.assertEqual(self.tc.words_doc_cofreq(1, 3, self.dictc), 2)
        self.assertEqual(self.tc.words_doc_cofreq(1, 6, self.dictc), 1)
        self.assertEqual(self.tc.words_doc_cofreq(2, 10, self.dictc), 0)
        self.assertEqual(self.tc.words_doc_cofreq(10, 11, self.dictc), 0)

    def test_topic_coherence(self):
        tlist = [3, 2, 6]
        print self.tc.coherence(tlist, self.dictc)
        self.assertEqual(round(self.tc.coherence(tlist, self.dictc), 3), -1.099)