import logging
import re
import signal
import string
from re import finditer
import nltk
import openai
import torch
from cffi.backend_ctypes import xrange
from nltk import word_tokenize, pos_tag
from nltk.corpus import stopwords
from sentence_transformers import util, SentenceTransformer
from spacy.matcher import Matcher
from spacy.util import filter_spans
from tqdm import tqdm

from bug_improving.event_extraction.placeholder import Placeholder

import spacy
import benepar

from bug_improving.utils.timeout_util import break_after
from config import SPACY_BATCH_SIZE


class SentUtil:
    # # tutorial for dependency parsing of spaCy: https://spacy.io/usage/linguistic-features#dependency-parse
    #
    # # NLP = spacy.load("en_core_web_sm", disable=["ner"])
    # NLP = spacy.load("en_core_web_sm")
    # NLP.add_pipe("merge_noun_chunks")
    # if spacy.__version__.startswith('2'):
    #     NLP.add_pipe(benepar.BeneparComponent("benepar_en3"))
    # else:
    #     NLP.add_pipe("benepar", config={"model": "benepar_en3"})

    SENT_LIST = None  # [sent (string), sent, ..., sent]
    SENT_HAS_CCONJ_LIST = None  # [True, False, ...,True]
    SENT_CONS_DOC_LIST = None  # [cons_doc, cons_doc, ..., cons_doc]

    @staticmethod
    def get_sent_has_cconj_list(sents):
        logging.warning("SpaCy NLP for pos ...")
        # SpacyModel.NLP.disable_pipes("benepar", "merge_noun_chunks")
        # logging.warning(NLPUtil.SPACY_NLP.pipe_names)
        # logging.warning(len(steps))
        pos_docs = NLPUtil.SPACY_NLP.pipe(sents, batch_size=SPACY_BATCH_SIZE, disable=["benepar", "merge_noun_chunks"])
        # list if step has cconj: True else: False
        logging.warning("get sents if has cconj list...")
        SentUtil.SENT_HAS_CCONJ_LIST = []
        # step_has_cconj_count = 0
        for doc in tqdm(pos_docs, ascii=True):
            doc_len = len(doc)
            if doc_len == 0:
                SentUtil.SENT_HAS_CCONJ_LIST.append(False)
            for token_index, token in enumerate(doc):
                if token.pos_ == "CCONJ":
                    SentUtil.SENT_HAS_CCONJ_LIST.append(True)
                    # step_has_cconj_count = step_has_cconj_count + 1
                    break
                if token_index == doc_len - 1:
                    SentUtil.SENT_HAS_CCONJ_LIST.append(False)

    @staticmethod
    def get_sent_cons_doc_list(sents):
        logging.warning("SpaCy NLP for Constituency Dependency...")
        NLPUtil.SPACY_NLP.enable_pipe("benepar")
        NLPUtil.SPACY_NLP.enable_pipe("merge_noun_chunks")
        logging.warning(NLPUtil.SPACY_NLP.pipe_names)
        # logging.warning(len(steps))
        SentUtil.SENT_CONS_DOC_LIST = NLPUtil.SPACY_NLP.NLP.pipe(sents, batch_size=SPACY_BATCH_SIZE)
        SentUtil.SENT_CONS_DOC_LIST = list(SentUtil.SENT_CONS_DOC_LIST)

    @staticmethod
    def extract_action_target_condition(sent):
        """
        merge phrases: https://spacy.io/api/pipeline-functions
        token.text:
        token.lemma_:
        token.tag_: Fine-grained part-of-speech tag.
        token.pos_:
        token.dep_:
        """
        # SentUtil.NLP.add_pipe("merge_noun_chunks")
        doc = NLPUtil.SPACY_NLP(sent)
        # for token in doc:
        #     print(token.text, token.lemma_, token.tag_, token.pos_, token.dep_)
        # get the action of the sentence
        root = [token for token in doc if token.head == token][0]
        # get the object of the sentence
        obj = [child for child in root.children if child.dep_ == "dobj" or child.dep_ == "nsubjpass"]
        if obj:
            obj = obj[0]
        else:
            obj = None
        # get the prep phrases of the sentence
        prep_phrases = SentUtil.extract_prep_phrases(doc)

        return root, obj, prep_phrases

    @staticmethod
    def extract_prep_phrases(doc):
        """
        PP -> PREP + NP
        Function to get PPs from a parsed document.
        """
        flag = True
        pps = []
        for token in doc:
            # Try this with other parts of speech for different subtrees.
            if token.pos_ == 'ADP':
                # print(token.subtree)
                pp = ' '.join([tok.orth_ for tok in
                               token.subtree])  # Verbatim text content (identical to `Token.text`). Exists mostly for consistency with the other attributes.
                # exclude the repetitive prep phrases
                for pp_in_pps in pps:
                    if pp in pp_in_pps:
                        flag = False
                        break
                if flag:
                    pps.append(pp)
                flag = True
        return pps

    # @staticmethod
    # def split_atomic_sents_by_benepar(sent):
    #     """
    #     Penn Treebank II Constituent Tags:
    #     http://www.surdeanu.info/mihai/teaching/ista555-fall13/readings/PennTreebankConstituents.html
    #     :param sent:
    #     :return:
    #     """
    #     # print(sent)
    #     atomic_sents = list()
    #     sents_list = list(SentUtil.NLP(sent).sents)
    #     if sents_list:
    #         doc = sents_list[0]
    #         # print(doc)
    #         sub_sents, index = SentUtil.find_cc_by_benepar(doc)
    #         # print(sub_sents)
    #         if not sub_sents or not index:
    #             return [sent]
    #
    #         cc = f" {sub_sents[index]} "
    #         left_index = index - 1
    #         right_index = index + 1
    #         if left_index < 0 or right_index >= len(sub_sents):
    #             return [sent]
    #         if str(sub_sents[index - 1]) == ",":
    #             cc = ',' + cc
    #             left_index -= 1
    #         if left_index < 0:
    #             return [sent]
    #         # infinite recursion
    #         # Go to twitter.com (or facebook, github)
    #         if str(sub_sents[index - 1]) == "(":
    #             return [sent]
    #
    #         left_part = sub_sents[left_index]
    #         right_part = sub_sents[right_index]
    #         # print(sent._.parse_string)
    #         # print(left_part)
    #         # print(right_part)
    #         left_sent = sent.replace(f"{cc}{right_part}", "")
    #         rigth_sent = sent.replace(f"{left_part}{cc}", "")
    #
    #         # atomic_sents = list()
    #         atomic_sents.extend(SentUtil.split_atomic_sents_by_benepar(left_sent))
    #         atomic_sents.extend(SentUtil.split_atomic_sents_by_benepar(rigth_sent))
    #
    #     return atomic_sents

    @staticmethod
    # @break_after(1)  # matching_rs = pattern.findall(text)  等待1s,还未运行结束，return None
    def split_atomic_sents_by_benepar(sent, sents_list=None):
        """
        SpacyModel.NLP need to enable "benepar" and "merge_noun_chunks"
        Penn Treebank II Constituent Tags:
        http://www.surdeanu.info/mihai/teaching/ista555-fall13/readings/PennTreebankConstituents.html
        :param sent:
        @param sents_list:
        @type sents_list:
        :return:
        """
        # print(sent)
        atomic_sents = list()
        if sents_list is None:
            # try:
            sents_list = list(NLPUtil.SPACY_NLP(sent).sents)
            # except Exception:
            #     pass
            # if sents_list is None:
            #     return [sent]

        if sents_list:
            doc = sents_list[0]
            doc_len = len(doc)
            for index, token in enumerate(doc):
                if token.pos_ == "CCONJ":
                    break
                if index == doc_len - 1:
                    return [sent]
            # print(doc)
            sub_sents, index = SentUtil.find_cc_by_benepar(doc)
            # print(sub_sents)
            if not sub_sents or not index:
                return [sent]

            cc = f" {sub_sents[index]} "
            left_index = index - 1
            right_index = index + 1
            if left_index < 0 or right_index >= len(sub_sents):
                return [sent]
            if str(sub_sents[index - 1]) == ",":
                cc = ',' + cc
                left_index -= 1
            if left_index < 0:
                return [sent]
            # infinite recursion
            # Go to twitter.com (or facebook, github)
            if str(sub_sents[index - 1]) == "(":
                return [sent]

            left_part = sub_sents[left_index]
            right_part = sub_sents[right_index]
            # print(sent._.parse_string)
            # print(left_part)
            # print(right_part)
            left_sent = sent.replace(f"{cc}{right_part}", "")
            rigth_sent = sent.replace(f"{left_part}{cc}", "")

            # atomic_sents = list()
            atomic_sents.extend(SentUtil.split_atomic_sents_by_benepar(left_sent))
            atomic_sents.extend(SentUtil.split_atomic_sents_by_benepar(rigth_sent))

        return atomic_sents

    @staticmethod
    def find_cc_by_benepar(sent):
        """
        doc = SentUtil.NLP(sent)
        sent = list(doc.sents)[0]
        find cc's layer
        :param doc:
        :return: subsents and cc's index
        """
        # print(list(sent._.children))
        sub_sents = list(sent._.children)
        # print(sub_sents)
        # print(type(sub_sents))
        for index, sub_sent in enumerate(sub_sents):
            tokens = list(sub_sent)
            if len(tokens) == 1 and tokens[0].pos_ == "CCONJ":
                # print(list(sub_sent))
                return sub_sents, index
                # left_part = sub_sents[index - 1]
                # right_part = sub_sents[index + 1]
        for sub_sent in sub_sents:
            _sub_sents, _index = SentUtil.find_cc_by_benepar(sub_sent)
            if _sub_sents and _index:
                return _sub_sents, _index
        return None, None


def time_out(b, c):
    raise TimeoutError


class NLPUtil:
    # use spacy instead https://explosion.ai/demos/displacy
    # spacy glossary https://github.com/explosion/spaCy/blob/master/spacy/glossary.py
    PATTERN_URL = [
        re.compile(
            r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))",
            flags=re.MULTILINE),
    ]

    PATTERNS_CLEAN = [
        # re.compile(r"[^a-zA-Z0-9]*"),
        re.compile(r"[^a-zA-Z]*"),
    ]

    PATTEN_SERIAL_NUMBER = [
        re.compile(r"^[\d]*[a-zA-Z]?[^a-zA-Z][\s]*")  # 1. / - / 1) / a)
    ]

    SPACY_NLP = None
    SBERT_MODEL = None
    # # SBert model, to do sentence embedding
    # # SENTENCE_TRANSFORMER = SentenceTransformer('all-MiniLM-L6-v2')
    # SENTENCE_TRANSFORMER = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    @staticmethod
    def load_spacy_model():
        # tutorial for dependency parsing of spaCy: https://spacy.io/usage/linguistic-features#dependency-parse

        # NLP = spacy.load("en_core_web_sm", disable=["ner"])
        NLP = spacy.load("en_core_web_sm")
        NLP.add_pipe("merge_noun_chunks")
        if spacy.__version__.startswith('2'):
            NLP.add_pipe(benepar.BeneparComponent("benepar_en3"))
        else:
            NLP.add_pipe("benepar", config={"model": "benepar_en3"})
        NLPUtil.SPACY_NLP = NLP

    @staticmethod
    def load_sbert_model():
        # # SBert model, to do sentence embedding
        # # SENTENCE_TRANSFORMER = SentenceTransformer('all-MiniLM-L6-v2')
        SENTENCE_TRANSFORMER = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        NLPUtil.SBERT_MODEL = SENTENCE_TRANSFORMER

    @staticmethod
    def replace_url_by_placeholder(text):
        """
        for regex into dead cycle, https://www.jianshu.com/p/e040b86e43d9
        :param text:
        :return:
        """
        for pattern in NLPUtil.PATTERN_URL:
            signal.signal(signal.SIGALRM, time_out)
            # 开启信号处理
            signal.alarm(1)
            try:
                text = pattern.sub(Placeholder.URL, text)
                signal.alarm(0)  # 关闭信号处理  # 这个一定要有，不会后续会报错timeoutError
            except TimeoutError:
                pass

        return text

    @staticmethod
    def remove_serial_number(text):
        """
        remove the begining of steps (1. /- /1) ...)
        :param text:
        :return:
        @todo: remove I in the begining of sentence, e.g. I created a WebExtensions with two locales (fr and en).
                                                        -> created a WebExtensions with two locales (fr and en).
        """
        for pattern in NLPUtil.PATTEN_SERIAL_NUMBER:
            signal.signal(signal.SIGALRM, time_out)
            # 开启信号处理
            signal.alarm(1)
            try:
                text = pattern.sub('', text)
                signal.alarm(0)  # 关闭信号处理  # 这个一定要有，不会后续会报错timeoutError
            except TimeoutError:
                pass

        return text

    @staticmethod
    def get_text_between_parenthesis(text):
        text_between_parenthesis = re.search(r'\((.*?)\)', text).group(1)
        return text_between_parenthesis

    @staticmethod
    def remove_text_between_parenthesis(text):

        text_between_parenthesis = re.search(r'\((.*?)\)', text)
        if text_between_parenthesis:
            text_between_parenthesis = text_between_parenthesis.group(1)
            text = text.replace(text_between_parenthesis, "")
            text = text.replace("(", "")
            text = text.replace(")", "")
        return text

    @staticmethod
    def is_non_alpha(text):
        """
        check if text only contains non-alphanumeric
        :param text:
        :return:
        """
        for pattern in NLPUtil.PATTERNS_CLEAN:
            matching_rs = pattern.fullmatch(text)  # matching result
            if matching_rs:
                return True
        return False

    @staticmethod
    def extract_steps(section):
        steps = section.splitlines()  # split section text into lines
        processed_steps = []
        for step in steps:
            if not NLPUtil.is_non_alphanumeric(step):
                processed_steps.append(step)

        return processed_steps

    @staticmethod
    def split_step_into_atomic_steps(nlp, step):
        """
        mine
        Penn Treebank II Constituent Tags:
        http://www.surdeanu.info/mihai/teaching/ista555-fall13/readings/PennTreebankConstituents.html
        :param nlp:
        :param step:
        :return:
        """
        atomic_steps = []
        doc = nlp(step)
        if not NLPUtil.exist_cc(doc):
            atomic_steps.append(step)
            return atomic_steps

        sent = list(doc.sents)[0]
        sub_sents, index = NLPUtil.find_cc(sent)
        cc = sub_sents[index]
        left_part = sub_sents[index - 1]
        right_part = sub_sents[index + 1]
        # print(sent._.parse_string)
        # print(left_part)
        # print(right_part)
        atomic_steps.append(step.replace(f" {cc} {right_part}", ""))
        atomic_steps.append(step.replace(f"{left_part} {cc} ", ""))

        return atomic_steps

        # print(sent._.labels)
        # print(sent._.parse_string)
        # print(list(sent._.constituents))
        # print(sent._.parent)
        # print(list(sent._.children))
        # print(list(sent._.children)[0]._.labels)
        # print(list(list(sent._.children)[0]._.children)[1]._.labels)

        # print(len(list(sent._.children)[4]))

        # print(len(list(sent._.children)[0]))
        # print(list(sent._.children)[0]._.labels)
        # print(list(sent._.children)[1]._.parse_string)
        # for token in list(sent._.children)[1]:
        #     print(token._.parse_string)

    @staticmethod
    def exist_cc(doc):
        """
        doc = nlp(sent)
        :param doc:
        :return:
        """
        for token in doc:
            if token.pos_ == "CCONJ":
                return True
        return False

    @staticmethod
    def find_cc(sent):
        """
        doc = nlp(sent)
        sent = list(doc.sents)[0]
        find cc's layer
        :param doc:
        :return: subsents and cc's index
        """
        # print(list(sent._.children))
        sub_sents = list(sent._.children)
        for index, sub_sent in enumerate(sub_sents):
            if NLPUtil.exist_cc(sub_sent):
                # print(list(sub_sent))
                if len(list(sub_sent)) == 1:
                    return sub_sents, index
                    # left_part = sub_sents[index - 1]
                    # right_part = sub_sents[index + 1]
                return NLPUtil.find_cc(sub_sent)

    @staticmethod
    def extract_noun_phrase(nlp, text):
        """
        get all the noun phrases, including nested phrases
        https://stackoverflow.com/questions/48925328/how-to-get-all-noun-phrases-in-spacy
        :param nlp:
        :param text:
        :return:
        """
        doc = nlp(text)
        noun_phrase = list()
        # print([chunk.text for chunk in doc.noun_chunks])

        for base_noun in doc.noun_chunks:
            # print(base_noun)
            # get base noun phrases
            # noun_phrase.append(base_noun.text)
            nested_noun = doc[base_noun.root.left_edge.i: base_noun.root.right_edge.i + 1]
            if nested_noun.text != base_noun.text:
                # print(nested_noun)
                # get nested noun phrases
                noun_phrase.append(nested_noun.text)
        return noun_phrase

    @staticmethod
    def extract_verb_phrase(nlp, text):
        pattern = [{'POS': 'VERB', 'OP': '?'},
                   {'POS': 'ADV', 'OP': '*'},
                   {'POS': 'AUX', 'OP': '*'},
                   {'POS': 'VERB', 'OP': '+'}]

        # instantiate a Matcher instance
        matcher = Matcher(nlp.vocab)
        matcher.add("Verb phrase", [pattern])

        doc = nlp(text)
        # call the matcher to find matches
        matches = matcher(doc)
        spans = [doc[start:end] for _, start, end in matches]

        return filter_spans(spans)

    @staticmethod
    def find_longest_common_substring(s1, s2):
        m = [[0] * (1 + len(s2)) for i in xrange(1 + len(s1))]
        longest, x_longest = 0, 0
        for x in xrange(1, 1 + len(s1)):
            for y in xrange(1, 1 + len(s2)):
                if s1[x - 1] == s2[y - 1]:
                    m[x][y] = m[x - 1][y - 1] + 1
                    if m[x][y] > longest:
                        longest = m[x][y]
                        x_longest = x
                else:
                    m[x][y] = 0
        return s1[x_longest - longest: x_longest]

    @staticmethod
    def find_longest_common_sentence(s1, s2):
        s1_words = s1.split(' ')
        s2_words = s2.split(' ')
        return ' '.join(NLPUtil.find_longest_common_substring(s1_words, s2_words))

    @staticmethod
    def get_pairs_with_cossim_by_decreasing(embeddings1, embeddings2):
        """
        pair -> {'index': [i, j], 'score': cosine_scores[i][j]}
        @param embeddings1:
        @type embeddings1:
        @param embeddings2:
        @type embeddings2:
        @return:
        @rtype:
        """
        # # Compute embedding for both lists
        # embeddings1 = NLPUtil.SENTENCE_TRANSFORMER.encode(sentences1, convert_to_tensor=True)
        # embeddings2 = NLPUtil.SENTENCE_TRANSFORMER.encode(sentences2, convert_to_tensor=True)
        # Compute cosine-similarits
        cosine_scores = util.cos_sim(embeddings1, embeddings2)
        # Find the pairs with the highest cosine similarity scores
        pairs_list = []
        for i in range(len(embeddings1)):
            pairs = []
            for j in range(len(embeddings2)):
                pairs.append({'index': [i, j], 'score': cosine_scores[i][j]})
                # print("{} \t\t {} \t\t Score: {:.4f}".format(sentences1[i], sentences2[j], cosine_scores[i][j]))
            # Sort scores in decreasing order
            pairs = sorted(pairs, key=lambda x: x['score'], reverse=True)
            pairs_list.append(pairs)
            # print(pairs)

        return pairs_list

    @staticmethod
    def get_top_1_pairs_with_cossim(pairs_list):
        """
        get top-1 pairs from return of get_pairs_with_cossim_by_decreasing
        @param pairs_list:
        @type pairs_list:
        @return:
        @rtype:
        """
        top_1_pairs = list()
        for pairs in pairs_list:
            index_pair = pairs[0]
            top_1_pairs.append(index_pair)
        return top_1_pairs

    @staticmethod
    def convert_paraphrase_mining_result_into_dict(paraphrases):
        """
        convert paraphrase_mining_result into index_pair_score_dict
        @param paraphrases: paraphrase_mining_result
        @type paraphrases: [(score, index1, index2), ..., ]
        @return: index_pair_score_dict {(index1, index2): score, ..., }
        @rtype: dict
        """
        index_pair_score_dict = dict()
        for paraphrase in paraphrases:
            score, p_i, p_j = paraphrase
            index_pair_score_dict[(p_i, p_j)] = index_pair_score_dict.get((p_i, p_j), score)
        return index_pair_score_dict

    ######################################
    @staticmethod
    def sentence_tokenize_by_nltk(paragraph):
        """
        分句
        :param paragraph:
        :return: sentences
        """
        sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        sentences = sent_tokenizer.tokenize(paragraph)
        return sentences

    @staticmethod
    def sentence_tokenize_by_spacy_batch(paragraph_list, batch_size=SPACY_BATCH_SIZE):
        """
        分句
        @param paragraph_list:
        @type paragraph_list:
        @param batch_size:
        @type batch_size:
        :return: sentences
        """
        # SpacyModel.NLP.disable_pipes("benepar", "merge_noun_chunks")
        # logging.warning(SpacyModel.NLP.pipe_names)
        # SentUtil.NLP.add_pipe('sentencizer')
        paragraph_sents = list()
        for doc in tqdm(NLPUtil.SPACY_NLP.pipe(paragraph_list, disable=["benepar", "merge_noun_chunks"], batch_size=batch_size), ascii=True):
            sents = [sent.text.strip() for sent in doc.sents]
            # print(doc.sents)
            # doc = SpacyModel.NLP(paragraph)
            paragraph_sents.append(sents)
        return paragraph_sents

    @staticmethod
    def sentence_tokenize_by_spacy(paragraph):
        """
        分句
        :param paragraph:
        :return: sentences
        """
        # SpacyModel.NLP.disable_pipes("benepar", "merge_noun_chunks")
        # logging.warning(SpacyModel.NLP.pipe_names)
        # SentUtil.NLP.add_pipe('sentencizer')
        doc = NLPUtil.SPACY_NLP(paragraph)
        sentences = [sent.text.strip() for sent in doc.sents]
        return sentences

    @staticmethod
    def lemmatize_by_spacy(sentence):
        # SpacyModel.NLP.disable_pipes("benepar", "merge_noun_chunks")
        sentence = NLPUtil.SPACY_NLP(sentence)
        words = []
        for word in sentence:
            # if word.pos_ != "PUNCT":
            words.append(str(word.lemma_))
        return words

    @staticmethod
    def lemmatize_by_nltk(sentence):
        """
        分词 词性标注 词形还原
        :param sentence:
        :return:
        """
        wnl = nltk.WordNetLemmatizer()
        for word, tag in pos_tag(word_tokenize(sentence)):
            if tag.startswith('NN'):
                yield wnl.lemmatize(word, pos='n')
            elif tag.startswith('VB'):
                yield wnl.lemmatize(word, pos='v')
            elif tag.startswith('JJ'):
                yield wnl.lemmatize(word, pos='a')
            elif tag.startswith('R'):
                yield wnl.lemmatize(word, pos='r')
            else:
                yield word
                # yield wnl.lemmatize(word)
        # print(word,tag)
        # for word in word_tokenize(sentence):
        #     yield wnl.lemmatize(word)

    @staticmethod
    def remove_stopwords(words):
        """
        去除停用词
        :param words:
        :return:
        """
        filtered_words = [word for word in words if word not in stopwords.words('english')]
        return filtered_words

    @staticmethod
    def remove_punctuation(sentence):
        """
        去除标点符号
        :param sentence:
        :return:
        """
        # sentence_p = "".join([char for char in sentence if char not in string.punctuation])
        sentence_p = ""
        for char in sentence:
            if char not in string.punctuation:
                sentence_p = sentence_p + char
            else:
                sentence_p = sentence_p + ' '
        return sentence_p

    @staticmethod
    def remove_number(token_list):
        token_list = list(filter(lambda x: not str(x).isdigit(), token_list))
        return token_list

    @staticmethod
    def camel_case_split(identifier):
        """
        驼峰分词
        :param identifier:
        :return:
        """
        matches = finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
        words = ""
        for m in matches:
            words = words + " " + m.group(0)
        return words.strip()

    @staticmethod
    def filter_by_pos_tag(words):
        """
        only keep noun and verb
        :param words:
        :return:
        """
        filtered_words = []
        for word, tag in pos_tag(words):
            if tag.startswith('NN') or tag.startswith('VB'):
                filtered_words.append(word)
        return filtered_words

    @staticmethod
    def filter_paragraph_by_pos_tag(paragraph):
        """
        in a paragraph, only keep noun and verb
        :param paragraph:
        :return:
        """
        filtered_paragraph = list()
        for sentence in paragraph:
            filtered_paragraph.append(NLPUtil.filter_by_pos_tag(sentence))
        return filtered_paragraph

    @staticmethod
    def preprocess(paragraph):
        """
        预处理
        1. 驼峰
        2. caselower
        3. sentence split
        4. remove punctuations
        5. 分词 词性标注 词形还原
        # 6. remove stopword
        # 7. remove number
        :param paragraph:
        :return:
        """
        # 去掉回车，换成空格
        paragraph = paragraph.replace('\n', ' ')
        # print(paragraph)
        # 驼峰
        paragraph = NLPUtil.camel_case_split(paragraph)
        # print(paragraph)

        # 变成小写表示
        paragraph = paragraph.lower()
        # print(paragraph)

        # 分句
        # sentences = NLPUtil.sentence_tokenize_by_nltk(paragraph)
        sentences = NLPUtil.sentence_tokenize_by_spacy(paragraph)

        # print(sentences)

        filtered_words_list = []
        for sentence in sentences:
            # print(sentence)
            # 去标点符号
            sentence_p = NLPUtil.remove_punctuation(sentence)
            # print(sentence_p)
            # 分词 词性标注 词形还原
            filtered_words_list.extend(NLPUtil.lemmatize_by_spacy(sentence_p))

            # words = NLPUtil.lemmatize_by_nltk(sentence_p)
            # # 去除停词
            # # words = NLPUtil.remove_stopwords(words)
            # # 去数字
            # # words = NLPUtil.remove_number(words)
            # for fword in words:
            #     # print(fword)
            #     filtered_words_list.append(fword)
        return filtered_words_list

    @staticmethod
    def get_embedding_by_openai(corpus, model="text-embedding-ada-002"):
        # text = text.replace("\n", " ")
        # return openai.Embedding.create(input=corpus, model=model)['data'][0]['embedding']
        response = openai.Embedding.create(input=corpus, model=model)
        corpus_embeddings = [d['embedding'] for d in response['data']]
        return torch.tensor(corpus_embeddings)


