import time
import spacy
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.meteor_score import meteor_score
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
import nltk
nltk.download('wordnet')
nltk.download('punkt')
from rouge_score import rouge_scorer
from openai import OpenAI
from dotenv import load_dotenv
import os
from IPython.display import display, HTML


class SummaryGrader:
    def __init__(self):
        self._model = SentenceTransformer('bert-base-nli-mean-tokens')
        load_dotenv()
        self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def _split_text(self, text:str)->list:
        """
        Split text into sentences
        Args:
            text: the text to be split

        Returns:
            a list of sentences
        """
        sentence_list = sent_tokenize(text)
        return sentence_list

    def _sentence2embedding(self, sentences: list[str]) -> np.ndarray:
        """
        Convert sentences to embeddings
        Args:
            sentences: a list of sentences

        Returns:
            a matrix of embeddings, each row is an embedding
        """
        embeddings = self._model.encode(sentences)
        return embeddings

    def _cosine_similarity(self, embed_text: np.ndarray, embed_summary: np.ndarray) -> np.ndarray:
        """
        Calculate the cosine similarities between sentences of summary and sentences of text
        Args:
            embed_text: embedding matrix of text sentences
                        each row is an embedding
            embed_summary: embedding matrix of summary sentences
                        each row is an embedding

        Returns:
            a matrix of cosine similarities
        """
        
        dot_prod = embed_summary @ embed_text.T # [i,j] is the dot product of summary sentence i and text sentence j
        norm = np.linalg.norm(embed_summary, axis=1, keepdims=True) @ np.linalg.norm(embed_text, axis=1, keepdims=True).T # [i,j] is the norm of summary sentence i and text sentence j
        return dot_prod / norm

    def _topk_related(self, sim_matrix: np.ndarray, k: int) -> np.ndarray:
        """
        Find the indices of top k related sentences in text for each sentence in summary
        Args:
            sim_matrix: cosine similarity matrix
            k: number of sentences to be selected

        Returns:
            a matrix of indices
        """
        return sim_matrix.argsort(axis=1)[:, -k:]

    def _checker(self, sens_text: list[str], sen_summary: str) -> bool:
        """
        Check if the sentence from the summary con be obtained from the sentence from the text.
        Args:
            sens_text: list of sentences from the text
            sen_summary: the sentence from the summary

        Returns:
            a tuple of (bool, float)
            bool: True if the sentence from the summary can be obtained from the sentence from the text
            float: the probability that the sentence from the summary can be obtained from the sentence from the text
                True: >0.5
                False: <0.5
        """


        source_text = ''.join(sens_text)

        prompt = f"""
        As a compliance officer at a financial institution, you're tasked with evaluating the accuracy of a summary sentence based on its alignment with source sentences from a financial document. Consider the following criteria carefully:

        1. The summary accurately reflects the content of the source sentences, especially numerical information.
        2. All named entities in the summary are present in the source sentences.
        3. Relationships between entities in the summary are consistent with those in the source sentences.
        4. The directional flow of relationships among named entities matches between the summary and source sentences.
        5. There are no factual discrepancies between the summary and source sentences.
        6. The summary does not introduce any entities not found in the source sentences.

        Your job is to determine if the summary adheres to these criteria. Answer "Yes" if it does, or "No" if it doesn't.

        Summary sentence: ```{sen_summary}```

        Source sentences: ```{source_text}```

        Final Answer (Yes/No only): 
        """

        response = self._client.chat.completions.create(
            model='gpt-4',
            messages=[{'role': "user", 'content': prompt}],
            max_tokens=1
        )

        res = response.choices[0].message.content.lower().capitalize()
        return True if res == 'Yes' else False

    def evaluate(self, text:str, summary:str, k:int) -> (float, list[int]):
        """
        evaluate the quality of the summary according to the given text
        Args:
            text: original text
            summary: summary to be evaluated
            k: number of sentences to be selected from the text

        Returns:
            a float number between 0 and 1, the higher the better
            a list of indices of sentences from the summary that are rejected by LLM
        """

        # split the text into sentences
        sens_text = self._split_text(text)
        # split the summary into sentences
        sens_summary = self._split_text(summary)

        # convert sentences to embeddings
        embed_text = self._sentence2embedding(sens_text)
        embed_summary = self._sentence2embedding(sens_summary)

        # calculate cosine similarity
        sim_matrix = self._cosine_similarity(embed_text, embed_summary)

        # find top k related sentences
        topk = self._topk_related(sim_matrix, k)

        # check if the sentence from the summary can be obtained from the sentence from the text
        denominator = 0
        numerator = 0
        sent_idx_rejected = []
        for idx, sen in enumerate(sens_summary):
            time.sleep(5)
            sens_text_selected = [sens_text[i] for i in topk[idx]]
            res = self._checker(sens_text_selected, sen)
            if res:
                numerator += 1
            else:
                sent_idx_rejected.append(idx)
            denominator += 1
        return numerator / denominator, sent_idx_rejected


class NER_comparison:
    def __init__(self):
        self._nlp = spacy.load('en_core_web_sm')
        self._NER_cat = ["PERSON", "ORG", "DATE", "GPE", "MONEY"]

    def extraction(self, text: str) -> set[str]:

        """Extract the name entities in the text

        Args:
            text (str): original text

        Returns:
            _type_: set of name entites
        """

        sample_summary_doc = self._nlp(text)
        entities = set()
        for ent in sample_summary_doc.ents:
            if ent.label_ in self._NER_cat:
                entities.add((ent.text))
        return entities

    def comparison_summary(self, original: set[str], summary: set[str]) -> (float, set[str]):
        """compare the name entities in summary with those in original text

        Args:
            original (_type_): name entities of original text
            set (_type_): name entities of summary

        Returns:
            _type_: the ratio of name entities in summary which in original text
        """
        res = summary - original
        return (1 - len(res) / len(summary), res)

    def comparison_original(self, original: set[str], summary: set[str]) -> (float, set[str]):
        """compare the name entities in original text with those in summary

        Args:
            original (_type_): name entities of original text
            set (_type_): name entities of summary

        Returns:
            _type_: the ratio of name entities in original text which in summary
        """
        res = original - summary
        return (1 - len(res) / len(original), res)

    def comparison_display(self, text: str, ents: set[str]) -> str:
        """highlight entites which are presented in the text

        Args:
            text (str): text
            ents (set[str]): name entities

        Returns:
            str: text with highlighted name entities
        """
        for entity in ents:
            text = text.replace(entity, f"**{entity}**")
        return text

    def process(self, original: str, summary: str) -> (float, float):
        """Get two ratio
        Args:
            original (str): original text
            summary (str): stummary
        Returns:
            (float, float): the ratio of name entities in summary which in original text,
                            the ratio of name entities in original text which in summary
        """
        original_ents = self.extraction(original)
        summary_ents = self.extraction(summary)
        summary_ratio = self.comparison_summary(original_ents, summary_ents)
        original_ratio = self.comparison_original(original_ents, summary_ents)
        return (summary_ratio[0], original_ratio[0])


def highlight_sent(text_list:list[str], indices:list[int], color:str='yellow'):
    """
    Highlight the sentences in the text
    Args:
        text: the text to be highlighted
        indices: the indices of sentences to be highlighted

    Returns:
        display the highlighted text sentence by sentence
    """
    highlighted_sentences = []
    for i, sentence in enumerate(text_list):

        if i in indices:
            highlighted_sentence = f"<div><span style='background-color: {color};'>{sentence}</span></div>"
        else:
            highlighted_sentence = f"<div>{sentence}</div>"

        highlighted_sentences.append(highlighted_sentence)

    final_text = "".join(highlighted_sentences)
    display(HTML(final_text))

def cos_similariy(original:str, summary:str, falsified_summary:str)->(float, float):
    """
    Calculate the cosine similarities between sentences of summary and sentences of text
        and falsified summary and sentences of text
    Args:
        original: original text
        summary: summary of original text
        falsified_summary: falsified summary of original text

    Returns:
        cosine similarities between sentences of summary and sentences of text
        cosine similarities between sentences of falsified summary and sentences of text
    """
    model = SentenceTransformer('bert-base-nli-mean-tokens')
    summary_good_embeddings = model.encode(summary).reshape(1,-1)
    summary_bad_embeddings = model.encode(falsified_summary).reshape(1,-1)
    original_embeddings = model.encode(original).reshape(1,-1)
    good_score = cosine_similarity(original_embeddings, summary_good_embeddings)
    bad_score = cosine_similarity(original_embeddings, summary_bad_embeddings)
    return (good_score.tolist()[0][0], bad_score.tolist()[0][0])

class Baseline:
    def __init__(self):
        self._model = SentenceTransformer('all-mpnet-base-v2')

    def cal_cos_similarity(self, summary: str, falsified_summary: str) -> float:
        """
        Calculate the cosine similarities between good summary and falsified summary 
        Args:
            summary: good summary of original text
            falsified_summary: falsified summary of original text

        Returns:
            cosine similarities between good summary and falsified summary
        """
        summary_good_embeddings = self._model.encode(summary).reshape(1,-1)
        summary_bad_embeddings = self._model.encode(falsified_summary).reshape(1,-1)
        score = cosine_similarity(summary_good_embeddings, summary_bad_embeddings)
        return score.tolist()[0][0]

    def cal_meteor_score(self, summary: str, falsified_summary: str) -> float:
        '''
        Calculate the meteor scores between good summary and falsified summary
        Args:
            summary: good summary of original text
            falsified_summary: falsified summary of original text

        Returns:
            meteor score between good summary and falsified summary
        '''
        score = meteor_score([summary.split()], falsified_summary.split())

        return score

    def cal_bleu_score(self, summary: str, falsified_summary: str) -> float:
        '''
        Calculate the bleu scores between good summary and falsified summary
        Args:
            summary: good summary of original text
            falsified_summary: falsified summary of original text

        Returns:
            bleu score between good summary and falsified summary
        '''
        summary_tokenized = nltk.word_tokenize(summary)
        falsified_summary_tokenized = nltk.word_tokenize(falsified_summary)

        score = sentence_bleu([summary_tokenized], falsified_summary_tokenized, weights=(0.5, 0.5, 0, 0))

        return score
    
    def cal_rouge2_score(self, summary: str, falsified_summary: str) -> float:
        '''
        Calculate the rouge2 scores between good summary and falsified summary
        Args:
            summary: good summary of original text
            falsified_summary: falsified summary of original text

        Returns:
            rouge2 score between good summary and falsified summary
        '''
        scorer = rouge_scorer.RougeScorer(['rouge2'], use_stemmer=True)
        score = scorer.score(summary, falsified_summary)

        return score['rouge2'].fmeasure
