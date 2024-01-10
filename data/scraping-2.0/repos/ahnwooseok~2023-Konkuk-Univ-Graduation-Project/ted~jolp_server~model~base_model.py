import os
import openai
from dotenv import load_dotenv
from bertopic import BERTopic
from konlpy.tag import Mecab
from tqdm import tqdm
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from tika import parser
from bertopic.representation import OpenAI

prompt = """
I have a topic that contains the following documents: 
[DOCUMENTS]
The topic is described by the following keywords: [KEYWORDS]

Based on the information above, one word korean topic label in the following format:
topic: <topic label>
"""


load_dotenv()
OPEN_API_KEY = os.environ["gpt_api_key"]
openai.api_key = OPEN_API_KEY
representation_model = OpenAI(
    model="gpt-3.5-turbo", prompt=prompt, delay_in_seconds=10, chat=True
)


class CustomTokenizer:
    def __init__(self, tagger):
        self.tagger = tagger

    def __call__(self, sent):
        sent = sent[:1000000]
        word_tokens = self.tagger.morphs(sent)
        result = [word for word in word_tokens if len(word) > 1]
        return result


class BaseModel:
    def __init__(self):
        self.model = BERTopic(
            embedding_model="sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens",
            vectorizer_model=CountVectorizer(
                tokenizer=CustomTokenizer(Mecab()), max_features=3000
            ),
            nr_topics="auto",
            top_n_words=5,
            calculate_probabilities=True,
        )

    def fit_transform(self, documents):
        topics, probs = self.model.fit_transform(documents)
        return topics, probs
