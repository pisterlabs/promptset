import itertools
import os
import pickle

import nltk
import numpy as np
import openai
import pandas as pd
import requests
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from keras.models import load_model
from keras.utils import pad_sequences
from nltk.corpus import stopwords
from pydantic import BaseModel

from experiments.high_precision.high_precision_modeling import (
    exponent_neg_manhattan_distance,
    text_to_word_list,
)
from experiments.high_recall.document_level_vectorization.document_level_vectorization import (
    DocumentLevelVectorization,
)
from experiments.high_recall.tourbert_embeddings.tourbert_embeddings import (
    TourBERTEmbeddings,
)
from experiments.high_recall.word_level_vectorization.word_level_vectorization import (
    WordLevelVectorization,
)


def find_answer(question: str) -> str:
    df = pd.read_csv("../data/traveling_qna_dataset.csv", sep="\t")
    df.drop(columns=df.columns[0], axis=1, inplace=True)
    answer = df.loc[df["Question"] == question]["Answer"].tolist()

    return " ".join(answer)


def query(model, payload):
    response = requests.post(API_URL + model, headers=headers, json=payload)
    return response.json()


API_URL = "https://api-inference.huggingface.co/models/"
headers = {"Authorization": f"Bearer {os.environ['HF_API_TOKEN']}"}
app = FastAPI()

allowed_origins = [
    "http://localhost",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)


class Question(BaseModel):
    question: str
    model: str
    preprocessing: str
    weight: str


@app.post("/faq/default")
def get_answer(question: Question):
    candidates = preloaded_high_recall_model.get_n_similar_documents(question.question)
    candidates_ = candidates.copy()
    questions = [question.question for _ in range(len(candidates))]

    vocabulary = vocabulary_pretrained_wv

    for i, c in enumerate(candidates):
        candidates[i] = [
            vocabulary.get(word, 0)
            for word in text_to_word_list(c)
            if word not in stops
        ]
    for i, q in enumerate(questions):
        questions[i] = [
            vocabulary.get(word, 0)
            for word in text_to_word_list(q)
            if word not in stops
        ]
    candidates = pad_sequences(candidates, maxlen=212)
    questions = pad_sequences(questions, maxlen=212)

    high_precision_model = load_model(
        "../experiments/high_precision/model/malstm_model_pretrained_wv.h5",
        compile=False,
    )

    output = high_precision_model.predict([questions, candidates])
    output = list(itertools.chain.from_iterable(output))

    index_max = np.argmax(output)

    if output[index_max] < 0.7:
        return "Sorry, but I do not understand your question. Can you rephrase it and try again?"

    return find_answer(candidates_[index_max])


@app.post("/faq/customized")
def get_answer(question: Question):
    if question.model in ["custom", "pretrained"]:
        high_recall_model = WordLevelVectorization(
            train=False,
            n_neighbours=100,
            metric="cosine",
            logging=False,
            word_vectors=question.model,
            strategy="sum",
            weight=question.weight,
        )
    elif question.model in ["tf", "tf-idf"]:
        high_recall_model = DocumentLevelVectorization(
            n_neighbours=100,
            metric="cosine",
            logging=False,
            vectorizer_type=question.model,
            preprocessing=question.preprocessing,
            stemmer="snowball",
            stop_words="english",
            ngram_range=(1, 1),
        )
    else:
        high_recall_model = TourBERTEmbeddings(
            train=False,
            n_neighbours=100,
            metric="cosine",
            logging=False,
        )

    candidates = high_recall_model.get_n_similar_documents(question.question)
    candidates_ = candidates.copy()
    questions = [question.question for _ in range(len(candidates))]

    if question.model == "custom":
        vocabulary = vocabulary_custom_wv
    else:
        vocabulary = vocabulary_pretrained_wv

    for i, c in enumerate(candidates):
        candidates[i] = [
            vocabulary.get(word, 0)
            for word in text_to_word_list(c)
            if word not in stops
        ]
    for i, q in enumerate(questions):
        questions[i] = [
            vocabulary.get(word, 0)
            for word in text_to_word_list(q)
            if word not in stops
        ]

    if question.model == "custom":
        candidates = pad_sequences(candidates, maxlen=244)
        questions = pad_sequences(questions, maxlen=244)
    else:
        candidates = pad_sequences(candidates, maxlen=212)
        questions = pad_sequences(questions, maxlen=212)

    if question.model == "custom":
        high_precision_model = load_model(
            "../experiments/high_precision/model/malstm_model_custom_wv.h5",
            compile=False,
        )
    else:
        high_precision_model = load_model(
            "../experiments/high_precision/model/malstm_model_pretrained_wv.h5",
            compile=False,
        )

    output = high_precision_model.predict([questions, candidates])
    output = list(itertools.chain.from_iterable(output))

    index_max = np.argmax(output)

    if output[index_max] < 0.7:
        return "Sorry, but I do not understand your question. Can you rephrase it and try again?"

    return find_answer(candidates_[index_max])


@app.post("/faq/similarity")
def get_answer(question: Question):
    candidates = preloaded_high_recall_model.get_n_similar_documents(
        question.question, n_neighbours=1000
    )

    output = query(
        "sentence-transformers/msmarco-distilbert-base-tas-b",
        {"inputs": {"source_sentence": question.question, "sentences": candidates}},
    )

    if isinstance(output, dict) and "error" in output:
        return output["error"] + ". Try again in a few seconds."

    index_max = np.argmax(output)

    if output[index_max] < 0.7:
        return "Sorry, but I do not understand your question. Can you rephrase it and try again?"

    return find_answer(candidates[index_max])


@app.post("/faq/table_qa")
def get_answer(question: Question):
    candidates = preloaded_high_recall_model.get_n_similar_documents(question.question)
    answers = [find_answer(c) for c in candidates]

    output = query("google/tapas-base-finetuned-wtq", {
        "inputs": {
            "query": question.question,
            "table": {"Answer": answers}
        }
    })

    if isinstance(output, dict) and "error" in output:
        return output["error"] + ". Try again in a few seconds."

    return output["answer"]


@app.post("/faq/gpt2")
def get_answer(question: Question):
    output = query("gpt2", {
        "inputs": question.question,
        "parameters": {"max_length": 128},
    })

    if isinstance(output, dict) and "error" in output:
        return output["error"] + ". Try again in a few seconds."

    return output[0]["generated_text"]


@app.post("/faq/bloom")
def get_answer(question: Question):
    output = query("bigscience/bloom-560m", {
        "inputs": question.question,
        "parameters": {"max_length": 128},
    })

    if isinstance(output, dict) and "error" in output:
        return output["error"] + ". Try again in a few seconds."

    return output[0]["generated_text"]


@app.post("/faq/chat_gpt")
def get_answer(question: Question):
    openai.api_key = os.environ["OPENAI_API_TOKEN"]

    messages.append(
        {"role": "user", "content": question.question},
    )
    try:
        chat = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=messages
        )
    except Exception:
        return "Problem with OpenAI API. Check potential causes or try in a few minutes."

    reply = chat.choices[0].message.content
    messages.append({"role": "assistant", "content": reply})

    return reply


if __name__ == "__main__":
    nltk.download("stopwords")
    stops = set(stopwords.words("english"))

    with open(
        "../experiments/high_precision/vocabulary/vocabulary_custom_wv.pkl", "rb"
    ) as f:
        vocabulary_custom_wv = pickle.load(f)
    with open(
        "../experiments/high_precision/vocabulary/vocabulary_pretrained_wv.pkl",
        "rb",
    ) as f:
        vocabulary_pretrained_wv = pickle.load(f)

    preloaded_high_recall_model = WordLevelVectorization(
        train=False,
        n_neighbours=100,
        metric="cosine",
        logging=False,
        word_vectors="pretrained",
        strategy="sum",
        weight=None,
    )

    messages = [{"role": "system", "content": "You are a intelligent assistant."}]

    uvicorn.run(app, host="127.0.0.1", port=8000)
