import abc
import json
import os
import pickle
import requests

import jsonlines
import openai
import numpy as np
import pymagnitude
import sklearn.neighbors
import tensorflow as tf
import tensorflow_hub as hub
import tqdm
import yaml


class ScoringModel(abc.ABC):
    def __init__(self, save_dir):
        self.save_dir = save_dir

    @abc.abstractmethod
    def train(self, question_data):
        pass

    def save(self):
        pass

    @abc.abstractmethod
    def score(self, response):
        pass


class FuzzyKeywordScorer(ScoringModel):
    def __init__(self, save_dir, threshold=0.75, verbose=True):
        super().__init__(save_dir)
        self.verbose = verbose

        os.makedirs(self.save_dir, exist_ok=True)
        if os.path.exists(f"{self.save_dir}/keywords.pth"):
            with open(f"{self.save_dir}/keywords.pth", "rb") as f:
                self.keywords = pickle.load(f)
        else:
            self.keywords = {}

        if not os.path.exists(f"{self.save_dir}/wiki-news-300d-1M-subword.magnitude"):
            self.download_word_vectors()

        self.wv = pymagnitude.Magnitude(
            f"{self.save_dir}/wiki-news-300d-1M-subword.magnitude"
        )
        self.threshold = threshold

    def download_word_vectors(self):
        url = "http://magnitude.plasticity.ai/fasttext/medium/wiki-news-300d-1M-subword.magnitude"
        file_size = int(requests.head(url).headers.get("content-length", 0))
        r = requests.get(url, stream=True)

        with open(f"{self.save_dir}/wiki-news-300d-1M-subword.magnitude", "wb") as f:

            with tqdm.tqdm(
                total=file_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                miniters=1,
            ) as pbar:

                for chunk in r.iter_content(chunk_size=32 * 1024):
                    f.write(chunk)
                    f.flush()
                    pbar.update(len(chunk))

    def train(self, question_data):
        self.keywords = {q_id: data["keywords"] for q_id, data in question_data.items()}

    def save(self):
        with open(f"{self.save_dir}/keywords.pth", "wb") as f:
            pickle.dump(self.keywords, f)

    def score(self, q_id, response):
        if len(self.keywords[q_id]) == 0:
            return 1

        response_words = response.strip().lower().split()
        scores = [
            max(
                [
                    max(self.wv.similarity(str(kw).lower(), response_words))
                    for kw in kw_group
                ]
            )
            for kw_group in self.keywords[q_id]
        ]
        final_score = int(min(scores) >= self.threshold)

        if final_score == 0:
            low_score_kws = [
                kw_group
                for kw_group, score in zip(self.keywords[q_id], scores)
                if score < self.threshold
            ]
            if self.verbose:
                print(f"Forgot to mention these keywords: {low_score_kws}")

        return final_score


class NNScorer(ScoringModel):
    def __init__(self, save_dir):
        super().__init__(save_dir)

        if os.path.exists(self.save_dir):
            self.embedding_model = tf.keras.models.load_model(
                f"{self.save_dir}/embedding_model.ckpt"
            )
            with open(f"{self.save_dir}/classifiers.pth", "rb") as f:
                self.classifiers = pickle.load(f)
        else:
            self.embedding_model = hub.load(
                "https://tfhub.dev/google/universal-sentence-encoder/4"
            )
            self.classifiers = {}

    def train(self, question_data):
        for q_id, data in tqdm.tqdm(question_data.items()):
            self.classifiers[q_id] = sklearn.neighbors.KNeighborsClassifier(1)
            self.classifiers[q_id].fit(
                self.embedding_model(data["X"]).numpy(), data["y"]
            )

    def save(self):
        os.makedirs(self.save_dir, exist_ok=True)
        tf.saved_model.save(
            self.embedding_model, f"{self.save_dir}/embedding_model.ckpt"
        )

        with open(f"{self.save_dir}/classifiers.pth", "wb") as f:
            pickle.dump(self.classifiers, f)

    def score(self, q_id, response):
        embedding = self.embedding_model([response]).numpy()
        return self.classifiers[q_id].predict(embedding).item()


class GPT3_Scorer(ScoringModel):
    def __init__(self, save_dir):
        super().__init__(save_dir)

        try:
            with open(f"{self.save_dir}/openai_ids.yaml") as f:
                self.openai_ids = yaml.safe_load(f)
        except FileNotFoundError:
            self.openai_ids = {}

    def train(self, question_data):
        os.makedirs(self.save_dir, exist_ok=True)

        # Write new classification files and upload them to openai API
        for qid, data in tqdm.tqdm(question_data.items()):
            fname = f"{self.save_dir}/{qid}.jsonlines"
            new_data = [
                {
                    "text": x,
                    "label": "Correct" if y == 1 else "Incorrect",
                    "metadata": {"qid": qid},
                }
                for x, y in zip(data["X"], data["y"])
            ]

            # Only update openai file if training data has changed for this question
            if os.path.exists(fname):
                with open(fname) as f:
                    old_data = [line.replace("\n", "") for line in f]
                if set(old_data) == set(json.dumps(d) for d in new_data):
                    continue

            # If this question's training data has changed, then update on openai API
            if qid in self.openai_ids:
                openai.File.delete(self.openai_ids[qid])
            with jsonlines.open(fname, mode="w") as writer:
                writer.write_all(new_data)
            with open(fname) as f:
                self.openai_ids[qid] = openai.File.create(
                    file=f, purpose="classifications"
                )["id"]

        with open(f"{self.save_dir}/openai_ids.yaml", "w") as f:
            yaml.dump(self.openai_ids, f)

    def score(self, q_id, response):
        try:
            label = openai.Classification.create(
                file=self.openai_ids[q_id],
                query=response,
                search_model="ada",
                model="curie",
                max_examples=3,
            ).label
        except:
            label = "Incorrect"
        return int(label == "Correct")


class CombinedScorer(ScoringModel):
    def __init__(self, phrase_scorer, keyword_scorer):
        self.phrase_scorer = phrase_scorer
        self.keyword_scorer = keyword_scorer

    def train(self, question_data):
        self.keywords_only = {
            q_id: data["keywords_only"] for q_id, data in question_data.items()
        }
        self.phrase_scorer.train(question_data)
        self.keyword_scorer.train(question_data)

    def save(self):
        self.phrase_scorer.save()
        self.keyword_scorer.save()

    def score(self, q_id, response):
        kw_score = self.keyword_scorer.score(q_id, response)

        # Return keyword score if question was 'keywords only' or if th
        # keyword score is 0 (since keyword score takes precedence over phrase
        # score)
        if kw_score == 0:
            return kw_score

        # Otherwise, return phrase score
        return self.phrase_scorer.score(q_id, response)


def new_scorer(root=".", verbose=True):
    return CombinedScorer(
        NNScorer(f"{root}/trained_models/nn_use4_scorer"),
        FuzzyKeywordScorer(
            f"{root}/trained_models/fuzzy_keyword_scorer", verbose=verbose
        ),
    )


always_incorrect = ["I don't know", "Not sure", "ok", "yes", "no", "skip", "idk"]


def get_train_data():
    data = {}

    for topic in os.listdir("../../questions/topics"):
        with open(f"../../questions/topics/{topic}/questions.yaml") as f:
            questions = yaml.safe_load(f)

        for q in questions:
            # Load answers and labels into numpy arrays
            incorrect = (
                q.get("IncorrectAnswers", [])
                + q["MultipleChoice"]["Incorrect"]
                + always_incorrect
            )
            correct = q.get("CorrectAnswers", []) + [q["MultipleChoice"]["Correct"]]
            X = np.array(incorrect + correct)
            y = np.array([0] * len(incorrect) + [1] * len(correct))

            # Shuffle the dataset
            perm = np.random.permutation(len(X))
            X = X[perm]
            y = y[perm]

            data[q["Id"]] = {
                "X": X,
                "y": y,
                "keywords": q.get("Keywords", []),
                "keywords_only": "KeywordsOnly" in q,
            }

    return data


def main():
    scorer = new_scorer()

    data = get_train_data()
    scorer.train(data)

    scorer.save()


if __name__ == "__main__":
    main()
