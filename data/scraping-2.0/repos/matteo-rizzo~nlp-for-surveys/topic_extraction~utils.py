from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from gensim import corpora
from gensim.models import CoherenceModel

from topic_extraction.classes.Document import Document
from topic_extraction.link_from_id import get_scopus_link


def expand_scores(keywords_dict: dict[int, list[tuple[str, float]]]) -> dict[int, list[str]]:
    """ Utility to pretty insert columns in dataframe """

    words_with_score: dict[int, list[str]] = {k: [f"{w} ({s:.3f})" for w, s in ws] for k, ws in keywords_dict.items()}

    # words_score = {f"{k}_scores": [s for _, s in ws] for k, ws in keywords_dict.items()}
    # words_with_score = {
    #     **{k: [w for w, _ in ws] for k, ws in keywords_dict.items()},
    #     **words_score
    # }
    return words_with_score


def load_yaml(path: str | Path) -> Any:
    """
    Load YAML as python dict

    @param path: path to YAML file
    @return: dictionary containing data
    """
    with open(path, encoding="UTF-8") as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    return data


def dump_yaml(data, path: str | Path) -> None:
    """
    Load YAML as python dict

    @param path: path to YAML file
    @param data: data to dump
    @return: dictionary containing data
    """
    with open(path, encoding="UTF-8", mode="w") as f:
        yaml.dump(data, f, Dumper=yaml.SafeDumper)


def save_csv_results(docs: list[Document],
                     themes: list[int], subjects: list[int], alt_subjects: list[int] | None,
                     theme_keywords: dict[int, list[tuple[str, float]]], subj_keywords: dict[int, list[tuple[str, float]]],
                     csv_path: str | Path,
                     papers_by_subject: dict[str, list[str]] = None,
                     agrifood_papers: list[int] = None, theme_probs: np.ndarray | None = None, subj_probs: list[float] | None = None,
                     write_ods: bool = True, file_suffix: str | None = None) -> None:
    """
    Save clustering results to CSV file

    :param docs: documents
    :param themes: 1st level of labels
    :param subjects: 2nd level of labels
    :param alt_subjects: 2nd level of labels fixed
    :param theme_keywords: keywords associated with 1st level topics
    :param subj_keywords: keywords associated with 2nd level topics
    :param agrifood_papers: agrifood topics from clustering
    :param csv_path: the path where to write results
    :param subj_probs: confidence for cluster assignment of subjects
    :param theme_probs: confidence for cluster assignment with themes
    :param write_ods: write all results in a single ODS sheet, or in three CSV files
    :param file_suffix: str to add to the filename as suffix
    """

    csv_path.mkdir(exist_ok=True, parents=True)

    ids = [str(d.id) for d in docs]

    assert theme_probs is None or len(theme_probs) == len(themes), f"Themes probabilities and assigned clusters have different sizes: {len(theme_probs)} - {len(themes)}"
    assert subj_probs is None or len(subj_probs) == len(subjects), f"Subjects probabilities and assigned clusters have different sizes: {len(subj_probs)} - {len(subjects)}"

    theme_df = pd.DataFrame(expand_scores(theme_keywords))
    subjects_df = pd.DataFrame(expand_scores(subj_keywords))

    a_args = dict()
    if agrifood_papers:
        a_args["agrifood"] = agrifood_papers
    if alt_subjects:
        a_args["alt_subjects"] = alt_subjects
    if theme_probs is not None:
        for a in range(theme_probs.shape[1]):
            a_args[f"theme_{a}_prob"] = theme_probs[:, a].round(3)
    if subj_probs is not None:
        a_args["subj_prob"] = [round(p, 3) for p in subj_probs]

    a_args["link"] = [get_scopus_link(s_id) for s_id in ids]
    classification_df = pd.DataFrame(dict(themes=themes, subjects=subjects, index=ids, **a_args)).sort_values(by=["subjects", "themes", "index"]).set_index("index")

    if write_ods:
        # Write Sheet with three tabbed sheets
        with pd.ExcelWriter(csv_path / f"all_results_{file_suffix}.ods", engine="odf") as exc_writer:
            classification_df.to_excel(exc_writer, sheet_name="classification", index=True)
            theme_df.to_excel(exc_writer, sheet_name="themes", index=False)
            subjects_df.to_excel(exc_writer, sheet_name="subjects", index=False)
    else:
        # Write three csv files
        classification_df.to_csv(csv_path / f"classification_{file_suffix}.csv", index=True)
        theme_df.to_csv(csv_path / f"themes_{file_suffix}.csv", index=False)
        subjects_df.to_csv(csv_path / f"subjects_{file_suffix}.csv", index=False)


def vector_rejection(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
     Compute vector rejection of a from b. This means that the new vector will be "a" minus projection of "a" on "b"

    :param a: 1D or 2D vector. If 2D rejection is done row-wise
    :param b: 1D or 2D vectors to reject from.
    :return: new vector having the components of a that are orthogonal to b
    """
    # Compute centroid of unwanted cluster
    if b.ndim > 1:
        b = np.sum(b, axis=0)
    assert b.ndim == 1, "Cannot reject multiple vectors"

    # Subtract projection onto the unwanted direction
    rejected_a = a - ((np.dot(a, b) / np.dot(b, b)).reshape(-1, 1) * b.reshape(1, -1))
    return rejected_a


def validate_coherence(topic_model, docs: list[Document]):
    # Preprocess documents
    texts = [d.body for d in docs]
    cleaned_docs = topic_model._preprocess_text(texts)

    # Extract vectorizer and analyzer from BERTopic
    vectorizer = topic_model.vectorizer_model
    analyzer = vectorizer.build_analyzer()

    # Extract features for Topic Coherence evaluation
    # words = vectorizer.get_feature_names_out()
    tokens = [analyzer(doc) for doc in cleaned_docs]
    dictionary = corpora.Dictionary(tokens)
    corpus = [dictionary.doc2bow(token) for token in tokens]
    topics = topic_model.get_topics()
    topics.pop(-1, None)
    topic_words = [[words for words, _ in topic_model.get_topic(topic)] for topic in range(len(set(topics)) - 1)]

    # Evaluate
    coherence_model_cv = CoherenceModel(topics=topic_words,
                                        texts=tokens,
                                        corpus=corpus,
                                        dictionary=dictionary,
                                        coherence="c_v")

    coherence_model_umass = CoherenceModel(topics=topic_words,
                                           texts=tokens,
                                           corpus=corpus,
                                           dictionary=dictionary,
                                           coherence="u_mass")

    cv = coherence_model_cv.get_coherence()
    umass = coherence_model_umass.get_coherence()

    print(cv, umass)
