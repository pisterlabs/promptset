import boto3

from .utils import bedrock, print_ww
from langchain.embeddings import BedrockEmbeddings
import numpy as np
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
from .embeddings import generate_embeddings_parallel
from .llm import llm_wrapper, save_llm_result
import os
import json
from .prompts import *
import time
import pandas as pd
from sklearn.cluster import OPTICS


def get_json_from_llm(llm_output: str) -> dict:
    try:
        start = llm_output.find("```")
        if start == -1:
            raise ValueError("No code block found")

        end = llm_output.find("```", start + 3)
        if end == -1:
            raise ValueError("No closing code block found")

        json_string = llm_output[start + 3 : end].strip()

        if json_string.startswith("json"):
            json_string = json_string[4:].strip()

        return json.loads(json_string)

    except ValueError as ve:
        raise ve
    except Exception as e:
        raise ValueError("Invalid or not JSON format") from e


def validate_comment_json(comment_json: dict):
    if "issues" not in comment_json:
        raise ValueError("No issues found in json")

    # We check that issues is a list
    if not isinstance(comment_json["issues"], list):
        raise ValueError("Issues is not a list")

    # We check that every issue has the required fields
    for issue in comment_json["issues"]:
        if "issue" not in issue:
            raise ValueError("Issue does not have the field issue")
        if "confidence" not in issue:
            raise ValueError("Issue does not have the field confidence")
        if "severity" not in issue:
            raise ValueError("Issue does not have the field severity")
        if "comment" not in issue:
            raise ValueError("Issue does not have the field comment")


def get_json_for_comment(product_name: str, comment: str, iter=0):
    try:
        completion = llm_wrapper(create_issue_extraction_prompt(product_name, comment))
        comment_json = get_json_from_llm(completion)
        validate_comment_json(comment_json)
        save_llm_result(create_issue_extraction_prompt(product_name, comment), completion)
        return comment_json
    except ValueError as ve:
        if iter > 25:
            return {"issues": []}
        # sleep for 5 seconds
        time.sleep(5)
        return get_json_for_comment(comment, iter + 1)

def get_cluster_name(cluster_items: list[str], iter=0):
    try:
        completion = llm_wrapper(create_cluster_summarizer_prompt(cluster_items))
        cluster_name_json = get_json_from_llm(completion)
        if "cluster" not in cluster_name_json:
            raise ValueError("No cluster name found in json")
        save_llm_result(create_cluster_summarizer_prompt(cluster_items), completion)
        return cluster_name_json["cluster"]
    except ValueError as ve:
        if iter > 25:
            return cluster_items[0] if len(cluster_items) > 0 else ""
        # sleep for 5 seconds
        time.sleep(5)
        return get_cluster_name(cluster_items, iter + 1)

def get_highlight(product_review: str, issue: str, iter=0):
    try:
        completion = llm_wrapper(create_highlight_prompt(product_review, issue))
        highlight = get_json_from_llm(completion)
        if "text" not in highlight:
            raise ValueError("No text found in json")
        if not(highlight["text"] in product_review):
            raise ValueError("Text not found in product review")
        save_llm_result(create_highlight_prompt(product_review, issue), completion)
        return highlight["text"]
    except ValueError as ve:
        if iter > 25:
            return product_review
        # sleep for 5 seconds
        time.sleep(5)
        return get_highlight(product_review, issue, iter + 1)

def get_json_for_comment_paralllel(product_name: str, comment: str):
    return get_json_for_comment(product_name, comment)


def compute_issues_for_reviews(product_reviews: pd.DataFrame, max_workers=100):
    # Create a list to store the futures
    futures = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for row in product_reviews.itertuples():
            future = executor.submit(get_json_for_comment_paralllel, row[1], row[5])
            futures.append(future)

        results = [future.result() for future in futures]

    # We now have json and we need to add the embeddings of the issue names for every single issue.
    # We flatten the issues into a list
    llm_outputs = results
    issues = []
    for llm_output in llm_outputs:
        issues.extend(llm_output["issues"])

    # Issue names
    issue_names = [issue["issue"] for issue in issues]

    # We generate the embeddings for the issue names
    embeddings_issue_names = generate_embeddings_parallel(issue_names)
    issue_name_to_embedding = dict(zip(issue_names, embeddings_issue_names))
    # We add the embeddings to the issues
    for llm_output in llm_outputs:
        for issue in llm_output["issues"]:
            issue["embedding"] = issue_name_to_embedding[issue["issue"]]

    return llm_outputs


def cluster_issues_for_reviews(product_reviews: pd.DataFrame):
    if "LLM_OUTPUT" not in product_reviews.columns:
        raise ValueError("LLM_OUTPUT column not found")

    # We flatten the issues into a list
    llm_outputs = product_reviews["LLM_OUTPUT"].tolist()
    # Get all issue names and their embeddings
    issues = []
    for llm_output in llm_outputs:
        issues.extend(llm_output["issues"])

    # If we have no issues, we return an empty dict
    if len(issues) == 0:
        return {}

    if len(issues) == 1:
        return {issues[0]["issue"]: [issues[0]["issue"]]}

    # Embeddings of issue names
    embeddings_issue_names = np.array([issue["embedding"] for issue in issues])

    # OPTICS Clustering
    clustering = OPTICS(
        metric="cosine",
        min_samples=min(4, len(embeddings_issue_names))
        if len(embeddings_issue_names) > 10
        else 2,
    ).fit(embeddings_issue_names)
    labels = clustering.labels_

    # We group the issues by cluster
    clusters = {}
    for i, label in enumerate(labels):
        if label == -1:
            continue
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(issues[i])

    # For each cluster, we give it a name (in parallel)
    clusters_issues = {}

    def process_cluster(cluster):
        cluster_items = [issue["issue"] for issue in clusters[cluster]]
        cluster_name = get_cluster_name(cluster_items)
        return cluster_name, cluster_items

    with ThreadPoolExecutor(max_workers=100) as executor:
        results = executor.map(process_cluster, clusters)

    for cluster_name, cluster_items in results:
        clusters_issues[cluster_name] = cluster_items

    return clusters_issues
