import copy
import sys
import openai
import time
import numpy as np
import json

from logging import getLogger
from enum import Enum


def convert_config_dict(config_dict):
    r"""This function convert the str parameters to their original type."""
    check_set = (str, int, float, list, tuple, dict, bool, Enum)
    for key in config_dict:
        param = config_dict[key]
        if not isinstance(param, str):
            continue
        try:
            value = eval(
                param
            )  # convert str to int, float, list, tuple, dict, bool. use ',' to split integer values
            if value is not None and not isinstance(value, check_set):
                value = param
        except (NameError, SyntaxError, TypeError, ValueError):
            if isinstance(param, str):
                if param.lower() == "true":
                    value = True
                elif param.lower() == "false":
                    value = False
                else:
                    if "," in param:  # split by ',' if it is a string
                        value = []
                        for v in param.split(","):
                            if len(v) == 0:
                                continue
                            try:
                                v = eval(v)
                            except (NameError, SyntaxError, TypeError, ValueError):
                                v = v
                            value.append(v)
                    else:
                        value = param
            else:
                value = param
        config_dict[key] = value
    return config_dict


def load_cmd_line():
    """
    Load command line arguments
    :return: dict
    """
    cmd_config_dict = {}
    unrecognized_args = []
    if "ipykernel_launcher" not in sys.argv[0]:
        for arg in sys.argv[1:]:
            if not arg.startswith("--") or len(arg[2:].split("=")) != 2:
                unrecognized_args.append(arg)
                continue
            cmd_arg_name, cmd_arg_value = arg[2:].split("=")
            if (
                cmd_arg_name in cmd_config_dict
                and cmd_arg_value != cmd_config_dict[cmd_arg_name]
            ):
                raise SyntaxError(
                    "There are duplicate commend arg '%s' with different value." % arg
                )
            else:
                cmd_config_dict[cmd_arg_name] = cmd_arg_value
    if len(unrecognized_args) > 0:
        logger = getLogger()
        logger.warning(
            f"Unrecognized command line arguments(correct is '--key=value'): {' '.join(unrecognized_args)}"
        )
    cmd_config_dict = convert_config_dict(cmd_config_dict)
    cmd_config_dict["cmd_args"] = copy.deepcopy(cmd_config_dict)
    return cmd_config_dict


def chat(prompt, model="gpt-3.5-turbo", max_try=5, **model_params):
    """Set up the Openai API key using openai.api_key = api_key"""
    try:
        if "gpt" in model.lower():
            chat_completion = openai.ChatCompletion.create(
                model=model, messages=[{"role": "user", "content": prompt}], **model_params
            )
            content = chat_completion.choices[0].message["content"]
        else:
            import google.generativeai as palm
            content = palm.chat(prompt=[prompt], model=model).last
        if content is None:
            if max_try > 0:
                time.sleep(20)
                content = chat(prompt, model, max_try - 1)
            else:
                content = ""
        return content
    except:
        if max_try > 0:
            time.sleep(20)
            return chat(prompt, model, max_try - 1)
        else:
            raise Exception("Max try exceeded")


def get_history_candidate_prompt(news_df, behavior):
    history_news = news_df[news_df["news_id"].isin(behavior["history"].split())]
    cand_news_index = [i.split("-")[0] for i in behavior["impressions"].split()]
    cand_label = [i.split("-")[1] for i in behavior["impressions"].split()]
    # get candidate news from news_df and save them to a list with the same order as cand_news_index
    candidate_news = [
        news_df[news_df["news_id"] == i].iloc[0]["title"] for i in cand_news_index
    ]
    history_prompt = "\n".join(
        [f"H{i + 1}: {news}" for i, news in enumerate(history_news["title"].values)]
    )
    candidate_prompt = "\n".join(
        [f"C{i + 1}: {news}" for i, news in enumerate(candidate_news)]
    )
    return (
        history_prompt,
        candidate_prompt,
        ",".join([f"C{i + 1}" for i, l in enumerate(cand_label) if int(l)]),
    )


def calculate_metrics(true_pos, rank_hat):
    """
    true_pos: pos item ['C1', 'C3', 'C5']
    rank_hat: ranking list ['C2', 'C5', 'C1', 'C3', 'C4']
    return: auc, mrr, ndcg5, ndcg10
    """

    # Get the ranks of the true positives in rank_hat
    ranks = [rank_hat.index(item) + 1 if item in rank_hat else len(rank_hat) + 1 for item in true_pos]

    # AUC calculation
    num_negatives = len(rank_hat) - len(true_pos)
    num_better_ranks = sum([r for r in ranks if r <= len(rank_hat)])
    auc = (num_better_ranks - len(true_pos) * (len(true_pos) + 1) / 2) / (len(true_pos) * num_negatives)
    # MRR calculation
    mrr = 0
    for rank in ranks:
        if rank <= len(rank_hat):
            mrr += 1.0 / rank
    mrr /= len(true_pos)

    # DCG and NDCG calculation
    def dcg_at_k(r, k):
        r = np.asarray(r)[:k]
        return np.sum(r / np.log2(np.arange(2, r.size + 2)))

    def ndcg_at_k(r, k, method=0):
        dcg_max = dcg_at_k(sorted(r, reverse=True), k)
        if not dcg_max:
            return 0.
        return dcg_at_k(r, k) / dcg_max

    binary_relevance = [1 if i in true_pos else 0 for i in rank_hat]
    ndcg5 = ndcg_at_k(binary_relevance, 5)
    ndcg10 = ndcg_at_k(binary_relevance, 10)

    return auc, mrr, ndcg5, ndcg10


def evaluate_one(labels, predictions):
    """
    Compute nDCG@k and MRR for a single row in recommendation tasks.

    Parameters:
    - labels: list of true labels, like ['C1', 'C3', 'C5']
    - predictions: list of predicted labels in order, like ['C2', 'C5', 'C1', 'C3', 'C4']

    Returns:
    - ndcg5_at_k: nDCG@5 for the row
    - ndcg10_at_k: nDCG@10 for the row
    - mrr: MRR for the row
    """
    # Compute relevance scores for nDCG@k
    relevance = [1 if item in labels else 0 for item in predictions]

    def dcg_at_k(r, k):
        """Compute DCG@k for a single sample.
        Args:
        r: list of relevance scores in the order they were ranked
        k: number of results to consider
        Returns:
        DCG@k
        """
        r = np.asfarray(r)[:k]
        return np.sum(r / np.log2(np.arange(2, r.size + 2)))

    def ndcg_at_k(r, k):
        """Compute nDCG@k for a single sample.
        Args:
        r: list of relevance scores in the order they were ranked
        k: number of results to consider
        Returns:
        nDCG@k
        """
        dcg_max = dcg_at_k(sorted(r, reverse=True), k)
        if not dcg_max:
            return 0.0
        return dcg_at_k(r, k) / dcg_max

    # Calculate nDCG@k values
    ndcg5_at_k = ndcg_at_k(relevance, 5)
    ndcg10_at_k = ndcg_at_k(relevance, 10)

    # Helper function to compute MRR for a single row
    def compute_mrr(labels, predictions):
        for index, item in enumerate(predictions, 1):
            if item in labels:
                return 1.0 / index
        return 0

    # Calculate MRR
    mrr = compute_mrr(labels, predictions)
    return ndcg5_at_k, ndcg10_at_k, mrr


def load_api_key(json_path="openai_key.json"):
    return json.load(open(json_path))["api_key"]
