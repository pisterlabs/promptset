from statistics import mean

import numpy as np

from evaluation.scoring.core import *
from openai_api.openai_backend import DEFAULT_OPENAI_SIM_PARAMS

# NOTE: All metrics defined in this file have the same signature. This is required by `Metrics`.


def remove_duplicates(l: list, use_lowercase: bool = True) -> list:
    return (
        list(set([" ".join(item.lower().strip().split()) for item in l]))
        if use_lowercase
        else list(set(l))
    )


def custom_precision(
    predictions: list[str],
    references: list[str],
    comparator: str = "all-mpnet-base-v2",
    agg: callable = mean,
    use_lowercase: bool = True,
    openai_params: dict = DEFAULT_OPENAI_SIM_PARAMS,  # only needed for comparator "openai"
    modification: Optional[str] = None,  # options: "stem" or "lemmatize"
    similarity_metric: str = "cosine_relu",
) -> float:
    if len(predictions) == 0:
        return int(len(references) == 0)
    else:
        predictions = remove_duplicates(predictions, use_lowercase)
        similarities = [
            get_most_similar(
                label=prediction,
                options=references,
                comparator=comparator,
                use_lowercase=use_lowercase,
                openai_params=openai_params,
                modification=modification,
                similarity_metric=similarity_metric,
            )[0]
            for prediction in predictions
        ]
        return agg(similarities)


def custom_recall(
    predictions: list[str],
    references: list[str],
    comparator: str = "all-mpnet-base-v2",
    agg: callable = mean,
    use_lowercase: bool = True,
    openai_params: dict = DEFAULT_OPENAI_SIM_PARAMS,  # only needed for comparator "openai"
    modification: Optional[str] = None,  # options: "stem" or "lemmatize"
    similarity_metric: str = "cosine_relu",
) -> float:
    if len(references) == 0:
        return int(len(predictions) == 0)
    else:
        references = remove_duplicates(references, use_lowercase)
        similarities = [
            get_most_similar(
                label=reference,
                options=predictions,
                comparator=comparator,
                use_lowercase=use_lowercase,
                openai_params=openai_params,
                modification=modification,
                similarity_metric=similarity_metric,
            )[0]
            for reference in references
        ]
        return agg(similarities)


def custom_f1_score(
    predictions: list[str],
    references: list[str],
    comparator: str = "all-mpnet-base-v2",
    agg: callable = mean,
    use_lowercase: bool = True,
    openai_params: dict = DEFAULT_OPENAI_SIM_PARAMS,  # only needed for comparator "openai"
    modification: Optional[str] = None,  # options: "stem" or "lemmatize"
    similarity_metric: str = "cosine_relu",
) -> float:
    precision = custom_precision(
        predictions=predictions,
        references=references,
        comparator=comparator,
        agg=agg,
        use_lowercase=use_lowercase,
        openai_params=openai_params,
        modification=modification,
        similarity_metric=similarity_metric,
    )
    recall = custom_recall(
        predictions=predictions,
        references=references,
        comparator=comparator,
        agg=agg,
        use_lowercase=use_lowercase,
        openai_params=openai_params,
        modification=modification,
        similarity_metric=similarity_metric,
    )

    if precision == recall == 0:
        return 0
    else:
        return 2 * precision * recall / (precision + recall)


# the following "ak" scores were designed by Avetis and Konstantin


def custom_precision_ak(
    predictions: list[str],
    references: list[str],
    comparator: str = "all-mpnet-base-v2",
    use_lowercase: bool = True,
    openai_params: dict = DEFAULT_OPENAI_SIM_PARAMS,  # only needed for comparator "openai"
    modification: Optional[str] = None,  # options: "stem" or "lemmatize"
    similarity_metric: str = "cosine_relu",
) -> float:
    if len(predictions) == 0:
        return int(len(references) == 0)
    else:
        predictions = remove_duplicates(predictions, use_lowercase)
        similarities = [
            get_most_similar(
                label=prediction,
                options=references,
                comparator=comparator,
                use_lowercase=use_lowercase,
                openai_params=openai_params,
                modification=modification,
                similarity_metric=similarity_metric,
            )[0]
            for prediction in predictions
        ]
        if len(predictions) == 1:
            weights = np.array(
                [1]
            )  # if there is only one prediction, we don't need to calculate weights
        else:
            similarity_matrix = [
                [
                    get_similarity(
                        label_1=prediction_1,
                        label_2=prediction_2,
                        comparator=comparator,
                        use_lowercase=use_lowercase,
                        openai_params=openai_params,
                        modification=modification,
                        similarity_metric=similarity_metric,
                    )
                    for prediction_2 in predictions
                ]
                for prediction_1 in predictions
            ]
            weights = np.array(
                [
                    (1 - (sum(similarity_matrix[i])) / (len(similarity_matrix[i])))
                    for i in range(len(similarity_matrix))
                ]
            )

        return np.dot(similarities, weights) / sum(weights)


def custom_recall_ak(
    predictions: list[str],
    references: list[str],
    comparator: str = "all-mpnet-base-v2",
    use_lowercase: bool = True,
    openai_params: dict = DEFAULT_OPENAI_SIM_PARAMS,  # only needed for comparator "openai"
    modification: Optional[str] = None,  # options: "stem" or "lemmatize"
    similarity_metric: str = "cosine_relu",
) -> float:
    if len(references) == 0:
        return int(len(predictions) == 0)
    else:
        references = remove_duplicates(references, use_lowercase)
        similarities = [
            get_most_similar(
                reference,
                predictions,
                comparator=comparator,
                use_lowercase=use_lowercase,
                openai_params=openai_params,
                modification=modification,
                similarity_metric=similarity_metric,
            )[0]
            for reference in references
        ]
        if len(references) == 1:
            weights = np.array(
                [1]
            )  # if there is only one reference, we don't need to calculate weights
        else:
            similarity_matrix = [
                [
                    get_similarity(
                        label_1=reference_1,
                        label_2=reference_2,
                        comparator=comparator,
                        use_lowercase=use_lowercase,
                        openai_params=openai_params,
                        modification=modification,
                        similarity_metric=similarity_metric,
                    )
                    for reference_2 in references
                ]
                for reference_1 in references
            ]
            weights = np.array(
                [
                    (1 - (sum(similarity_matrix[i])) / (len(similarity_matrix[i])))
                    for i in range(len(similarity_matrix))
                ]
            )

        return np.dot(similarities, weights) / sum(weights)


def custom_f1_score_ak(
    predictions: list[str],
    references: list[str],
    comparator: str = "all-mpnet-base-v2",
    use_lowercase: bool = True,
    openai_params: dict = DEFAULT_OPENAI_SIM_PARAMS,  # only needed for comparator "openai"
    modification: Optional[str] = None,  # options: "stem" or "lemmatize"
    similarity_metric: str = "cosine_relu",
) -> float:
    precision = custom_precision_ak(
        predictions=predictions,
        references=references,
        comparator=comparator,
        use_lowercase=use_lowercase,
        openai_params=openai_params,
        modification=modification,
        similarity_metric=similarity_metric,
    )
    recall = custom_recall_ak(
        predictions=predictions,
        references=references,
        comparator=comparator,
        use_lowercase=use_lowercase,
        openai_params=openai_params,
        modification=modification,
        similarity_metric=similarity_metric,
    )

    if precision == recall == 0:
        return 0
    else:
        return 2 * precision * recall / (precision + recall)


from itertools import product
from collections import defaultdict

import numpy as np
from scipy.spatial.distance import euclidean
import pulp
import gensim
from sentence_transformers import SentenceTransformer, util
from scipy.stats import beta


def distance(
    label_1: str,
    label_2: str,
    comparator: str = "all-mpnet-base-v2",
    use_lowercase: bool = True,
    openai_params: dict = DEFAULT_OPENAI_SIM_PARAMS,  # only needed for comparator "openai"
    modification: Optional[str] = None,  # options: "stem" or "lemmatize"
    similarity_metric: str = "cosine_relu",  # alternative: "euclidean", "cosine"
):
    return float(
        1
        - get_similarity(
            label_1,
            label_2,
            comparator=comparator,
            use_lowercase=use_lowercase,
            openai_params=openai_params,
            modification=modification,
            similarity_metric=similarity_metric,
        )
    )


def log_distance(
    label_1: str,
    label_2: str,
    comparator: str = "all-mpnet-base-v2",
    use_lowercase: bool = True,
    openai_params: dict = DEFAULT_OPENAI_SIM_PARAMS,  # only needed for comparator "openai"
    modification: Optional[str] = None,  # options: "stem" or "lemmatize"
    similarity_metric: str = "cosine_relu",  # alternative: "euclidean", "cosine"
):
    if label_1 == label_2:
        return 0.0
    sim = max(
        0.000001,  # needed to ensure that log(0) is not called
        get_similarity(
            label_1,
            label_2,
            comparator=comparator,
            use_lowercase=use_lowercase,
            openai_params=openai_params,
            modification=modification,
            similarity_metric=similarity_metric,
        ),
    )

    return -np.log(sim) if sim < 1 else 0.0


def labels_to_fracdict(
    labels,
    comparator: str = "all-mpnet-base-v2",
    use_lowercase: bool = True,
    openai_params: dict = DEFAULT_OPENAI_SIM_PARAMS,  # only needed for comparator "openai"
    modification: Optional[str] = None,  # options: "stem" or "lemmatize"
    similarity_metric: str = "cosine_relu",
):
    labels = remove_duplicates(labels, use_lowercase)
    if len(labels) == 1:
        return {labels[0]: 1.0}
    tokendict = defaultdict(lambda: 0)
    for label_1 in labels:
        for label_2 in labels:
            tokendict[label_1] += distance(
                label_1,
                label_2,
                comparator=comparator,
                use_lowercase=use_lowercase,
                openai_params=openai_params,
                modification=modification,
                similarity_metric=similarity_metric,
            )
    totaldist = sum(tokendict.values())
    return {token: float(dist) / float(totaldist) for token, dist in tokendict.items()}


import re


def replace_symbols_with_spaces(input_string):
    # Use a regular expression to match non-alphanumeric characters except '&', and replace them with a space
    cleaned_string = re.sub(r"[^a-zA-Z0-9+&]", " ", input_string)
    return cleaned_string


def word_movers_similarity(
    predictions: list[str],
    references: list[str],
    comparator: str = "all-mpnet-base-v2",
    use_lowercase: bool = True,
    openai_params: dict = DEFAULT_OPENAI_SIM_PARAMS,  # only needed for comparator "openai"
    modification: Optional[str] = None,  # options: "stem" or "lemmatize"
    similarity_metric: str = "cosine_relu",
):
    if len(predictions) == 0 or len(references) == 0:
        if len(predictions) == 0 and len(references) == 0:
            return 1.0
        else:
            return 0.0

    predictions = [
        replace_symbols_with_spaces(prediction) for prediction in predictions
    ]
    references = [replace_symbols_with_spaces(reference) for reference in references]

    predictions = remove_duplicates(predictions, use_lowercase)
    references = remove_duplicates(references, use_lowercase)

    all_labels = list(set(predictions + references))

    pred_labels_buckets = labels_to_fracdict(
        predictions,
        comparator=comparator,
        use_lowercase=use_lowercase,
        openai_params=openai_params,
        modification=modification,
        similarity_metric=similarity_metric,
    )
    ref_labels_buckets = labels_to_fracdict(
        references,
        comparator=comparator,
        use_lowercase=use_lowercase,
        openai_params=openai_params,
        modification=modification,
        similarity_metric=similarity_metric,
    )

    T = pulp.LpVariable.dicts(
        "T_matrix", list(product(all_labels, all_labels)), lowBound=0
    )
    solver = pulp.COIN_CMD(msg=0)
    prob = pulp.LpProblem("WMD", sense=pulp.LpMinimize)
    prob += pulp.lpSum(
        [
            T[label_1, label_2]
            * log_distance(
                label_1,
                label_2,
                comparator=comparator,
                use_lowercase=use_lowercase,
                openai_params=openai_params,
                modification=modification,
                similarity_metric=similarity_metric,
            )
            for label_1, label_2 in product(all_labels, all_labels)
        ]
    )
    for token2 in ref_labels_buckets:
        prob += (
            pulp.lpSum([T[token1, token2] for token1 in pred_labels_buckets])
            == ref_labels_buckets[token2]
        )
    for token1 in pred_labels_buckets:
        prob += (
            pulp.lpSum([T[token1, token2] for token2 in ref_labels_buckets])
            == pred_labels_buckets[token1]
        )

    # supress prob solve output
    prob.solve(solver)

    dist = pulp.value(prob.objective) if pulp.value(prob.objective) else 0.0

    return np.exp(-1 * dist)
