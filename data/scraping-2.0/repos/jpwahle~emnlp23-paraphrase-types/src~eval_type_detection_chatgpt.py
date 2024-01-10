import argparse
import json
import os
import time

import numpy as np
import openai
from sklearn.metrics import f1_score
from tqdm import tqdm

ALL_TYPES = {
    "Derivational Changes",
    "Inflectional Changes",
    "Modal Verb Changes",
    "Spelling changes",
    "Change of format",
    "Same Polarity Substitution (contextual)",
    "Same Polarity Substitution (habitual)",
    "Same Polarity Substitution (named ent.)",
    "Converse substitution",
    "Opposite polarity substitution (contextual)",
    "Opposite polarity substitution (habitual)",
    "Synthetic/analytic substitution",
    "Coordination changes",
    "Diathesis alternation",
    "Ellipsis",
    "Negation switching",
    "Subordination and nesting changes",
    "Direct/indirect style alternations",
    "Punctuation changes",
    "Syntax/discourse structure changes",
    "Entailment",
    "Identity",
    "Non-paraphrase",
    "Addition/Deletion",
    "Change of order",
    "Semantic-based",
}


def load_data(filename):
    """Loads data from a file in JSON format.

    Args:
        filename (str): The path to the file to load.

    Returns:
        list: The loaded data as a list of dictionaries.

    Example:
        ```python
        filename = "data.json"
        data = load_data(filename)
        print(data)
        ```
    """

    with open(filename, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f.readlines()]
    return data


def classify(data, model_id, num_examples=100):
    """
    Classifies data using a given model.

    Args:
        data (list): The data to classify.
        model_id (str): The ID of the model to use for classification.
        num_examples (int, optional): The number of examples to classify. Defaults to 100.

    Returns:
        tuple: A tuple containing two lists: y_true and y_pred. y_true contains
        the true response labels, and y_pred contains the predicted response labels.

    Example:
        ```python
        data = [...]  # List of data instances
        model_id = "model123"
        num_examples = 100

        y_true, y_pred = classify(data, model_id, num_examples)
        print(y_true)
        print(y_pred)
        ```"""

    y_true = []
    y_pred = []

    for instance in tqdm(data[:num_examples]):
        user_message = instance["messages"][0]["content"]
        true_response_labels = set(instance["messages"][1]["content"].split(", "))

        # Call the API and retry if it fails
        while True:
            try:
                completion = openai.ChatCompletion.create(
                    model=model_id,
                    messages=[
                        {"role": "user", "content": user_message},
                    ],
                )
                break
            except Exception as e:
                print(e)
                time.sleep(5)

        predicted_response_labels = set(
            completion.choices[0].message["content"].split(", ")
        )

        if completion.choices[0].message["content"]:
            y_true.append(true_response_labels)
            y_pred.append(predicted_response_labels)

    return y_true, y_pred


def evaluate(y_true, y_pred):
    """Evaluates the performance of a classification model.

    Args:
        y_true (list): The true labels.
        y_pred (list): The predicted labels.

    Returns:
        tuple: A tuple containing the F1 score and accuracy.

    Example:
        ```python
        y_true = [[1, 0, 1], [0, 1, 0]]
        y_pred = [[1, 1, 0], [0, 1, 1]]

        f1, acc = evaluate(y_true, y_pred)
        print(f1)
        print(acc)
        ```
    """

    y_true_bin = [[1 if t in labels else 0 for t in ALL_TYPES] for labels in y_true]
    y_pred_bin = [[1 if t in labels else 0 for t in ALL_TYPES] for labels in y_pred]

    # Compute F1 score
    f1 = f1_score(y_true_bin, y_pred_bin, average="micro")

    # Convert lists to numpy arrays for easier calculations
    y_true_np = np.array(y_true_bin)
    y_pred_np = np.array(y_pred_bin)

    # Calculate per-class accuracy
    acc = np.mean(np.equal(y_true_np, y_pred_np).astype(int))

    return f1, acc


def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        help="The id of the model to use for detection.",
    )
    parser.add_argument(
        "--data_file",
        type=str,
        required=True,
        help="The path to the data file.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    openai.api_key = os.getenv("OPENAI_API_KEY")
    MODEL_ID = args.model_id
    DATA_FILE = args.data_file

    # Load data and predict
    test_data = load_data(DATA_FILE)
    y_true, y_pred = classify(test_data, MODEL_ID, num_examples=100)  # -1 for all
    f1, acc = evaluate(y_true, y_pred)

    print(f"Model: {MODEL_ID}")
    print(f"Eval set size: {len(y_pred)}")
    print(f"F1 Score: {f1:.2f}")
    print(f"Accuracy: {acc:.2f}")
