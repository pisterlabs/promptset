import argparse
import json
import os
import time

import evaluate
import openai
from sklearn.metrics import accuracy_score
from tqdm import tqdm


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
        true_response_labels = (
            1 if "yes" in instance["messages"][1]["content"].lower() else 0
        )

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

        predicted_response_labels = (
            1 if "yes" in completion.choices[0].message["content"].lower() else 0
        )

        if completion.choices[0].message["content"]:
            y_true.append(true_response_labels)
            y_pred.append(predicted_response_labels)

    return y_true, y_pred


def eval_(y_true, y_pred):
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

        f1, acc = eval_(y_true, y_pred)
        print(f1)
        print(acc)
        ```
    """

    # Compute F1 score
    metric = evaluate.load("f1")

    f1 = metric.compute(predictions=y_pred, references=y_true, average="micro")

    acc = accuracy_score(y_pred, y_true)

    return acc, f1["f1"]


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
    y_true, y_pred = classify(test_data, MODEL_ID, num_examples=5000)  # -1 for all
    acc, f1 = eval_(y_true, y_pred)

    print(f"Model: {MODEL_ID}")
    print(f"Eval set size: {len(y_pred)}")
    print(f"F1 Score: {f1:.3f}")
    print(f"Accuracy: {acc:.3f}")
