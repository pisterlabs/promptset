import argparse
import json
import os
import time

import openai
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge import Rouge
from tqdm import tqdm

# Install required libraries first
# pip install rouge nltk sacrebleu


# Load data
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


def generate_paraphrases(data, model_id, num_examples=3):
    """Generates paraphrases using a given model.

    Args:
        data (list): The data to generate paraphrases for.
        model_id (str): The ID of the model to use for paraphrase generation.
        num_examples (int, optional): The number of paraphrases to generate. Defaults to 3.

    Returns:
        list: The generated paraphrases.

    Example:
        ```python
        data = [...]  # List of data instances
        model_id = "model123"
        num_examples = 3

        paraphrases = generate_paraphrases(data, model_id, num_examples)
        print(paraphrases)
        ```
    """

    paraphrases = []
    for instance in tqdm(data[:num_examples]):
        user_message = instance["messages"][0]["content"]
        while True:
            try:
                completion = openai.ChatCompletion.create(
                    model=model_id,
                    messages=[
                        {"role": "user", "content": user_message},
                    ],
                )
                paraphrases.append(completion.choices[0].message["content"])
                break
            except Exception as e:
                print(e)
                time.sleep(5)
    return paraphrases


def evaluate(paraphrases, references):
    """
    Evaluates the quality of paraphrases compared to reference texts.

    Args:
        paraphrases (list): The generated paraphrases.
        references (list): The reference texts.

    Returns:
        dict: A dictionary containing the evaluation scores.

    Example:
        ```python
        paraphrases = ["Paraphrase 1", "Paraphrase 2"]
        references = ["Reference 1", "Reference 2"]

        scores = evaluate(paraphrases, references)
        print(scores)
        ```
    """

    rouge = Rouge()

    # ROUGE scores
    rouge_scores = rouge.get_scores(paraphrases, references, avg=True)

    # BLEU scores
    smoothie = SmoothingFunction().method4
    bleu_scores = [
        sentence_bleu([ref], paraphrase, smoothing_function=smoothie)
        for ref, paraphrase in zip(references, paraphrases)
    ]
    avg_bleu = sum(bleu_scores) / len(bleu_scores)

    return {
        "ROUGE-1": rouge_scores["rouge-1"]["f"],
        "ROUGE-2": rouge_scores["rouge-2"]["f"],
        "ROUGE-L": rouge_scores["rouge-l"]["f"],
        "BLEU": avg_bleu,
    }


def parse_args():
    """
    Parses command line arguments.

    Returns:
        argparse.Namespace: The parsed command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate the quality of generated paraphrases."
    )
    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        help="The id of the model to use for generation.",
    )
    parser.add_argument(
        "--data_file", type=str, required=True, help="The path to the data file."
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    openai.api_key = os.getenv("OPENAI_API_KEY")
    MODEL_ID = args.model_id
    DATA_FILE = args.data_file
    NUM_EXAMPLES = 100  # -1 for all

    # Load data and predict
    data = load_data(DATA_FILE)
    references = [item["messages"][1]["content"] for item in data[:NUM_EXAMPLES]]
    generated_paraphrases = generate_paraphrases(data, MODEL_ID, NUM_EXAMPLES)
    scores = evaluate(generated_paraphrases, references)
    print(scores)
