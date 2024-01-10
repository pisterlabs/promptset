import tqdm
import json
import openai
import logging
import argparse

from collections import defaultdict

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--openai_api_key",
        default=None,
        type=str,
        required=True,
        help="API key to use GPT-3.",
    )
    parser.add_argument(
        "--eval_file",
        default=None,
        type=str,
        required=True,
        help="The dev/test set json file from which examples will be used for evaluation.",
    )
    parser.add_argument(
        "--out_prediction_file",
        default=None,
        type=str,
        required=True,
        help="Where to save the predictions.",
    )

    args = parser.parse_args()

    openai.api_key = args.openai_api_key

    # Get the test examples
    test_examples = [json.loads(ex) for ex in open(args.eval_file)]
    test_examples = group_continuations(test_examples)

    predictions = solve(openai, test_examples)

    # Save the predictions
    with open(args.out_prediction_file, "w") as f_out:
        for ex, pred in zip(test_examples, predictions):
            ex["predictions"] = [pred]
            f_out.write(json.dumps(ex) + "\n")


def group_continuations(examples):
    """
    Group continuations of the same narrative.
    Takes a list of {"narrative", "option1", "option2", "correctanswer"} and returns
    a list of {"narrative", "gold"}
    """
    golds = defaultdict(list)
    [golds[ex["narrative"]].append(ex[ex["correctanswer"]]) for ex in examples]
    examples = [{"narrative": narrative, "gold": curr_gold} for narrative, curr_gold in golds.items()]
    return examples


def solve(openai, test_examples):
    """
    Evaluate GPT-3 with zero-shot learning
    :param openai: the OpenAI API object
    :return: the predictions
    """
    predictions = [
        openai.Completion.create(
            engine="davinci",
            prompt=f"{ex['narrative'].replace('<b>', '').replace('</b>', '')}",
            temperature=0.7,
            max_tokens=20,
            top_p=0.9,
            frequency_penalty=0,
            presence_penalty=0,
            stop=["\n"]
        )['choices'][0]['text'].strip()
        for ex in tqdm.tqdm(test_examples)
    ]

    return predictions


if __name__ == "__main__":
    main()
