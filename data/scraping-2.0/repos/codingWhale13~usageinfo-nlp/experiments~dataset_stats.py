import argparse
import training.utils as utils
from helpers.review_set import ReviewSet
from langchain import PromptTemplate
from pathlib import Path
import json
import seaborn as sns
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="Dataset statistics")
    parser.add_argument(
        "dataset", type=str, help="dataset name to get the statistics for"
    )
    parser.add_argument(
        "tokenizer", type=str, help="Modelname for the tokenizer to use"
    )
    parser.add_argument(
        "--prompt_id",
        "-p",
        type=str,
        default=None,
        help="Model prompt. Default is no prompt, then just the review body is used",
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None, help="Output file for the statistics"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = utils.get_dataset_path(args.dataset)
    print(f"Dataset: {dataset}")
    rs = ReviewSet.from_files(dataset)
    _, tokenizer, max_length, _ = utils.initialize_model_tuple(args.tokenizer)
    lengths = []
    for review in rs.reviews.values():
        review = (
            review.get_prompt(args.prompt_id)
            if args.prompt_id
            else review["review_body"]
        )
        tokenized_review = tokenizer(review)["input_ids"]
        lengths.append(len(tokenized_review))

    print(f"Number of reviews: {len(rs.reviews)}")
    print(f"Average length: {sum(lengths) / len(rs.reviews)}")
    print(f"Median length: {sorted(lengths)[len(lengths) // 2]}")
    print(f"Max length: {max(lengths)}")
    print(f"Min length: {min(lengths)}")
    print(
        f"Number of reviews with length > {max_length}: {len([l for l in lengths if l > max_length])}"
    )
    if args.output:
        sns.displot(lengths)
        plt.savefig(args.output)


if __name__ == "__main__":
    main()
