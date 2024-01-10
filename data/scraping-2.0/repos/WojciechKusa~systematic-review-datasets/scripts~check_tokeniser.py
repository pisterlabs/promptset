import pandas as pd
import tiktoken
from datasets import load_dataset
from langchain.text_splitter import TokenTextSplitter


def encode_csmed_ft(dataset):
    tokenised_dataset = []
    for example in dataset:
        review = (
            example["review_title"]
            + "\n"
            + example["review_abstract"]
            + "\n"
            + example["review_criteria"]
        ).strip()
        paper = (
            example["title"]
            + "\n\n"
            + example["abstract"]
            + "\n\n"
            + example["main_text"]
        ).strip()

        tokenised_dataset.append(
            {
                "review": len(enc.encode(review)),
                "paper": len(enc.encode(paper)),
                "total": len(enc.encode(review + "\n\n" + paper)),
            }
        )

    return tokenised_dataset


if __name__ == "__main__":
    enc = tiktoken.encoding_for_model("gpt-4")

    csmed_ft = load_dataset(
        "../csmed/datasets/datasets/csmed_ft", name="csmed_ft_all_source"
    )

    max_review_size = 2048
    review_splitter = TokenTextSplitter(
        chunk_size=max_review_size,
        chunk_overlap=0,
        model_name="gpt-3.5-turbo-0301",
    )

    tokeniser_stats = {}
    for split in ["train", "validation", "test", "sample"]:
        reviews = []
        for example in csmed_ft[split]:
            review = f'{example["review_title"]} {example["review_criteria"]}'
            review = review_splitter.split_text(review)
            reviews.append(review)

        tokeniser_stats[split] = {
            "avg_chunks": sum(len(review) for review in reviews) / len(reviews),
            "median_chunks": sorted(len(review) for review in reviews)[
                len(reviews) // 2
            ],
            "max_chunks": max(len(review) for review in reviews),
            "min_chunks": min(len(review) for review in reviews),
            "%_more_than_1_chunk": sum(len(review) > 1 for review in reviews)
            / len(reviews),
        }
    df = pd.DataFrame(tokeniser_stats)
    print(df)

    print(
        df.to_latex(
            float_format="{:0.2f}".format,
            column_format="l" * len(df.columns),
            header=False,
        ).replace("_", " ")
    )
