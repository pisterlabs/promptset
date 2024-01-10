"""Generate frozen/fixed embeddings for GPT-3 with OpenAI's API."""
import os
import pickle as pkl  # noqa
import time
from typing import List, Optional

import click
import openai
import pandas as pd
import torch
from tqdm import tqdm

from sentencefmricomparison.constants import (
    OPENAI_API_KEY,
    OPENAI_DEFAULT_MODEL,
    PEREIRA_OUTPUT_DIR,
    PEREIRA_PERMUTED_SENTENCES_PATH,
)


@click.command()
@click.option(
    "--column_names",
    multiple=True,
    default=["center_sents", "paragraphs", "permuted_sents"],
)
@click.option("--pereira_input_file", default=PEREIRA_PERMUTED_SENTENCES_PATH, type=str)
@click.option("--openai_api_key", default=OPENAI_API_KEY, type=str)
@click.option("--openai_model", default=OPENAI_DEFAULT_MODEL, type=str)
@click.option("--output_dir", default=PEREIRA_OUTPUT_DIR, type=str)
def generate_gpt3_embeddings(
    column_names: List[str] = ["center_sents", "paragraphs", "permuted_sents"],  # noqa
    pereira_input_file: str = PEREIRA_PERMUTED_SENTENCES_PATH,
    openai_api_key: Optional[str] = OPENAI_API_KEY,
    openai_model: str = OPENAI_DEFAULT_MODEL,
    output_dir: str = PEREIRA_OUTPUT_DIR,
):
    """Generate frozen/fixed embeddings for the Pereira paragraphs/middle sentences/permuted paragraphs.

    :param column_names: Names of the columns in the input file that contain the paragraphs/middle sentences/permuted
        sentences, defaults to ["center_sents", "paragraphs", "permuted_sents"]
    :type column_names: List[str]
    :param pereira_input_file: Path to the Pereira input file containing all three types of paragraphs
    :type pereira_input_file: str
    :param openai_api_key: OpenAI API key specific to your account
    :type openai_api_key: str
    :param openai_model: OpenAI model to use for generating the embeddings
    :type openai_model: str
    :param output_dir: Output directory to save the embeddings to
    :type output_dir: str
    """
    # Load the input file
    pereira_paragraphs = pd.read_csv(pereira_input_file)

    # Set the OpenAI API key
    openai.api_key = openai_api_key

    # Generate the embeddings for all three types of paragraphs
    # The resulting embeds dictionary has the following structure: {sentence: embedding vector}
    embeds = {}
    for col in column_names:
        for _, row in tqdm(
            pereira_paragraphs.iterrows(), desc=f"Generating embeddings for {col}"
        ):
            # Sleep for 1 second to avoid hitting the OpenAI API rate limit
            time.sleep(1)
            # Generate the embedding for the current input by calling the OpenAI API
            embeds[row[col]] = torch.tensor(
                openai.Embedding.create(input=[row[col]], model=openai_model)["data"][
                    0
                ]["embedding"]
            )

    # Save the embeddings to the output directory
    with open(os.path.join(output_dir, "gpt3_embeds.pkl"), "wb") as f:
        pkl.dump(embeds, f)


@click.group()
def cli() -> None:
    """Generate embeddings for GPT-3 with OpenAI's API."""


if __name__ == "__main__":
    cli.add_command(generate_gpt3_embeddings)
    cli()
