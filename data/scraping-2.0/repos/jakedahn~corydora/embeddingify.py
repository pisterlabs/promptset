import os
import json
import click
import openai
import logging
from retrying import retry
from tqdm import tqdm


def retry_if_result_none(result):
    """Return True if we should retry (in this case when result is None), False otherwise"""
    return result is None


@retry(
    retry_on_result=retry_if_result_none,
    wait_exponential_multiplier=1000,
    wait_exponential_max=10000,
)
def get_embedding(text):
    # generate embedding for the text field
    embedding = (
        openai.Embedding.create(model="text-embedding-ada-002", input=text)
        .data[0]
        .embedding
    )

    return embedding


@click.command()
@click.argument("input_file")
@click.argument("output_file")
def process_file(input_file, output_file):
    openai.api_key = os.environ["OPENAI_API_KEY"]

    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Count the lines in the file for the progress bar
    num_lines = sum(1 for _ in open(input_file))

    with open(input_file, "r") as infile, open(output_file, "w") as outfile:
        for line in tqdm(infile, total=num_lines):
            data = json.loads(line)

            if os.path.exists("./embedded/" + data["id"]):
                logging.info(f"Item {data['id']} already exists in index. Skipping...")
                continue

            # get the embedding with retry + backoff
            embedding = get_embedding(data["text"])

            # log the id and length of the embedding
            logging.info(
                f'Generated embedding for id {data["id"]}, length: {len(embedding)}'
            )

            # add the embedding before the metadata key
            data = {
                "id": data["id"],
                "text": data["text"],
                "embedding": embedding,
                "metadata": data["metadata"],
            }

            json.dump(data, outfile)
            outfile.write("\n")

            # after upserting, write the itemid to a file in the ./indexed folder, so we know it's been indexed
            with open("./embedded/" + data["id"], "w") as f:
                f.write("")


if __name__ == "__main__":
    process_file()
