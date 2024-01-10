from glob import glob
from typing import List
import fire
import json
import logging
import openai
from langame import LangameClient
import os


def to_openai(
    in_files: List[str] = [],
    out_file: str = "openai.jsonl",
    validation_size: float = 0.2,
):
    """
    Convert a dataset of format [topics,] ### [starter] to openai fine tune format.
    :param in_files: List of files to read from and merge into one file.
    :param out_file: File to write to.
    """
    logger = logging.getLogger("prepare_openai.to_openai")
    logging.basicConfig(level=logging.INFO)
    logger.warning(
        "Assuming the input datasets rows are of format [topics,] ### [sentence]"
    )
    # Check if in_files is a list of strings
    if not isinstance(in_files, list):
        raise TypeError("in_files must be a list of strings")
    # Check if out file exists, if yes, ask user if he wants to overwrite it
    for f in glob(f"{out_file}_*"):
        logger.warning(f"{f} already exists, overwrite? [y/n]")
        if input() == "y":
            os.remove(f)
    total_size = 0
    for in_file in in_files:
        logger.info(f"processing {in_file}")
        with open(in_file) as f:
            lines = f.readlines()
            total_size += len(lines)
            for i, e in enumerate(lines):
                splitted = [j.strip() for j in e.split("###")]
                if len(splitted) != 2:
                    continue
                topics, starter = splitted
                if not topics or not starter:
                    continue
                # Append "train" or "validation" to the out_file depending on the validation_size.
                with open(
                    f"{out_file}_train.jsonl"
                    if i > (validation_size * total_size)
                    else f"{out_file}_validation.jsonl",
                    "a+",
                ) as out:
                    json.dump(
                        {
                            "prompt": "",
                            "completion": f" {topics} ### {starter}\n",
                        },
                        out,
                    )
                    out.write("\n")
    logger.info(f"done, output file {glob(f'{out_file}_*')}")
    # Print each file length
    for f in glob(f"{out_file}_*"):
        with open(f) as g:
            logger.info(f"{f} has {len(g.readlines())} lines")


def from_openai(
    in_file: str = "",
    out_file: str = "openai.jsonl",
):
    """
    Convert a file of openai fine tune format to [topics,] ### [starter] format.
    :param in_file: File to read from.
    :param out_file: File to write to.
    """
    logger = logging.getLogger("prepare_openai.from_openai")
    logging.basicConfig(level=logging.INFO)
    with open(in_file) as f:
        for e in f.readlines():
            json_data = json.loads(e)
            if "completion" not in json_data:
                continue
            splitted = [j.strip() for j in json_data["completion"].split("###")]
            if len(splitted) != 2:
                continue
            topics, starter = splitted
            if not topics or not starter:
                continue
            with open(out_file, "a+") as outfile:
                outfile.write(f"{topics} ### {starter}\n")
    logger.info(f"done, output file {out_file}")


def create_fine_tune(
    train_dataset_path: str = "",
    validation_dataset_path: str = "",
    model: str = "curie",
    description: str = "",
):
    """
    Create a fine-tuned OpenAI model.
    :param dataset_path: Path to the dataset.
    :param model: Model to use.
    """
    # Initialise OpenAI configuration.
    c = LangameClient()
    train = openai.File.create(file=open(train_dataset_path), purpose="fine-tune")
    validation = openai.File.create(
        file=open(validation_dataset_path), purpose="fine-tune"
    )

    f_t = openai.FineTune.create(
        training_file=train["id"],
        validation_file=validation["id"],
        model=model,
    )
    logger = logging.getLogger("prepare_openai.create_fine_tune")
    logging.basicConfig(level=logging.INFO)
    logger.info(f"done, model {f_t}")
    c._firestore_client.collection("models").document(f_t["id"]).set(
        {
            **f_t,
            "description": description,
        }
    )


if __name__ == "__main__":
    fire.Fire(
        {
            "to_openai": to_openai,
            "from_openai": from_openai,
            "create_fine_tune": create_fine_tune,
        }
    )
