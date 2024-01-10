import os
import pathlib
from datetime import datetime
from typing import List, Tuple

import datasets
import hydra
import openai
import pandas as pd
import torch
import typer
from datasets import load_dataset
from omegaconf import DictConfig
from prediction_utils.logger import wandb_logger
from prediction_utils.prompt import promptify
from sentence_transformers import SentenceTransformer, util
from syntok.tokenizer import Tokenizer
from tqdm import tqdm

app = typer.Typer()
openai.api_key = os.getenv("OPENAI_API_KEY")
tok = Tokenizer()
data_path = pathlib.Path(__file__).parents[1] / "data"
datasets.logging.set_verbosity_error()


def find_nearest_examples(text, layer, test_layer, top_k) -> torch.Tensor:
    """Find the nearest examples in the dataset.

    Args:
        text: The text to find the nearest examples for.
        layer: The dataset layer (language + e3c annotation layers) to use for the generation of
            the dataset.
        top_k: The number of the nearest examples to return.

    Returns:
        The nearest examples.
    """
    query = layer.filter(lambda x: x["text"] == text)[0]["vector"]
    corpus = test_layer.filter(lambda x: x["text"] != text)
    cos_scores = util.cos_sim(query, corpus["vector"])[0]
    top_results = torch.topk(cos_scores, k=top_k)
    indices = top_results.indices.tolist()
    sub_corpus = corpus[indices]["text"]
    return sub_corpus


def predict_layer(
    cfg: DictConfig = {},
    layer: str = "fr.layer2",
    temperature: float = 0,
    debug: bool = False,
    limit: int = -1,
    top_k: int = 3,
) -> None:
    """Generate an annotated dataset of clinical entities from a given model and temperature.

    Args:
        layer: The dataset layer (language + e3c annotation layers) to use for the generation of
            the dataset.
        model: The openai model to use for the generation of the dataset.
        temperature: The temperature to use for the generation of the dataset. If the temperature is
            close 0, the model is more deterministic while if the temperature is close to 1,
            the model is more sensitive to the randomness.
        debug: If True, the dataset generation will be done on a small subset of the dataset.
        limit: The number of examples to use during the dataset generation.
        prompt_tuning_path: The prompt file path.
        wandb_run_name: The wandb run name.
    """
    test_layer = f"{layer.split('.')[0]}.layer1"
    e3c_dataset = load_dataset("bio-datasets/e3c")
    predictions: dict = {"text": [], "prediction": [], "ground truth": []}
    sentence_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    text = []
    offsets_entities = []
    labels = []
    if limit == -1:
        limit = len(e3c_dataset[layer])
        cfg.limit = limit
    e3c_dataset[test_layer] = e3c_dataset[test_layer].map(
        lambda example: {
            "vector": sentence_model.encode(
                example["text"],
            )
        },
        batched=False,
    )
    e3c_dataset[layer] = e3c_dataset[layer].map(
        lambda example: {
            "vector": sentence_model.encode(
                example["text"],
            )
        },
        batched=False,
    )

    model = hydra.utils.instantiate(cfg.model)

    for example in tqdm(e3c_dataset[layer], desc=f"Processing {layer}", leave=True, position=0):
        few_shots_examples = create_prompt(
            e3c_dataset,
            example,
            layer,
            test_layer,
            top_k,
            cfg.static_examples,
        )
        text_example = example["text"]
        text.append(text_example)
        text_response = model(
            few_shots_examples=few_shots_examples,
            text=text_example,
        )
        labels_example, entities_offsets_example = convert_text_response(
            text_example,
            text_response,
        )
        offsets_entities.append(entities_offsets_example)
        labels.append(labels_example)
        predictions["text"].append(text_example)
        predictions["prediction"].append(text_response)
        predictions["ground truth"].append(labels_example)
        if limit == len(labels):
            break
        if len(labels) == 3 and debug:
            break

    prediction_path = write_dataset(
        labels=labels,
        model=model.model_name,
        offsets_entities=offsets_entities,
        text=text,
        limit=limit,
        top_k=top_k,
        layer=layer,
    )

    wandb_logger(
        cfg=cfg,
        layer=layer,
        prediction_path=prediction_path,
        wandb_run_name=f"layer_{layer}_topk_{top_k}_limit_{limit}",
    )


def create_prompt(e3c_dataset, example, layer, test_layer, top_k, static_examples):
    few_shots_examples = []
    few_shots_examples = find_nearest_examples(
        example["text"],
        e3c_dataset[layer],
        e3c_dataset[test_layer],
        top_k=top_k,
    )
    text_static_examples = e3c_dataset[test_layer].filter(
        lambda x: x["text"] in e3c_dataset[test_layer][static_examples]["text"]
    )["text"]
    few_shots_examples.extend(text_static_examples)
    few_shots_examples = e3c_dataset[test_layer].filter(lambda x: x["text"] in few_shots_examples)
    few_shots_examples = datasets.Dataset.from_pandas(
        few_shots_examples.to_pandas().drop_duplicates(subset="text")
    )
    few_shots_examples
    few_shots_examples = promptify(
        dataset=few_shots_examples,
        tags_dict=e3c_dataset[layer].features["clinical_entity_tags"].feature.int2str,
    )
    return few_shots_examples


def write_dataset(
    labels: list,
    model: str,
    offsets_entities: list[list],
    text: List[str],
    limit: int,
    top_k: int,
    layer: str,
) -> str:
    """Write the dataset to a csv file. And name the file with the model and temperature used.

    Args:
        labels: The labels of the dataset.
        model: The model used to generate the dataset.
        temperature: The temperature used to generate the dataset.
        offsets_entities: The offsets of the entities in the dataset.
        text: The text of the dataset.
        limit: The number of examples to use during the dataset generation.
        top_k: The number of examples to use during the dataset generation using k-nearest
        neighbors method with sentence embeddings.
        layer: The dataset layer (language + e3c annotation layers) to use for the generation of
            the dataset.

    Returns:
        The path of the csv file.

    """

    folder_name = os.path.join(data_path, f"layer={layer}_limit={limit}".replace(".txt", ""))
    csv_path = f"{folder_name}/model_{model}_{datetime.now().strftime('%Y%m%d-%I%M%S')}.csv"
    df = pd.DataFrame(data={"text": text, "labels": labels, "entities_offsets": offsets_entities})
    pathlib.Path(folder_name).mkdir(exist_ok=True, parents=True)
    df.to_csv(csv_path, index=False)
    return csv_path


def convert_text_response(
    text_example: str,
    text_response: str,
) -> Tuple[List[str], List[List]]:
    """Convert the text response of the LLM model into list of labels and offsets.

    We browse the bullet points of the response, and we check if the text of the bullet point is
    present in the text of the example. If it is, we add the label and the offset of the entity.
    We convert the phrase entity in IOB format. To simplify the parsing during the training.

    Args:
        text_example: The text of the example.
        text_response: The text response of the LLM model.

    Returns:
        The list of offsets of the entities in the example.
    """
    tokens = list(tok.tokenize(text_example))
    labels_example = ["O"] * len(tokens)
    entities_offsets_example = [[token.offset, token.offset + len(token.value)] for token in tokens]
    for line in text_response.split("\n"):
        if line.startswith("- "):
            if len(line[2:].split(" ")) <= 10:
                entity = line[3:-1].split('"')[0]
                start = text_example.replace("-", " ").lower().find(entity.lower())
                end = start + len(entity.lower())
                if start != -1 and end != -1:
                    annotated_tokens = [
                        idx_token
                        for idx_token, token in enumerate(entities_offsets_example)
                        if token[0] >= start and token[1] <= end
                    ]
                    for idx_token in annotated_tokens:
                        if idx_token == annotated_tokens[0]:
                            labels_example[idx_token] = "B-CLINENTITY"
                        else:
                            labels_example[idx_token] = "I-CLINENTITY"
    return labels_example, entities_offsets_example


@hydra.main(version_base="1.3", config_path="configs", config_name="prediction.yaml")
def main(cfg: DictConfig):
    """Create few-shot examples for a given layer.

    This method is used to create few-shot examples in a prompt format. The aim is to use this
    examples to guide a large language model in an extract task.

    Args:
        cfg: Hydra configuration.
    """
    predict_layer(
        cfg=cfg,
        layer=cfg.layer,
        limit=cfg.limit,
        debug=cfg.debug,
        top_k=cfg.top_k,
    )


if __name__ == "__main__":
    main()
