""" 
embeddings_generator.py

Generate embeddings by OpenAI API.
"""

from typing import List, Any, Dict
import numpy as np

import modal

from .config import ProjectConfig, Config

stub = modal.Stub(ProjectConfig._stub_embedding)
SHARED_ROOT = "/root/.cache"


@stub.function(image=modal.Image.debian_slim(), retries=0, cpu=1)
def dict_list_to_list_dict(papers: List[Any], keys: List[str]) -> Dict[str, List[str]]:
    """
    Convert a list of dictionaries to a dictionary of lists.

    Args:
        papers (List[Any]): List of dictionaries.
        keys (List[str]): List of keys.

    Returns:
        Dict[str, List[str]]: Dictionary with lists as values.
    """
    print(type(papers[0]))
    converted = {k: [paper[k].replace("\n", " ") for paper in papers] for k in keys}
    return converted


@stub.function(
    image=modal.Image.debian_slim().pip_install("openai", "numpy"),
    shared_volumes={
        SHARED_ROOT: modal.SharedVolume.from_name(ProjectConfig._shared_vol)
    },
    secret=modal.Secret.from_name("my-openai-secret"),
)
def generate_embeddings(
    text_list: List[str], model: str, batch_size: int = 20
) -> np.ndarray:
    """
    Generate embeddings for a list of texts using the specified model.

    Args:
        text_list (List[str]): List of texts.
        model (str): Name of the model.
        batch_size (int): Batch size for generating embeddings. Default is 20.

    Returns:
        np.ndarray: Array of embeddings.
    """
    import time
    import numpy as np
    import openai

    embeddings = []
    for i in range(0, len(text_list), batch_size):
        print(
            f"Generating embedings from id {i} to {min(i + batch_size, len(text_list))}..."
        )
        response = openai.Embedding.create(
            input=text_list[i : min(i + batch_size, len(text_list))],
            model=model,
        )["data"]
        batch_embeddings = [np.array(item["embedding"]) for item in response]
        embeddings += batch_embeddings
        time.sleep(0.1)
    assert len(embeddings) == len(text_list)
    return np.array(embeddings)


@stub.function(
    image=modal.Image.debian_slim().pip_install("openai", "numpy"),
    shared_volumes={
        SHARED_ROOT: modal.SharedVolume.from_name(ProjectConfig._shared_vol)
    },
    secret=modal.Secret.from_name("my-openai-secret"),
    timeout=36000,
)
def save_embeddings(config: Config) -> None:
    """
    Save embeddings for the given configuration.

    Args:
        config (Config): Configuration object.

    Returns:
        None
    """
    from pathlib import Path
    import numpy as np

    papers = []
    papers = modal.Function.lookup(ProjectConfig._stub_db, "get_all_papers").call(
        db_config=config.db, max_item_count=config.project.max_papers, force=True
    )
    papers = sorted(papers, key=lambda x: int(x["id"]))

    converted = dict_list_to_list_dict(papers, config.embedding.keys)
    for key, list_data in converted.items():
        save_path = Path(SHARED_ROOT) / config.files.embeddings_files[key]
        Path(save_path.parent).mkdir(parents=True, exist_ok=True)  # recursive
        embeddings = generate_embeddings(
            list_data,
            model=config.embedding.model,
            batch_size=config.embedding.batch_size,
        )
        np.save(save_path, embeddings)


@stub.local_entrypoint()
def main(config_file: str = "configs/defaults.toml"):
    """
    Main entry point of the script.

    Args:
        config_file (str): Path to the configuration file. Default is 'configs/defaults.toml'.
    """
    import dacite
    import toml

    config = dacite.from_dict(data_class=Config, data=toml.load(config_file))
    save_embeddings.call(config)
