"""Check if all the files have an annotation."""

import os
import logging

import fire
import dotenv
import openai
from constants import (
    BEST_TEMPLATES,
    ENTITIES,
    ROOT,
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

dotenv.load_dotenv(ROOT / ".env")
openai.api_key = os.getenv("OPENAI_API_KEY")

RESULTS_PATH = ROOT / "results"


def main():
    """Main script."""

    models = [
        "chatgpt",
        "gpt3",
        "gpt4",
        "llama2-7b",
        "llama2-7b-chat",
        "llama2-13b",
        "llama2-13b-chat",
        "llama2-70b",
        "llama2-70b-chat",
    ]

    templates = [
        "ext",
        "cls",
        "ext_def",
        "ext_exp",
        "cls_def",
        "ext_def_exp",
        "cls_exp",
        "cls_def_exp"
    ]

    languages = ["english", "portuguese"]
    for mid in models:
        for language in languages:

            entities = ENTITIES[language]
            results_path = RESULTS_PATH / "test"
            for entity in entities:
                tid = BEST_TEMPLATES[language][(mid, entity)]

                path = results_path / language / mid / entity / tid
                n_docs = len(list(path.glob("*.txt")))
                if language == "portuguese":
                    assert n_docs == 96, f"Only {n_docs} for {path}"
                elif language == "english":
                    assert n_docs == 161, f"Only {n_docs} for {path}"

            results_path = RESULTS_PATH / "prompt_selection"
            for entity in entities:
                for tid in templates:
                    path = results_path / language / mid / entity / tid
                    n_docs = len(list(path.glob("*.txt")))
                    assert n_docs == 20, f"Only {n_docs} for {path}"


if __name__ == "__main__":
    fire.Fire(main)
