import json
from pathlib import Path

import typer
from dotenv import load_dotenv
from langchain.cache import SQLiteCache
from langchain.globals import set_llm_cache

from bellek.lang.qdecomp import make_question_decomposer

load_dotenv()

set_llm_cache(SQLiteCache(database_path="/tmp/langchain-cache.db"))


def main(dataset_file: Path = typer.Option(...), out: Path = typer.Option(...)):
    qdecomposer = make_question_decomposer()
    with open(dataset_file) as src:
        with open(out, "w") as dst:
            for line in src:
                example = json.loads(line)
                sub_questions = qdecomposer(question=example["question"])
                output = {
                    "id": example["id"],
                    "question": example["question"],
                    "question_decomposition": [{"question": q} for q in sub_questions],
                }
                dst.write(json.dumps(output, ensure_ascii=False))
                dst.write("\n")


if __name__ == "__main__":
    typer.run(main)
