import random
import itertools
import json
from typer import Typer
from typing import List
from textbook.dataset_gen.dataset_gen import (
    load_leaves,
    mass_generation,
    OpenAIGenerator,
    MonkeyGenerator,
    write_results_to_jsonl,
)
import openai
import os
from pathlib import Path

from textbook.dataset_gen.create_prompts import Topic, Query
from textbook.dataset_gen.filtering import load_and_filter_exos
from datasets import Dataset

app = Typer()


def create_prompt_query(topic_1: Topic, topic_2: Topic, profession: str) -> str:
    query = f'''
            Create a code completion exercise on the intersection of “{topic_1.topic}” and “{topic_2.topic}”.  
            Write it for a {profession}. 

            The exercise must be of the style: 

            ```
            def name(args):

            """Docstring explaining the exercise"""

            python code to solve the exercise
            ```

            NO CLASSES

            MAKE IT VERY DIFFICULT
            '''
    query = "\n".join([m.lstrip() for m in query.strip().split("\n")])
    return query


def create_prompts(
    topic: Topic,
    combination_options: List[Topic],
    professions: List[str],
) -> List[Query]:
    random.shuffle(combination_options)

    prompts: List[Query] = []

    for loc_topic in combination_options:
        if (
            loc_topic.mixing
            and loc_topic.parent != topic.parent
            and loc_topic.topic != topic.topic
        ):
            for profession in professions:
                query = create_prompt_query(topic, loc_topic, profession)
                prompts.append(Query(query=query, topic_1=topic, topic_2=loc_topic))

    return prompts


@app.command()
def generate(
    tree_path: str,
    leaves_path: str,
    output_path: str,
    retries: int = 10,
    pool_size: int = 10,
    debug: bool = False,
    debug_speed: int = 2,
    n_prompts: int = 100,
):
    with open(tree_path, "r") as openfile:
        # Reading from json file
        professions = list(json.load(openfile))

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if not debug:
        openai.api_key = os.environ["OPENAI_API_KEY"]

        def get_generator():
            return OpenAIGenerator()

    else:

        def get_generator():
            return MonkeyGenerator(speed=debug_speed)

    leaves = load_leaves(leaves_path)
    prompts: List[List[Query]] = [
        create_prompts(
            i,
            combination_options=leaves,
            professions=professions,
        )
        for i in leaves
    ]

    prompts_flat = list(itertools.chain(*prompts))
    if n_prompts > len(prompts_flat):
        raise ValueError(
            f"Cannot generate({n_prompts}) prompts because it is larger than the number of"
            f" available prompts ({len(prompts_flat)})"
        )
    prompts_selection = [i.query for i in prompts_flat]

    print(f"prompts: {len(prompts_selection)}")

    solo_prompts = list(set(prompts_selection))

    print(f"solo prompts: {len(solo_prompts)}")
    prompts_selection = solo_prompts[:n_prompts]
    print(f"total prompts: {len(prompts_selection)}")

    mass_generation(
        prompts_selection,
        get_generator,
        save_dir=output_path,
        pool_size=pool_size,
        retries=retries,
    )


@app.command()
def filter(exo_path: Path, dataset_file: str):
    print(exo_path)
    exos = load_and_filter_exos(exo_path)
    print(len(exos))
    write_results_to_jsonl(dataset_file, exos)


@app.command()
def push(repo_name: str, dataset_file: Path):
    with open(dataset_file, "r") as file:
        lines = file.readlines()
        exercises = [json.loads(line) for line in lines]

    def gen():
        for exo in exercises:
            yield exo

    dataset = Dataset.from_generator(gen)
    dataset.push_to_hub(repo_name)


if __name__ == "__main__":
    app()
