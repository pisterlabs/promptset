import os
import random
import re
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
import typer
from datasets import load_dataset
from InstructorEmbedding import INSTRUCTOR
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from rich import print
from rich.progress import track
from sentence_transformers import util


# The maximum number of tokens in the input and output
MAX_INPUT_TOKENS = 6192
MAX_OUTPUT_TOKENS = 2000

# The valid stategies for selecting in-context examples
SIMILAR = "similar"
RANDOM = "random"
STRATEGIES = [SIMILAR, RANDOM]

# These are all related to the submission
TASK_A = "A"
TASK_B = "B"
TASK_C = "C"
TASKS = [TASK_A, TASK_B, TASK_C]

RUN_1 = "1"
RUN_2 = "2"
RUN_3 = "3"
RUNS = [RUN_1, RUN_2, RUN_3]

# These are all related to the output files
ID_COL = "ID"
ENCOUNTER_ID_COL = "encounter_id"
TEST_ID = "TestID"
SYSTEM_OUTPUT_1 = "SystemOutput1"
SYSTEM_OUTPUT_2 = "SystemOutput2"
SYSTEM_OUTPUT = "SystemOutput"
TEAM_NAME = "wanglab"


def _fetch_in_context_examples(
    train, test, k: int = 3, retrieve_similar: bool = True, filter_by_dataset: bool = False
) -> List[int]:
    """Fetches the indices of k in-context examples from the train set to use for each example in the test set.
    If strategy is "similar", then examples are chosen based on similarity to the test example. Otherwise, examples
    are chosen randomly. If filter_by_dataset is True, only examples from the same dataset as the test example will be used.
    """
    if retrieve_similar:
        # Embed the train and test dialogues
        embedder = INSTRUCTOR("hkunlp/instructor-large")
        embedding_instructions = "Represent the Medicine dialogue for clustering:"
        test_dialogues = embedder.encode(
            [
                [embedding_instructions, f"dataset: {dataset} dialogue: {dialogue}"]
                for dataset, dialogue in zip(test["dataset"], test["dialogue"])
            ],
            show_progress_bar=True,
        )
        train_dialogues = embedder.encode(
            [
                [embedding_instructions, f"dataset: {dataset} dialogue: {dialogue}"]
                for dataset, dialogue in zip(train["dataset"], train["dialogue"])
            ],
            show_progress_bar=True,
        )
    # Get top-k most similar examples in the train set for each example in the test set
    top_k_indices = []
    for i, test_dataset in enumerate(test["dataset"]):
        # If filter_by_dataset, restrict to example of the same dataset type
        if filter_by_dataset:
            train_indices = [j for j, train_dataset in enumerate(train["dataset"]) if train_dataset == test_dataset]
        else:
            train_indices = list(range(len(train["dataset"])))
        # If strategy is "similar", then we select examples based on similarity to the test example
        if retrieve_similar:
            scores = util.cos_sim(np.expand_dims(test_dialogues[i], 0), train_dialogues[train_indices])
            top_k_indices_ds = torch.topk(scores, k=min(k, len(scores))).indices.flatten().tolist()
            # Map these back to the original indices
            top_k_indices.append([train_indices[idx] for idx in top_k_indices_ds])
        else:
            top_k_indices.append(random.sample(range(len(train_indices)), k=min(k, len(train_indices))))
    return top_k_indices


def _format_in_context_example(train: Dict[str, Any], idx: int, include_dialogue: bool = False) -> str:
    example = ""
    if include_dialogue:
        example += f'\nEXAMPLE DIALOGUE:\n{train["dialogue"][idx].strip()}'
    example += f'\nEXAMPLE NOTE:\n{train["note"][idx].strip()}'
    return example


def main(
    test_fp: str = typer.Argument("Filepath (or URL) to the test set (should be a CSV file)."),
    output_dir: str = typer.Argument("Path to the directory where predictions will be written."),
    model_name: str = typer.Option("gpt-4", help="Name of the model to use. Must be supported by LangChain."),
    temperature: float = typer.Option(0.2, help="Temperature for the LLM."),
    train_fp: str = typer.Option(
        None,
        help=(
            "Filepath (or URL) to the train set (should be a CSV file). In-context examples will be sourced from"
            " here. Required if k > 0."
        ),
    ),
    k: int = typer.Option(
        3, help="Maximum number of in-context examples to use. If 0, no in-context examples will be used."
    ),
    retrieve_similar: bool = typer.Option(
        True,
        help="If True, in-context examples will be selected by retrieving examples from the train set with similar dialogues to the input dialogue. Has no effect if k=0.",
    ),
    include_dialogue: bool = typer.Option(
        False, help="If True, will include the dialogue in the in-context examples."
    ),
    filter_by_dataset: bool = typer.Option(
        True, help="If True, will only use in-context examples from the same dataset. Has no effect if k=0."
    ),
    task: str = typer.Option(TASK_B, help=f"Task name. Should be one of {TASKS}"),
    run: str = typer.Option(RUN_1, help=f"Which challenge run to produce predictions for. Should be one of {RUNS}"),
    debug: bool = typer.Option(False, help="If True, will only run for a single example of the test set."),
) -> None:
    """Generates predictions using LangChain for the given task and run on the given test set.

    Example usage:

    # Without in-context examples
    OPENAI_API_KEY="..." \
        python scripts/run_langchain.py "MEDIQA-Chat-TestSets-March-15-2023/TaskB/taskB_testset4participants_inputConversations.csv" \
        "./outputs/wanglab/taskB/run1" \
        --k 0 \
        --task "B" \
        --run "1"

    # With in-context examples
    OPENAI_API_KEY="..." \
        python scripts/run_langchain.py "MEDIQA-Chat-TestSets-March-15-2023/TaskB/taskB_testset4participants_inputConversations.csv" \
        "./outputs/wanglab/taskB/run1" \
        --train-fp "MEDIQA-Chat-Training-ValidationSets-Feb-10-2023/TaskB/TaskB-TrainingSet.csv" \
        --task "B" \
        --run "1"
    """
    # Error handling
    if k < 0:
        raise ValueError(f"k should be non-negative. Got: {k}")
    if k > 0 and not train_fp:
        raise ValueError("If k > 0, train_fp should be provided.")
    if task not in TASKS:
        raise ValueError(f"Task should be one of {TASKS}. Got: '{task}'")
    if run not in RUNS:
        raise ValueError(f"Run should be one of {RUNS}. Got: '{run}'")

    # Load the dataset
    if k > 0:
        train = load_dataset(
            "csv",
            data_files={
                "train": train_fp,
            },
        )["train"]
    test = load_dataset(
        "csv",
        data_files={
            "test": test_fp,
        },
    )["test"]

    if debug:
        print("[yellow]--debug passed. Will only run for a single example of the test set.[/yellow]")
        test = test[:1]
        if k > 0:
            train = train[:k]

    # Setup the LLM
    llm = ChatOpenAI(
        model_name=model_name,
        temperature=temperature,
        max_tokens=MAX_OUTPUT_TOKENS,
    )

    # Setup the chain
    if task == TASK_B:
        prompt = PromptTemplate(
            input_variables=["examples", "dialogue"],
            template="""Write a clinical note reflecting this doctor-patient dialogue. Use the example notes below to decide the structure of the clinical note. Do not make up information.
{examples}

DIALOGUE: {dialogue}
CLINICAL NOTE:
        """,
        )
    else:
        raise NotImplementedError(f"Task {task} is not implemented yet.")
    chain = LLMChain(llm=llm, prompt=prompt)

    # Optionally, retrieve the top-k most similar dialogues as the in-context examples
    if k > 0:
        print(
            f"Retrieving {k} in-context examples with the following strategy:\n"
            f"- retrieve similar: {retrieve_similar}\n"
            f"- include dialogue {include_dialogue}\n"
            f"- dataset filtering {filter_by_dataset}"
        )
        top_k_indices = _fetch_in_context_examples(
            train, test, k=k, retrieve_similar=retrieve_similar, filter_by_dataset=filter_by_dataset
        )

    example_prompt = prompt.format(
        examples=_format_in_context_example(train, idx=top_k_indices[0][0], include_dialogue=include_dialogue)
        if k > 0
        else "",
        dialogue=test["dialogue"][0],
    )
    print(f"[blue]Example prompt: {example_prompt}[/blue]")

    # Run the chain to generate predictions
    predictions = []
    for i, dialogue in track(
        enumerate(test["dialogue"]),
        description="Generating predictions with LangChain",
        total=len(test["dialogue"]),
    ):
        examples = ""

        if k > 0:
            # Collect as many in-context examples as we can fit within the max input tokens
            prompt_length = llm.get_num_tokens(prompt.format(dialogue=dialogue, examples=""))
            for top_k_idx in top_k_indices[i]:
                example = _format_in_context_example(train, idx=top_k_idx, include_dialogue=include_dialogue)
                if (prompt_length + llm.get_num_tokens(example)) < MAX_INPUT_TOKENS:
                    examples += example
                    prompt_length += llm.get_num_tokens(example)

        prediction = chain.run(dialogue=dialogue, examples=examples)

        # GPT sometimes makes the mistake of placing Imaging by itself, which won't be picked up by the official
        # evaluation script. So we prepend "RESULTS" to it here, which will be picked up.
        prediction = re.sub(
            r"(\n|^)(?!.*\\nRESULTS.*)[Ii][Mm][Aa][Gg][Ii][Nn][Gg](:)?", r"\1RESULTS\nImaging\2", prediction
        )

        predictions.append(prediction)

    if task == TASK_B:
        ct_output = {TEST_ID: test[ENCOUNTER_ID_COL], SYSTEM_OUTPUT: predictions}
    else:
        ct_output = {
            TEST_ID: test[ID_COL],
            # TODO: Need to provide section headers and section text
            SYSTEM_OUTPUT_1: ...,
            SYSTEM_OUTPUT_2: ...,
        }

    # Save outputs to disk
    output_dir = Path(output_dir)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Write the predictions to a CSV file that conforms to the challenge format
    ct_fn = f"task{task}_{TEAM_NAME}_run{run}.csv"
    ct_fp = os.path.join(output_dir, ct_fn)
    pd.DataFrame.from_dict(ct_output).to_csv(ct_fp, index=False)

    print(f"[green]Predictions saved to {output_dir}[/green]")


if __name__ == "__main__":
    typer.run(main)
