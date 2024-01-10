import logging
import os
import openai
import pandas as pd
from evalplus.data import write_jsonl

from llm_battle_ground.completion_provider import CompletionProvider, RunMode
from llm_battle_ground.constants import DATA_DIRECTORY
from llm_battle_ground.helpers import LeetCodeProcessor
from llm_battle_ground.scripts import common_arg_parser
from llm_battle_ground.types import DataDirectories, Datasets, LLMProviders
from llm_battle_ground.utils import (
    get_configured_logger,
    get_root_fpath,
    read_jsonl,
)

# Pathing
IN_FILE_NAME = Datasets.LEETCODE_FULL.value
OUT_DIR = os.path.join(
    get_root_fpath(),
    DataDirectories.RESULTS.value,
    "leetcode",
    "provider",
    "{PROVIDER}",
    "{MODEL}",
    "generation",
)
OUT_FILE_NAME = "generation_{IN_FILE_NAME}__{MODEL}__run_mode_eq_{RUN_MODE}__temperature_eq_{TEMPERATURE}__n_pass_{N_PASS}.jsonl"


# Local configurations
PROVIDER = "openai"
RUN_MODE = "vanilla-zero-shot"
MODEL = "gpt-4-0613"
TEMPERATURE = 0.7
N_PASS = 1

MAX_VAL = int(1e10)


# TODO - Build a helper class to assist this script in experiment running.
def main(
    logger: logging.Logger,
    in_path: str,
    out_path: str,
    provider: LLMProviders,
    run_mode: RunMode,
    model: str,
    temperature: float,
    max_samples: int,
    n_pass: int,  # TODO - Implement this.
) -> list[dict]:
    logger.info(f"Loading dataset file from {in_path}.")
    dataset = pd.read_csv(in_path).sort_values(by=["frontend_question_id"])[
        ::-1
    ]

    completion_provider = CompletionProvider(
        run_mode=run_mode,
        model=model,
        temperature=temperature,
        provider=provider,
    )
    processor = LeetCodeProcessor()

    logger.info(f"Saving results to {out_path}.")
    if os.path.exists(out_path):
        logger.info(f"Loading existing results from {out_path}.")
        outputs = read_jsonl(out_path)
    else:
        logger.info("Beginning new results.")
        outputs = []

    observed_ids = {x["frontend_question_id"] for x in outputs}
    min_observed_id = (
        outputs[-1]["frontend_question_id"] if outputs else MAX_VAL
    )

    for loc in range(len(dataset)):
        entry = dataset.iloc[loc]
        difficulty = entry["difficulty"]

        # Skipping past previously processed problems
        frontend_question_id = int(dataset.iloc[loc]["frontend_question_id"])
        if frontend_question_id in observed_ids:
            logger.info(
                f"Continuing because {frontend_question_id} problem has been processed previously."
            )
            continue

        # Since we use rng, we need to skip past old max when re-loading data
        if frontend_question_id > min_observed_id:
            logger.info(
                f"Continuing because {frontend_question_id} has been processed previously."
            )
            continue
        else:
            if min_observed_id != MAX_VAL:
                logger.info(
                    f"Processing first example, resetting min_observed_id to {MAX_VAL}."
                )
                min_observed_id = MAX_VAL

        # Generating input for completion
        task_input = f"LeetCode Problem #{frontend_question_id}\nTitle: {entry['question_title']}\nDescription:\n{processor.clean_html_content(entry['raw_content'])}\n\n"

        code_snippet = entry["python3_snippet"]
        raw_response = completion_provider.get_completion(
            task_input=task_input, code_snippet=code_snippet + "\n"
        )
        result = {
            "task_input": task_input,
            "code_snippet": code_snippet,
            "raw_response": raw_response,
            "n_pass": n_pass,
            "difficulty": difficulty,
            "frontend_question_id": frontend_question_id,
            "loc": loc,
        }
        outputs.append(result)
        if len(outputs) >= max_samples:
            break
        write_jsonl(out_path, outputs)
    return outputs


if __name__ == "__main__":
    parser = common_arg_parser()

    args = parser.parse_args()
    logger = get_configured_logger(__name__, args.log_level)

    openai.api_key = os.getenv("OPENAI_API_KEY_LOCAL", "")

    in_path = os.path.join(
        args.in_dir or DATA_DIRECTORY, args.in_file_name or IN_FILE_NAME
    )

    args.provider = args.provider or LLMProviders(PROVIDER)

    out_file_name = args.out_file_name or OUT_FILE_NAME.format(
        IN_FILE_NAME=(args.in_file_name or IN_FILE_NAME)
        .replace(".csv", "")
        .replace(".jsonl", ""),
        PROVIDER=args.provider.value,
        MODEL=(args.model or MODEL).replace("-", "_").replace(".", "p"),
        # note if temperature = 0, regular approach will evaluate to false and use default
        TEMPERATURE=str(args.temperature).replace(".", "p")
        if args.temperature is not None
        else TEMPERATURE,
        N_PASS=args.n_pass or N_PASS,
        RUN_MODE=args.run_mode or RUN_MODE,
    )
    out_dir = args.out_dir or OUT_DIR.format(
        PROVIDER=args.provider.value or PROVIDER,
        MODEL=(args.model or MODEL).replace("-", "_").replace(".", "p"),
    )

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    out_path = os.path.join(out_dir, out_file_name.replace("-", "_"))

    outputs = main(
        logger,
        in_path,
        out_path,
        args.provider,
        RunMode(args.run_mode or RUN_MODE),
        args.model or MODEL,
        args.temperature or TEMPERATURE,
        args.max_samples or MAX_VAL,
        args.n_pass or N_PASS,
    )
    write_jsonl(out_path, outputs)
