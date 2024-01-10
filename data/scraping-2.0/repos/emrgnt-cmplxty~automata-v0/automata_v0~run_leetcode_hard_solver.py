"""Study the dataset."""
import argparse
import logging
import os
from typing import Tuple

import openai
from agent.completion_provider import CompletionProvider, RunMode
from evalplus.data import write_jsonl
from leetcode_hard_gym.leetcode_env.environment import LeetCodeEnv
from leetcode_hard_gym.leetcode_env.leetcode_types import (
    LeetCodeSubmission,
    ProgrammingLanguage,
)
from leetcode_solver.leetcode_constants import LEETCODE_PROBLEMS_PATH
from leetcode_solver.leetcode_problem_solver import LeetCodeSolver
from leetcode_solver.leetcode_problems_loader import LeetCodeLoader
from utils import extract_code, parse_arguments

from automata_v0.utils import (
    get_configured_logger,
    get_root_fpath,
    load_existing_jsonl,
    prep_for_leetcode,
)

LEETCODE_PROBLEMS_PATH = os.path.join(
    get_root_fpath(),
    "leetcode_hard_gym",
    "leetcode_dataset",
    "data",
    "with_snippets",
    "leetcode_hard_with_snippets_uncontaminated_tests.csv",
)
LEETCODE_SOLUTIONS_FILE_NAME = "leetcode_hard_py_40__model_eq_{MODEL}__temp_eq_{TEMPERATURE}__run_mode_eq_{RUN_MODE}.jsonl"
LEETCODE_SOLUTIONS_OUTPUT_DIR = os.path.join(
    get_root_fpath(), "data", "results", "leetcode_results", "{MODEL}"
)


def load_existing_task_ids(existing_data: list[dict]) -> set[str]:
    """Load existing task ids from the data."""
    return {entry["task_id"] for entry in existing_data}


def configure_paths(args: argparse.Namespace) -> None:
    """Configure paths for the run."""
    args.problems_data_path = args.problems_data_path or LEETCODE_PROBLEMS_PATH
    args.solutions_output_data_dir = (
        args.solutions_output_data_dir
        or LEETCODE_SOLUTIONS_OUTPUT_DIR.format(
            MODEL=args.model.replace("-", "_").replace(".", "p")
        )
    )
    args.solutions_output_file_name = (
        args.solutions_output_file_name
        or LEETCODE_SOLUTIONS_FILE_NAME.format(
            MODEL=args.model.replace("-", "_").replace(".", "p"),
            TEMPERATURE=str(args.temperature).replace(".", "p"),
            RUN_MODE=args.run_mode,
        )
    )
    args.output_path = os.path.join(
        args.solutions_output_data_dir, args.solutions_output_file_name
    )


def load_data(args: argparse.Namespace) -> Tuple[list[dict], set[str]]:
    """Load existing data."""
    existing_data = load_existing_jsonl(args.output_path)

    existing_task_ids = (
        set() if args.overwrite else load_existing_task_ids(existing_data)
    )

    completion_seqs = existing_data or []

    return existing_task_ids, completion_seqs


def main(logger: logging.Logger):
    # Parse arguments
    args = parse_arguments()
    configure_paths(args)

    openai.api_key = os.getenv("OPENAI_API_KEY_LOCAL", "")

    print(f"Loading problem data from {args.problems_data_path}")
    loader = LeetCodeLoader(args.problems_data_path)

    num_examples = len(loader.data)
    print(f"Number of examples to run = {num_examples}")
    solver = LeetCodeSolver(num_examples)
    env = LeetCodeEnv()

    completion_provider = CompletionProvider(
        run_mode=RunMode(args.run_mode),
        model=args.model,
        temperature=args.temperature,
    )

    print(f"Loading from {args.output_path}")
    if not os.path.exists(args.output_path):
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    existing_task_ids, completion_seqs = load_data(args)

    for index in solver.indices:
        task_id = f"LeetCode-Hard/{index}"

        if task_id in existing_task_ids and not args.overwrite:
            print(
                f"Skipping task_id {task_id} as it already exists in the output file."
            )
            continue

        problem_context = loader.get_problem_context(index)
        print(
            f"Running w/ problem at index {index} and context:\n\n{problem_context}"
        )

        try:
            raw_completion = completion_provider.get_completion(
                task_input=loader.get_problem_context(index),
                code_snippet=loader.get_snippet(index),
            )
            try:
                clean_completion = extract_code(raw_completion)
                status, reward, done, submission_result = env.step(
                    LeetCodeSubmission(
                        code=clean_completion,
                        lang=ProgrammingLanguage.PYTHON3,
                        question_id=loader.get_backend_problem_id(index),
                        question_slug=loader.get_problem_slug(index),
                        timeout=12,
                    )
                )
                print(
                    f"status={status}, reward={reward}, done={done}, submission_result={submission_result}"
                )
                solver.log_result(index, reward)
            except Exception as e:
                print(f"Failed with exception {e}")
                clean_completion = ""
                status, reward, done, submission_result = "", "", "", ""

            completion_seqs.append(
                {
                    "task_id": f"LeetCode-Hard/{index}",
                    "completion": clean_completion,
                    "raw_completion": raw_completion,
                    "status": status,
                    "reward": reward,
                    "done": done,
                    "submission_result": submission_result,
                    "problem_slug": loader.get_problem_slug(index),
                    "problem_id": loader.get_backend_problem_id(index),
                    "frontend_problem_id": loader.get_frontend_problem_id(
                        index
                    ),
                }
            )
            print("completion_seqs = ", completion_seqs)
            print(f"Writing output to {args.output_path}")
            write_jsonl(args.output_path, completion_seqs)

        except Exception as e:
            print(f"Failed with exception {e}")
            write_jsonl(args.output_path, completion_seqs)
            solver.log_result(index, False)


if __name__ == "__main__":
    logger = get_configured_logger(__name__, "INFO")
    main(logger)
