"""
This module implements a standard experiment runner for all four-choice multiple choice question (MCQ) exams like
the MBE and MPRE.
"""

# imports
import datetime
import json
import logging
from pathlib import Path
from typing import Callable

# packages
import pandas
import tqdm

# project
from engines.base_engine import BaseEngine
from engines.engine_openai import OpenAIEngine
from exams.choice_prompts import MBE_PROMPT_LIST
from exams.experiment_data import get_next_session_path


def load_choice_question_data(exam_type: str, exam_id: str) -> pandas.DataFrame:
    """Use the basic CSV format for the question data so everyone can easily review.

    Columns:
        question_category
        question_number
        correct_answer
        question_prompt
        choice_a
        choice_b
        choice_c
        choice_d

    Args:
        exam_type (str): The exam type, e.g., mbe.
        exam_id (str): The exam id, e.g., ncbe-pdf-001

    Returns:
        pandas.DataFrame: The question data in rows
    """
    # build path
    question_path = (
        Path(__file__).parent.parent.parent  # src/exams  # src  # project root
        / "exams"
        / exam_type
        / exam_id
        / "questions.csv"
    )

    if not question_path.exists():
        raise FileNotFoundError(f"Question file not found: {question_path}")

    # load the data
    question_data = pandas.read_csv(
        question_path, index_col=0, encoding="utf-8", dtype=str
    )

    # return
    return question_data

def parse_choice_response(response: str) -> dict:
    """parse the model response like:
    First Choice: (C)
    Second Choice: (D)
    Third Choice: (A)
    Explanation: The answer is C because ...

    to return

    {
        "answer": "C",
        "backup_answer": "D",
        "explanation": "The answer is C because ..."
    }
    """
    response_data = {
        "answer": None,
        "second_answer": None,
        "third_answer": None,
        "explanation": None,
    }
    response_lines = response.strip().splitlines()

    for i, line in enumerate(response_lines):
        line = line.strip()

        if line.startswith("First Choice"):
            response_data["answer"] = (
                line.split()
                .pop()
                .replace("(", "")
                .replace(")", "")
                .replace(".", "")
                .strip()
            )
        elif line.startswith("Second Choice"):
            response_data["second_answer"] = (
                line.split()
                .pop()
                .replace("(", "")
                .replace(")", "")
                .replace(".", "")
                .strip()
            )
        elif line.startswith("Third Choice"):
            response_data["third_answer"] = (
                line.split()
                .pop()
                .replace("(", "")
                .replace(")", "")
                .replace(".", "")
                .strip()
            )
        elif line.startswith("Explanation"):
            reason_text = " ".join(response_lines[i:]).strip()
            reason_text = reason_text.split(":", 1).pop().strip()
            response_data["explanation"] = reason_text
            break

    # if we still don't have a first answer, then try to parse the first tokens of the string
    if response_data["answer"] is None and len(response_lines) > 0:
        # first line tokens
        first_line_tokens = response_lines[0].split()

        # check if the token is a letter after stripping (, ), and .
        if len(first_line_tokens) > 0:
            first_token = first_line_tokens[0]\
                .replace("(", "")\
                .replace(")", "")\
                .replace(".", "")\
                .strip()
            if first_token in ["A", "B", "C", "D"]:
                response_data["answer"] = first_token

    return response_data



def run_choice_exam(
    exam_type: str,
    exam_id: str,
    experiment_id: int,
    prompt_method: Callable,
    engine: BaseEngine,
    num_samples: int = 1,
) -> list[Path]:
    """Run an exam session with the given parameters.

    N.B.: The engine is passed in as a parameter and no parameters are modified/swept within this method.
    Parameter setting/sweeping is responsibility of the caller.

    Args:
        exam_type (str): The exam type, e.g., mbe.
        exam_id (str): The exam id, e.g., ncbe-pdf-001
        experiment_id (int): The experiment id, e.g., 1.
        prompt_method (Callable): The prompt method to use.
        engine (BaseEngine): The engine to use.
        num_samples (int, optional): The number of samples to generate per question. Defaults to 1.

    Returns:
        list[Path]: A list of paths to the session files.
    """

    # load the questions from the exam
    question_data = load_choice_question_data(exam_type, exam_id)

    # session path list
    session_path_list = []

    # get N samples
    for _ in range(num_samples):
        # get the next session path
        session_path = get_next_session_path(exam_type, exam_id, experiment_id)

        # create the session file path
        session_file_path = session_path / "session.json"

        # setup session data
        session_data = {
            "exam_type": exam_type,
            "exam_id": exam_id,
            "experiment_id": experiment_id,
            "prompt_method": prompt_method.__name__,
            "engine": engine.get_name(),
            "parameters": engine.get_current_parameters(),
            "session_start": datetime.datetime.now().isoformat(),
            "session_end": None,
            "question_data": [],
        }

        # iterate over the questions
        question_prog_bar = tqdm.tqdm(
            question_data.iterrows(), total=len(question_data)
        )
        for row_id, question_row in question_prog_bar:
            # update the progress bar
            question_prog_bar.set_description(
                f"{exam_type}:{exam_id}:{experiment_id}:Q {row_id+1}/{len(question_data)}"
            )

            # initialize question record data
            question_row_data = {
                "question_category": question_row["question_category"],
                "question_number": question_row["question_number"],
                "question_prompt": question_row["question_prompt"],
                "question_choice_a": question_row["choice_a"],
                "question_choice_b": question_row["choice_b"],
                "question_choice_c": question_row["choice_c"],
                "question_choice_d": question_row["choice_d"],
                "correct_answer": question_row["correct_answer"],
                "model_prompt": None,
                "model_response": None,
            }

            try:
                # get the question prompt
                question_row_data["model_prompt"] = prompt_method(question_row)

                # get the model response
                question_row_data["model_response"] = engine.get_completions(
                    question_row_data["model_prompt"]
                )
            except Exception as e:
                logging.error(f"Error in question row_id={row_id}: {e}")
            finally:
                # add the question data
                session_data["question_data"].append(question_row_data)

                # save the session data each question
                with open(session_file_path, "w", encoding="utf-8") as f:
                    json.dump(session_data, f, indent=4)

        # update the session end time
        session_data["session_end"] = datetime.datetime.now().isoformat()

        # save the session data
        with open(session_file_path, "w", encoding="utf-8") as f:
            json.dump(session_data, f, indent=4)
