from typing import Optional, Sequence
from collections import defaultdict

from pydantic import BaseModel, Field
from langchain.prompts import ChatPromptTemplate

from athena import emit_meta
from athena.programming import Exercise, Submission

from module_programming_llm.config import BasicApproachConfig
from module_programming_llm.helpers.llm_utils import (
    get_chat_prompt_with_formatting_instructions, 
    num_tokens_from_string, 
    num_tokens_from_prompt, 
    predict_and_parse
)
from module_programming_llm.helpers.utils import get_diff


class FileProblemStatement(BaseModel):
    file_name: str = Field(description="File name")
    problem_statement: str = Field(description="Problem statement relevant for this file")


class SplitProblemStatement(BaseModel):
    """Collection of problem statements split by file"""
    items: Sequence[FileProblemStatement] = Field(description="File problem statements")


# pylint: disable=too-many-locals
async def split_problem_statement_by_file(
        exercise: Exercise, 
        submission: Submission, 
        prompt: ChatPromptTemplate,
        config: BasicApproachConfig, 
        debug: bool
    ) -> Optional[SplitProblemStatement]:
    """Split the general problem statement by file

    Args:
        exercise (Exercise): Exercise to split the problem statement for (respecting the changed files)
        submission (Submission): Submission to split the problem statement for (respecting the changed files)
        prompt (ChatPromptTemplate): Prompt template to check for problem_statement
        config (BasicApproachConfig): Configuration

    Returns:
        Optional[SplitProblemStatement]: Split problem statement, None if it is too short or too long
    """
    
    # Return None if the problem statement is too short
    if num_tokens_from_string(exercise.problem_statement or "") <= config.split_problem_statement_by_file_prompt.tokens_before_split:
        return None
    
    # Return None if the problem statement not in the prompt
    if "problem_statement" not in prompt.input_variables:
        return None

    model = config.model.get_model()  # type: ignore[attr-defined]

    template_repo = exercise.get_template_repository()
    solution_repo = exercise.get_solution_repository()
    submission_repo = submission.get_repository()

    changed_files_from_template_to_solution = get_diff(
        src_repo=template_repo, 
        dst_repo=solution_repo, 
        file_path=None, 
        name_only=True
    ).split("\n")

    changed_files_from_template_to_submission = get_diff(
        src_repo=template_repo, 
        dst_repo=submission_repo, 
        file_path=None, 
        name_only=True
    ).split("\n")

    chat_prompt = get_chat_prompt_with_formatting_instructions(
        model=model, 
        system_message=config.split_problem_statement_by_file_prompt.system_message,
        human_message=config.split_problem_statement_by_file_prompt.human_message,
        pydantic_object=SplitProblemStatement
    )
    
    prompt_input = {
        "problem_statement": exercise.problem_statement or "No problem statement.",
        "changed_files_from_template_to_solution": ", ".join(changed_files_from_template_to_solution),
        "changed_files_from_template_to_submission": ", ".join(changed_files_from_template_to_submission)
    }

    # Return None if the prompt is too long
    if num_tokens_from_prompt(chat_prompt, prompt_input) > config.max_input_tokens:
        return None

    split_problem_statement = await predict_and_parse(
        model=model, 
        chat_prompt=chat_prompt, 
        prompt_input=prompt_input,
        pydantic_object=SplitProblemStatement,
        tags=[
            f"exercise-{exercise.id}",
            f"submission-{submission.id}",
            "split-problem-statement-by-file"
        ]
    )

    if debug:
        emit_meta("file_problem_statements", {
            "prompt": chat_prompt.format(**prompt_input),
            "result": split_problem_statement.dict() if split_problem_statement is not None else None
        })

    if split_problem_statement is None or not split_problem_statement.items:
        return None

    # Join duplicate file names (some responses contain multiple problem statements for the same file)
    file_problem_statements_by_file_name = defaultdict(list)
    for file_problem_statement in split_problem_statement.items:
        file_problem_statements_by_file_name[file_problem_statement.file_name].append(file_problem_statement)

    split_problem_statement.items = [
        FileProblemStatement(
            file_name=file_name,
            problem_statement="\n".join(
                file_problem_statement.problem_statement
                for file_problem_statement in file_problem_statements
            )
        )
        for file_name, file_problem_statements in file_problem_statements_by_file_name.items()
    ]

    return split_problem_statement
