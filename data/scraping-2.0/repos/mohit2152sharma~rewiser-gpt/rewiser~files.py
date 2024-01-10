import logging
import os
from datetime import datetime
from typing import List

from rewiser.gpt.agent import OpenAIAgent
from rewiser.gpt.utils import split_numbered_lines
from rewiser.utils import env_var, file_commit_date, read_env_var


@env_var(var="DOC_DIRECTORY")
def list_files(
    doc_directory: str | None = None, return_style: str = "filepath"
) -> List[str]:
    """List files in the given directory

    Args:
        doc_directory: directory path where all the documents are kept. The directory
        path is relative to project's or repository's root folder
        return_style: either `'filepath'` or `'filename'`, if it is equal to
        `'filepath'`, returns the filepath by appending `doc_directory`. If it is
        `'filename'` returns just the filenames

    Returns:
        List of file names
    """
    if return_style == "filepath":
        return [
            os.path.join(doc_directory, f)
            for f in os.listdir(doc_directory)
            if not os.path.isdir(f)
        ]  # type: ignore
    elif return_style == "filename":
        return [f for f in os.listdir(doc_directory) if not os.path.isdir(f)]
    else:
        raise ValueError(
            f"The return style: {return_style} provided is invalid. Valid values are `filepath` and `filename`"  # noqa
        )


@env_var(var="DOC_DIRECTORY")
def sort_files(doc_directory: str | None = None) -> List[str]:
    # sort the files using git
    # fetch the last committed date for each file and then sort by dates
    files = list_files(doc_directory=doc_directory, return_style="filepath")
    print(f"all files: {files}")
    rs = sorted(
        files,
        key=lambda x: datetime.strptime(file_commit_date(x), "%Y-%m-%d"),
        reverse=True,
    )
    logging.info(f"sorted files list: {rs}")
    return rs


def read_file(filepath: str) -> str:
    with open(filepath, "r") as file:
        content = file.read()

    return content


def extract_filename(filepath: str) -> str:
    return os.path.splitext(os.path.split(filepath)[-1])[0]


def concat_files(filepaths: List[str]) -> str:
    result = ""
    splits = []
    for file in filepaths:
        content = read_file(file)
        heading = extract_filename(file).capitalize()
        result += f"# {heading}\n{content}\n\n"
        splits.extend(split_numbered_lines(text=content))

    # for each split generate a question
    openai_api_key = read_env_var("OPENAI_API_KEY", raise_error=False)
    if openai_api_key:
        agent = OpenAIAgent(template_name="question_generator")
        questions = ""
        logging.info(
            f"Generating questions. Total questions to generate: {len(splits)}"
        )
        counter = 1
        if splits:
            for split in splits:
                question = agent.run(input_text=split)
                if question:
                    questions += f"{counter}. {question}\n"
                    counter += 1

            logging.info("total questions generated")
            result += f"# Questions\n\n{questions}"
    else:
        logging.info("openai_api_key is not provided skipping generating questions")

    return result
