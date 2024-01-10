import datetime
import glob
import os
import re
from pathlib import Path
from typing import Any, List, Optional, Tuple

from langchain.chat_models import AzureChatOpenAI


def get_azure_chatbot(
    openai_api_key: Optional[str] = None,
    deployment_name: Optional[str] = None,
    openai_api_type: Optional[str] = None,
    openai_api_base: Optional[str] = None,
    openai_api_version: Optional[str] = None,
    **kwargs: Any,
) -> AzureChatOpenAI:
    """
    Create an instance of AzureChatOpenAI.
    Usage:
        azure_chat_openai = get_azure_chatbot()
    Args:
        openai_api_key: The OpenAI API key. If not provided, the method will try to get it from the
            environment variable OPENAI_API_KEY.
        deployment_name: The name of the deployment. If not provided, the method will try to get it
            from the environment variable DEPLOYMENT_NAME.
        openai_api_type: The type of the OpenAI API. If not provided, the method will try to get it
            from the environment variable OPENAI_API_TYPE.
        openai_api_base: The base of the OpenAI API. If not provided, the method will try to get it
        from the environment variable OPENAI_API_BASE.
        openai_api_version: The version of the OpenAI API. If not provided, the method will try to
        get it from the environment variable OPENAI_API_VERSION.
        kwargs: Other optional parameters.
    Returns:
        An instance of AzureChatOpenAI.
    """

    openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")

    if not openai_api_key:
        raise ValueError(
            "openai_api_key is required. Please provide it as an argument or set the environment"
            " variable OPENAI_API_KEY."
        )

    deployment_name = deployment_name or os.getenv("DEPLOYMENT_NAME")
    if not deployment_name:
        raise ValueError(
            "deployment_name is required. Please provide it as an argument or set the environment"
            " variable DEPLOYMENT_NAME."
        )

    openai_api_type = openai_api_type or os.getenv("OPENAI_API_TYPE") or "azure"

    openai_api_base = openai_api_base or os.getenv("OPENAI_API_BASE")
    if not openai_api_base:
        raise ValueError(
            "openai_api_base is required. Please provide it as an argument or set the environment"
            " variable OPENAI_API_BASE."
        )

    openai_api_version = openai_api_version or os.getenv("OPENAI_API_VERSION")
    if not openai_api_version:
        raise ValueError(
            "openai_api_version is required. Please provide it as an argument or set the environment"
            " variable OPENAI_API_VERSION."
        )

    return AzureChatOpenAI(
        deployment_name=deployment_name,
        openai_api_type=openai_api_type,
        openai_api_base=openai_api_base,
        openai_api_version=openai_api_version,
        openai_api_key=openai_api_key,
        **kwargs,
    )


def read_files_from_directory(directory_path: str) -> List[Tuple[str, str]]:
    """
    This function reads all the .txt files from the specified directory path,
    and returns a list of tuples, where each tuple contains the filename and its corresponding content.

    Args:
        directory_path (str): The path to the directory containing the .txt files.

    Returns:
        file_content_list (list): A list of tuples, where each tuple has the filename as the first element and
                                  the corresponding content as the second element.
    """

    file_content_list = []
    txt_files = glob.glob(os.path.join(directory_path, "*.txt"))

    for txt_file in txt_files:
        with open(txt_file, "r", encoding="utf-8") as file:
            data = file.read()
            # Add a tuple (filename, content) to the list
            file_content_list.append((os.path.basename(txt_file), data))

    return file_content_list


def extract_folder_and_name_from_path(source_var: str, default_path: str) -> (str, str):
    """
    Extract and return the folder path and file name from a given source variable (can be an environment variable or regular variable).

    Args:
    - source_var (str): The name of the environment variable to fetch or directly a string path.
    - default_path (str): The default path to use if the environment variable is not set or the source_var doesn't contain a valid path.

    Returns:
    - folder_path (str): The folder path extracted from the provided source as a string.
    - file_name (str): The file name extracted from the provided source.
    """
    # If source_var is an environment variable name, fetch its value; otherwise use it directly as a path.
    file_path = os.environ.get(source_var, None)
    if not file_path:
        file_path = default_path
    folder_path = str(Path(file_path).parent)
    file_name = Path(file_path).name
    return folder_path, file_name


def compute_cost(prompt: int, completion: int) -> float:
    """
    Computes the cost for Azure OpenAI based on input tokens and output tokens.

    Args:
    - prompt (int): Number of input tokens.
    - completion (int): Number of output tokens.

    Returns:
    - float: Computed cost.
    """
    prompt_cost_per_1000 = 0.003
    completion_cost_per_1000 = 0.004

    prompt_cost = (prompt / 1000) * prompt_cost_per_1000
    completion_cost = (completion / 1000) * completion_cost_per_1000

    return prompt_cost + completion_cost


def find_estimate_used_cost_filename() -> str:
    """
    Finds the filename that matches the pattern "estimate_used_cost_$..." in the current directory.
    If not found, creates a new file with value $0.

    Returns:
    - str: Filename that matches the pattern.
    """
    for filename in os.listdir("."):
        if "estimate_used_cost_$" in filename:
            return filename

    # If not found, create a new file with $0.0000
    new_filename = "./estimate_used_cost_$0.0000"
    with open(new_filename, "w") as file:
        file.write("Initial creation with $0.0000\nOnly calculates costs executed locally, may not be accurate!\n")

    return new_filename


def update_estimate_used_cost_file(prompt: int, completion: int) -> None:
    """
    Updates the estimate used cost file by adding the new computed cost.

    Args:
    - prompt (int): Number of input tokens.
    - completion (int): Number of output tokens.

    Returns:
    - None
    """
    # Compute the cost
    cost = compute_cost(prompt, completion)

    # Find the file with the required pattern
    filename = find_estimate_used_cost_filename()

    # Extract current value from filename
    match = re.search(r"estimate_used_cost_\$(\d+\.\d{4})", filename)
    if match:
        current_value = float(match.group(1))
        new_value = current_value + cost
    else:
        raise ValueError("Could not extract current value from filename.")

    # Rename the file with the new cost
    os.rename(filename, f"./estimate_used_cost_${new_value:.4f}")

    # Write the transaction details into the file
    with open(f"./estimate_used_cost_${new_value:.4f}", "a") as file:
        file.write(
            f"{datetime.datetime.now()} prompt token: {prompt} completion token: {completion} cost: ${cost:.4f}\n"
        )
