""" Creates a complete data dictionary from two JSON files. """

import json
import os
import sys
from typing import Any, Dict, Set, Tuple
import logging
import openai
import local_util as util


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def get_column_names(json_obj: Dict[str, Any]) -> Set[str]:
    """ Returns a set of column names from a JSON object. """
    if "Columns" in json_obj:
        return {col["Column Name"] for col in json_obj["Columns"]}
    return set(json_obj.keys())


def compare_columns(columns1: Set[str], columns2: Set[str]) -> Tuple[Set[str], Set[str]]:
    """ Returns a tuple of sets containing the columns that are missing in each set. """
    return columns2 - columns1, columns1 - columns2


def clean_json_string(json_str: str) -> Dict[str, Any]:
    """ Cleans up the JSON string returned by the OpenAI API. """
    try:
        cleaned_str = json_str.strip('"').replace("```json\n", "").replace("\n```", "")
        cleaned_str = cleaned_str.replace("\\n", "\n").replace('\\"', '"')
        return json.loads(cleaned_str)
    except json.JSONDecodeError as e:
        logging.error("Error parsing JSON string: %s", e)
        sys.exit(1)


def create_thread(api_client: openai.Client, content: str, assistant_id: str) -> str:
    """ Creates a thread in the OpenAI API and returns the thread ID. """
    try:
        thread = api_client.beta.threads.create(messages=[{"role": "user", "content": content}])
        api_client.beta.threads.runs.create(thread_id=thread.id, assistant_id=assistant_id)
        return thread.id
    except Exception as e:
        logging.error("Error creating OpenAI thread: %s", e)
        sys.exit(1)


def describe(api_client: openai.Client, json_obj1: Dict[str, Any],
             json_obj2: Dict[str, Any]) -> Dict[str, Any]:
    """ Returns a dictionary containing the data dictionary for the two JSON objects. """
    assistant_id = os.environ.get("COLUMN_SUBSCRIBER")
    json_content = {**json_obj1, **json_obj2}
    json_content_str = json.dumps(json_content)
    thread_id = create_thread(api_client, json_content_str, assistant_id)
    return clean_json_string(util.get_response(api_client, thread_id))


def get_api_client() -> openai.Client:
    """ Returns an OpenAI API client. """
    api_key = os.environ.get("OPENAI_API_KEY")
    return openai.OpenAI(api_key=api_key)


def write_to_file(file_path: str, content: str) -> None:
    """ Writes the content to the file. """
    try:
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(content)
    except IOError as e:
        logging.error("Error writing to file {file_path}: %s", e)
        sys.exit(1)


def main(api_client: openai.Client, file1: str, file2: str) -> None:
    """ Main function. """
    try:
        json_obj1 = util.load_json(file1)
        json_obj2 = util.load_json(file2)
    except Exception as e:
        logging.error("Error loading JSON files: %s", e)
        sys.exit(1)

    columns1 = get_column_names(json_obj1)
    columns2 = get_column_names(json_obj2)

    missing_in_1, missing_in_2 = compare_columns(columns1, columns2)
    if missing_in_1 or missing_in_2:
        error_message = ""
        if missing_in_1:
            error_message += f"Columns missing in file1: {missing_in_1}\n"
        if missing_in_2:
            error_message += f"Columns missing in file2: {missing_in_2}\n"
        sys.exit(error_message)

    result = describe(api_client, json_obj1, json_obj2)

    if "Table Name" in result:
        table_name = result["Table Name"]
        output_data = json.dumps(result, indent=4)
        filename = f"{table_name}_data_dictionary.json"
        write_to_file(filename, output_data)
        logging.info(f"Data written to {filename}")
    else:
        logging.error("Table Name not found in JSON data.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Usage: python complete_data_dictionary.py <file1.json> <file2.json>")

    client = get_api_client()
    main(client, sys.argv[1], sys.argv[2])
