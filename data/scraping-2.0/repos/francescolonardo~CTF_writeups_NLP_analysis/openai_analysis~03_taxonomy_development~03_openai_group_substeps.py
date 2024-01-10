import json
import os
import ast
import openai


def get_response(messages: list):
    response = openai.ChatCompletion.create(  # https://platform.openai.com/docs/api-reference/chat/create
        model="gpt-3.5-turbo",  # The name of the OpenAI chatbot model to use
        messages=messages,
        # temperature=1.0,
        stream=False,
    )
    return response.choices[0].message.content


def find_list_substeps_files_without_grouped():
    list_substeps_filepaths = []
    # Define the directory to search in
    search_dir = "../data/list_substeps"
    for dirpath, _, filenames in os.walk(search_dir):
        for filename in filenames:
            # Check if the filename starts with "list_substeps_"
            # and ends with ".json" but not with "_grouped.json"
            if (
                filename.startswith("list_substeps_")
                and filename.endswith("_abstracted.json")
                and not filename.endswith("_grouped.json")
            ):
                # Get the complete path of the current file
                list_substeps_filepath = os.path.join(dirpath, filename)
                # Formulate the path of the corresponding _grouped.json file
                grouped_substeps_filepath = list_substeps_filepath.replace(
                    "_abstracted.json", "_grouped.json"
                )
                # Check if the _grouped.json file exists
                if not os.path.exists(grouped_substeps_filepath):
                    list_substeps_filepaths.append(list_substeps_filepath)
    return list_substeps_filepaths


def extract_json(response):
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(response)
        except (ValueError, SyntaxError):
            print(
                "Error: The response string is neither valid JSON nor a Python dict-like string"
            )
            return None


if __name__ == "__main__":
    with open("../api_keys/api_key.json", "r", encoding="utf-8") as f:
        api_key = json.load(f)["api_key"]
    openai.api_key = api_key
    with open("../prompts/02_prompt_group_substeps.txt", "r", encoding="utf-8") as f:
        prompt = f.read()

    try:
        list_substeps_filepaths = find_list_substeps_files_without_grouped()
        total_files = len(list_substeps_filepaths)

        for idx, list_substeps_filepath in enumerate(list_substeps_filepaths, start=1):
            with open(list_substeps_filepath, "r", encoding="utf-8") as f:
                list_substeps_chunk = f.read()
            messages = [
                {
                    "role": "user",
                    "content": prompt + list_substeps_chunk,
                }
            ]

            print(f"[{idx}/{total_files}] Processing `{list_substeps_filepath}`")
            try:
                response = get_response(messages=messages)
            except openai.error.InvalidRequestError as e:
                print(f"ERROR - OpenAI - {e}")

            # Attempt to parse the JSON data
            response_json_data = extract_json(response)

            # Check if JSON is well-formed and complete
            if response_json_data is not None:
                print("INFO - JSON appears to be well-formed and complete.")

                grouped_substeps_filepath = list_substeps_filepath.replace(
                    "_abstracted.json", "_grouped.json"
                )
                with open(grouped_substeps_filepath, "w", encoding="utf-8") as f:
                    json.dump(response_json_data, f, indent=4)

                print(f"Saved `{grouped_substeps_filepath}`")
            else:
                print("ERROR - JSON may be truncated or invalid.")

    except KeyboardInterrupt:
        print("Stop.")
