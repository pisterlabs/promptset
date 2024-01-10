import json
import os
import ast
import openai


def get_response(messages: list):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages,
        stream=False,
    )
    return response.choices[0].message.content


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


def find_substeps_groups_files_without_taxonomized():
    substeps_groups_filepaths = []
    search_dir = "../data/groups"
    for dirpath, _, filenames in os.walk(search_dir):
        for filename in filenames:
            if (
                filename.startswith("groups_")
                and filename.endswith(".json")
                and not filename.endswith(
                    "_taxonomized.json"
                )  # Change the file extension here
            ):
                substeps_groups_filepath = os.path.join(dirpath, filename)
                taxonomized_substeps_filepath = substeps_groups_filepath.replace(
                    ".json", "_taxonomized.json"  # Change the file extension here
                )
                if not os.path.exists(taxonomized_substeps_filepath):
                    substeps_groups_filepaths.append(substeps_groups_filepath)
    return substeps_groups_filepaths


if __name__ == "__main__":
    with open("../api_keys/api_key.json", "r", encoding="utf-8") as f:
        api_key = json.load(f)["api_key"]
    openai.api_key = api_key
    with open(
        "../prompts/prompts_taxonomy_development/02_prompt_taxonomize_substeps_groups.txt",
        "r",
        encoding="utf-8",
    ) as f:
        prompt = f.read()

    try:
        substeps_groups_filepaths = find_substeps_groups_files_without_taxonomized()
        total_files = len(substeps_groups_filepaths)

        for idx, substeps_groups_filepath in enumerate(
            substeps_groups_filepaths, start=1
        ):
            with open(substeps_groups_filepath, "r", encoding="utf-8") as f:
                substeps_groups_chunk = f.read()
            messages = [
                {
                    "role": "user",
                    "content": prompt + substeps_groups_chunk,
                }
            ]

            print(f"[{idx}/{total_files}] Processing `{substeps_groups_filepath}`")
            try:
                response = get_response(messages=messages)
            except openai.error.InvalidRequestError as e:
                print(f"ERROR - OpenAI - {e}")

            response_json_data = extract_json(response)

            if response_json_data is not None:
                print("INFO - JSON appears to be well-formed and complete.")

                taxonomized_groups_filepath = substeps_groups_filepath.replace(
                    ".json", "_taxonomized.json"  # Change the file extension here
                )
                with open(taxonomized_groups_filepath, "w", encoding="utf-8") as f:
                    f.write(response)

                print(f"Saved `{taxonomized_groups_filepath}`")
            else:
                print("ERROR - JSON may be truncated or invalid.")

    except KeyboardInterrupt:
        print("Stop.")
