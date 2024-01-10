import json
import os
import ast
import openai


def get_response(messages: list):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages,
        # temperature=1.0,
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


def find_taxonomized_groups_files_without_merged():
    taxonomized_groups_filepaths = []
    search_dir = "../data/groups"
    processed_ranges = []

    for dirpath, _, filenames in os.walk(search_dir):
        for filename in filenames:
            if filename.endswith("_merged.json"):
                groups_part = filename.split("_")[1].replace("_merged.json", "")
                start_group, end_group = map(int, groups_part.split("-"))
                processed_ranges.append(range(start_group, end_group + 1))

    for dirpath, _, filenames in os.walk(search_dir):
        for filename in filenames:
            if filename.startswith("groups_") and filename.endswith(
                "_taxonomized.json"
            ):
                groups_part = filename.split("_")[1].replace("_taxonomized.json", "")
                start_group, end_group = map(int, groups_part.split("-"))
                in_processed_range = any(
                    start_group in processed_range and end_group in processed_range
                    for processed_range in processed_ranges
                )
                if not in_processed_range:
                    taxonomized_groups_filepath = os.path.join(dirpath, filename)
                    taxonomized_groups_filepaths.append(taxonomized_groups_filepath)

    return taxonomized_groups_filepaths


if __name__ == "__main__":
    with open("../api_keys/api_key.json", "r", encoding="utf-8") as f:
        api_key = json.load(f)["api_key"]
    openai.api_key = api_key

    with open("../prompts/04_prompt_merge_taxonomies.txt", "r", encoding="utf-8") as f:
        prompt = f.read()

    try:
        taxonomized_groups_filepaths = find_taxonomized_groups_files_without_merged()
        total_files = len(taxonomized_groups_filepaths)

        idx = 0
        while idx < total_files:
            group_size = 2
            if total_files - idx == 3:
                group_size = 3

            taxonomies = []
            filepaths_to_merge = taxonomized_groups_filepaths[idx : idx + group_size]
            for taxonomized_groups_filepath in filepaths_to_merge:
                with open(taxonomized_groups_filepath, "r", encoding="utf-8") as f:
                    taxonomy = f.read()
                    taxonomies.append(taxonomy)

            messages = [
                {
                    "role": "user",
                    "content": prompt + "".join(taxonomies),
                }
            ]

            print(f"[{idx + 1}/{total_files}] Processing `{filepaths_to_merge}`")
            try:
                response = get_response(messages=messages)
            except openai.error.InvalidRequestError as e:
                print(f"ERROR - OpenAI - {e}")

            response_json_data = extract_json(response)

            if response_json_data is not None:
                print("INFO - JSON appears to be well-formed and complete.")

                first_file = filepaths_to_merge[0]
                last_file = filepaths_to_merge[-1]

                first_file_number = first_file.split("_")[1].split("-")[0]
                last_file_number = last_file.split("_")[1].split("-")[-1]

                merged_taxonomized_filepath = f"../data/groups/groups_{first_file_number}-{last_file_number}_merged.json"

                with open(merged_taxonomized_filepath, "w", encoding="utf-8") as f:
                    f.write(response)

                print(f"Saved `{merged_taxonomized_filepath}`")
            else:
                print("ERROR - JSON may be truncated or invalid.")

            idx += group_size

    except KeyboardInterrupt:
        print("Stop.")
