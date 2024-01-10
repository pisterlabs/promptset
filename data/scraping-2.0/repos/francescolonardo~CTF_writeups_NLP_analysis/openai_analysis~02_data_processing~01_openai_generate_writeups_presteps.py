import json
import os
import re
import ast
import openai


def get_response(prompt: str):
    response = openai.Completion.create(
        model="text-davinci-003",  # Replace with the latest GPT-3.5 model name
        prompt=prompt,
        max_tokens=1000,
    )
    return response.choices[0].text.strip()  # Extract text from GPT-3.5 response


def extract_list_from_response(response_text):
    # Use regex to find a list in the response
    match = re.search(r"\[\s*\"[^\[\]]*\"\s*(,\s*\"[^\[\]]*\"\s*)*\]", response_text)
    if match:
        list_str = match.group(0)
        try:
            return ast.literal_eval(list_str)
        except (ValueError, SyntaxError):
            print("Error in converting string to list.")
            return None
    else:
        print("No list found in the response.")
        return None


def find_original_files_without_presteps():
    original_filepaths = []
    for dirpath, _, filenames in os.walk("../dataset_writeups"):
        for filename in filenames:
            if filename.endswith("_original.md"):
                original_filepath = os.path.join(dirpath, filename)
                presteps_filepath = original_filepath.replace(
                    "_original.md", "_presteps.json"
                )
                if not os.path.exists(presteps_filepath):
                    original_filepaths.append(original_filepath)
    return original_filepaths


if __name__ == "__main__":
    with open("./api_keys.json", "r", encoding="utf-8") as f:
        api_key = json.load(f)["api_key"]
    openai.api_key = api_key
    with open(
        "../prompts/prompt_from_original_to_presteps.txt", "r", encoding="utf-8"
    ) as f:
        base_prompt = f.read()

    try:
        original_filepaths = find_original_files_without_presteps()
        total_files = len(original_filepaths)

        for idx, original_filepath in enumerate(original_filepaths, start=1):
            with open(original_filepath, "r", encoding="utf-8") as writeup_file:
                writeup = writeup_file.read()
            prompt = base_prompt + "\n```" + writeup + "```"

            print(f"[{idx}/{total_files}] Processing `{original_filepath}`")
            try:
                response_text = get_response(prompt=prompt)
                # print(f"Response: {response_text}")

                # First try to parse as JSON
                try:
                    response_json_data = json.loads(response_text)
                except json.JSONDecodeError:
                    response_json_data = None

                # If not JSON, try extracting list using regex and ast.literal_eval
                if response_json_data is None:
                    response_json_data = extract_list_from_response(response_text)

                if response_json_data is not None:
                    print("INFO - Data appears to be well-formed and complete.")
                    presteps_filepath = original_filepath.replace(
                        "_original.md", "_presteps.json"
                    )
                    with open(
                        presteps_filepath, "w", encoding="utf-8"
                    ) as presteps_file:
                        json.dump(response_json_data, presteps_file, indent=4)
                    print(f"Saved `{presteps_filepath}`")
                else:
                    print("ERROR - Could not extract valid data from the response")

            except openai.error.InvalidRequestError as e:
                print(f"ERROR - OpenAI - {e}")

    except KeyboardInterrupt:
        print("Stop.")
