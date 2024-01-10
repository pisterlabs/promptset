import json
import os
import re
import openai


def get_response(prompt: str):
    response = openai.Completion.create(
        model="text-davinci-003",  # Replace with the latest GPT-3.5 model name
        prompt=prompt,
        max_tokens=1000,
    )
    return response.choices[0].text.strip()  # Extract text from GPT-3.5 response


def find_steps_files_without_substeps():
    steps_filepaths = []
    for dirpath, dirnames, filenames in os.walk("../dataset_writeups"):
        for filename in filenames:
            if filename.endswith("_steps.json"):
                steps_filepath = os.path.join(dirpath, filename)
                substeps_filepath = steps_filepath.replace(
                    "_steps.json", "_substeps.json"
                )
                if not os.path.exists(substeps_filepath):
                    steps_filepaths.append(steps_filepath)
    return steps_filepaths


def extract_json(s: str):
    # Search for string patterns that look like JSON objects
    for match in re.finditer(r"{[^}]*}", s):
        substr = s[match.start() :]
        # Try to load the substring as json, extending the substring until it succeeds or it's clear it's not json.
        for end in range(len(substr), 0, -1):
            try:
                potential_json = json.loads(substr[:end])
                # If it succeeds, return the valid json object.
                return potential_json
            except json.JSONDecodeError:
                continue
    return None  # Return None if no valid JSON object is found.


if __name__ == "__main__":
    with open("./api_keys.json", "r", encoding="utf-8") as f:
        api_key = json.load(f)["api_key"]
    openai.api_key = api_key
    with open(
        "../prompts/prompt_from_steps_to_substeps.txt", "r", encoding="utf-8"
    ) as f:
        base_prompt = f.read()

    try:
        steps_filepaths = find_steps_files_without_substeps()
        total_files = len(steps_filepaths)

        for idx, step_filepath in enumerate(steps_filepaths, start=1):
            with open(step_filepath, "r", encoding="utf-8") as writeup_file:
                writeup = writeup_file.read()
            prompt = base_prompt + "\n```" + writeup + "```"

            print(f"[{idx}/{total_files}] Processing `{step_filepath}`")
            try:
                response_text = get_response(prompt=prompt)
                # print(f"Response: {response_text}")
            except openai.error.InvalidRequestError as e:
                print(f"ERROR - OpenAI - {e}")

            # Attempt to parse the JSON data
            response_json_data = extract_json(response_text)

            # Check if JSON is well-formed and complete
            if response_json_data is not None:
                try:
                    # Check if the structure contains the expected keys and values
                    steps_list = response_json_data.get("SubstepsModel", {}).get(
                        "Steps", []
                    )

                    if not steps_list:
                        raise ValueError("ERROR - JSON is missing 'Steps' list.")

                    for step in steps_list:
                        if "Substeps" not in step or not isinstance(
                            step["Substeps"], list
                        ):
                            raise ValueError(
                                f"ERROR - Step {step.get('StepNumber')} does not contain a 'Substeps' list or 'Substeps' is not a list."
                            )

                    print("INFO - JSON appears to be well-formed and complete.")

                    steps_filepath = step_filepath.replace(
                        "_steps.json", "_substeps.json"
                    )
                    with open(steps_filepath, "w", encoding="utf-8") as steps_file:
                        json.dump(response_json_data, steps_file, indent=4)

                    print(f"Saved `{steps_filepath}`")

                except ValueError as ve:
                    print(ve)

            else:
                print("ERROR - JSON may be truncated or invalid.")

    except KeyboardInterrupt:
        print("Stop.")
