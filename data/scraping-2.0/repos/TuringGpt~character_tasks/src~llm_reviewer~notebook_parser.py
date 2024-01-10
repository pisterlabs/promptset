import json

from fuzzywuzzy import fuzz
from openai import OpenAI

from src.llm_reviewer.constants import Roles, PATH_TO_SECRETS

with open(PATH_TO_SECRETS, "r") as f:
    openai_api_key = json.load(f)["openai_api_key"]


def get_closest_match(query, choices):
    """
    Get the closest match(es) to a query string from a list of choices.

    :param query: The query string.
    :param choices: A list of strings to match against.
    :param limit: The maximum number of matches to return.
    """
    best_role = None
    best_score = 0
    for choice in choices:
        score = fuzz.ratio(query, choice)
        if score > best_score and score > 25:
            best_score = score
            best_role = choice

    return best_role, best_score


def count_empty_from_end(cells):
    count = 0
    for message in reversed(cells):
        if message["source"].strip() == "":
            count += 1
        else:
            break
    return count


def extract_messages(notebook):
    """
    Parse a notebook and extract the message objects.

    :param notebook: The notebook object.
    """
    messages = []
    cut_tail = count_empty_from_end(notebook.cells)
    cells = notebook.cells[2:]
    if cut_tail:
        cells = cells[:-cut_tail]
    for cell in cells:
        if cell["cell_type"] == "markdown":
            headers = ["**User**", "**Assistant**"]
        elif cell["cell_type"] == "code":
            headers = ["# User", "# Assistant"]
        else:
            raise Exception(f'Unknown cell type {cell["cell_type"]}')

        lines = cell["source"].split("\n")
        first_line = lines[0]
        role, score = get_closest_match(first_line, headers)
        if score > 50:
            valid_role = role.replace("*", "").replace("#", "").strip()
            content = "\n".join(lines[1:]).strip("\n")
        else:
            valid_role = ""
            content = cell["source"]
        message = {"role": valid_role, "content": content, "type": cell["cell_type"]}
        messages.append(message)

    return messages


def fix_missing_roles(messages):
    """
    Fix missing roles in a list of messages.

    :param messages: The list of messages.
    """
    def predict_role(messages_subsequence):
        try:
            openai_client = OpenAI(api_key=openai_api_key)
            response = openai_client.chat.completions.create(
                model="gpt-4-1106-preview",
                messages=[
                    {"role":"system", "content": "Your task is to accurately predict whether the empty role is a User or an Assistant. You are only allowed to reply with a single word: 'User' or 'Assistant'."},
                    {"role":"user", "content": f"Here's a part of the conversation including an empty role:\n\n{messages_subsequence}"}
                ],
                temperature=0,
                seed=42
            )
            print(response.choices[0])
            missing_role = response.choices[0].message.content
            assert missing_role in ["User", "Assistant"]
            return missing_role, None
        except Exception as e:
            return None, e

    errors = []
    for i in range(len(messages)):
        if messages[i]["role"] == "":
            subsequence = messages[max(0, i-2):min(len(messages), i+3)]
            messages[i]["role"], error = predict_role(subsequence)
            if error is not None:
                errors.append(error)
    return messages, errors


def extract_metadata(notebook):
    # # Extract the first cell
    first_cell = notebook.cells[0]
    lines = first_cell["source"].split("\n")
    metadata = {}
    for line in lines:
        if "**Python Topics**" in line:
            metadata["topic"] = line.split(" - ")[1]
        if "**Type**" in line:
            metadata["type"] = line.split(" - ")[1]
        if "**Target Number of Turns (User + Assistant)**" in line:
            metadata["target_turns"] = line.split(" - ")[1]

    return metadata


def notebook_parser(notebook):
    messages = extract_messages(notebook)
    metadata = extract_metadata(notebook)
    messages, errors = fix_missing_roles(messages)
    if errors:
        raise Exception("Failed to predict missing roles.")
    return {"metadata": metadata, "messages": messages}


def split_messages_into_turns(messages):
    turns = []
    current_role_steps = []
    if not messages:
        return {
            "status": "ERROR",
            "reason": "No messages were provided to turn splitter.",
        }

    current_role = messages[0]["role"]
    for message in messages:
        role = message["role"]
        if current_role != role:
            turns.append({"role": current_role, "steps": current_role_steps})
            current_role_steps = []
            current_role = role
        current_role_steps.append(
            {"type": message["type"], "content": message["content"]}
        )
    if current_role_steps:
        turns.append({"role": current_role, "steps": current_role_steps})

    for turn in turns:
        if turn["role"] == "Assistant":
            turn["role"] = Roles.LLM.value
        elif turn["role"] == "User":
            turn["role"] = Roles.HUMAN.value
        else:
            return {"status": "ERROR", "reason": "Contains unrecognized header"}

    grouped_turns = []
    for i in range(0, len(turns), 2):
        group = turns[i : i + 2]
        grouped_turns.append(group)
    return {"status": "OK", "turns": grouped_turns}


def notebook_to_turns(notebook):
    parsed_notebook = {**notebook_parser(notebook)}
    turns = split_messages_into_turns(parsed_notebook["messages"])
    if turns["status"] == "OK":
        return turns["turns"]
    else:
        raise Exception("Bad notebook")
