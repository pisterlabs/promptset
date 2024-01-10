import random
import string
import openai
import os
import time
import json
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# Get the API key for the OpenAI API.
openai.api_key = os.getenv('API_KEY')


def get_solution_sys_prompt(question, solution):
    with open("gpt_expand_sme_soln.txt") as file:
        generate_solution_state = file.read()
        return generate_solution_state.format(question=question, solution=solution)


def run_chat_completion(history):
    """
    Run openai's chat completion on the current conversation history.

    Args:
        history (str): current conversation history

    Returns:
        str: chat completion response
    """
    completion = openai.ChatCompletion.create(
        model="gpt-4-0613",
        messages=history,
    )

    response = completion['choices'][0]['message']['content']

    return response


def run_solution_gpt(question, solution):
    """
    Run tutorbot gpt.

    Args:
        question (str): initial question that the student asked
        solution (str): provided step-by-step textbook solution

    Returns:
        str: generated solution
    """
    solution_history = [
        {"role": "system", "content": ""},
        {"role": "user", "content": get_solution_sys_prompt(question, solution)},
    ]

    return run_chat_completion(solution_history)


def validate_json_string(response: str):
    """
    Corrects the format of JSON string.

    Args:
        response (str): JSON string

    Returns:
        str: correctly formatted JSON string
    """
    # Add any missing commas
    response = response.replace("\"\n\"", "\",\n\"")

    # Strip any extra text before first bracket and after last bracket
    response = response[response.find('{'):]
    response = response[:response.rfind('}') + 1]

    return response


def generate_solution(question, solution):
    """
    Generates a dictionary that represents the generated solution as a JSON object.

    Args:
        question (str): initial question that the student asked
        solution (str): provided step-by-step textbook solution

    Returns:
        object: dictionary that represents the generated solution JSON object
    """
    response = run_solution_gpt(question, solution)

    # Fix formatting for JSON object
    response = validate_json_string(response)
    print(response)

    try:
        response_json = json.loads(response, strict=False)
    except Exception:
        print("regenerating\n")
        response_json = generate_solution(question, solution)

    return response_json


def run_on_problem_set(problem_set_filename: str, starting_row_idx=-1, user_ending_row_idx=-1):
    """
    Generate solutions sequentially for a problem set. File must be a .csv with a "Question" and "Solution" column.

    Args:
        problem_set_file (str): name of the problem set file
    """
    try:
        df = pd.read_csv(problem_set_filename)
    except FileNotFoundError:
        print("Problem set file not found.")
        exit()

    if starting_row_idx >= len(df.index):
        raise IndexError("Starting row index is out of bounds.")

    # Assign ending row index if included
    if user_ending_row_idx != -1:
        ending_row_idx = user_ending_row_idx
    else:
        ending_row_idx = len(df.index)

    if ending_row_idx > len(df.index):
        raise IndexError("Ending row index is out of bounds.")

    # Create column for 'step-by-step solution' if doesn't exist
    solutions_col_name = f'Step-By-Step Solution'
    if solutions_col_name not in df:
        df.insert(2, solutions_col_name, None)

    # Find row with the first empty 'step-by-step solution' cell if starting_row_idx is not specified
    is_full = True
    if starting_row_idx == -1:
        for index, row in df.iterrows():
            if pd.isnull(df.loc[index, solutions_col_name]):
                starting_row_idx = index
                is_full = False
                break

    if is_full and starting_row_idx == -1:
        # If dataframe is full, starting_row_idx should never have updated
        # Only care about full dataframe when starting_row_idx is unspecified
        print("Dataframe is full")
    else:
        # Else if we specified a starting_row_idx, always start at that idx
        print(f"Starting at row {starting_row_idx}")

        for index, row in df.iloc[starting_row_idx: ending_row_idx].iterrows():
            question = row['Question']
            solution = row['Solution']
            # print("question:\n" + question + "\n")
            # print("solution:\n" + solution + "\n")

            gpt_solution_json = generate_solution(question, solution)
            solution_json = {"Question": question, "Original Solution": solution,
                             **gpt_solution_json}  # unpack gpt_solution_json

            # Set column to be contents of solution file
            df.at[index, solutions_col_name] = gpt_solution_json

            # Generate filename with random string of 6 characters
            random_string = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
            solution_json_filename = f"solution_{index}_{random_string}"

            solution_json_file = f"datasets/v4/solution_jsons/{solution_json_filename}.json"

            json.dump(solution_json, open(solution_json_file, "w", encoding="utf-8"), indent=4)

            # Overwrite csv file to save dataframe
            df.to_csv(problem_set_filename, index=False)

            time.sleep(1)


if __name__ == '__main__':
    run_on_problem_set('problem set.csv')