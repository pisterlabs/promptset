import openai
import re
import json
import os
from concurrent.futures import ThreadPoolExecutor
from termcolor import colored

import tenacity

# define your openai api key
# openai.api_key = os.getenv("OPENAI_API_KEY")
# openai.api_key = "sk-..."

@tenacity.retry(wait=tenacity.wait_exponential(multiplier=1, min=4, max=10), stop=tenacity.stop_after_attempt(3))
def final_code_responses(model="gpt-4-0613", pseudo_code="", file_number=0):

    # grab the user requirements from user_instructions.txt
    with open('user_instructions.txt', 'r', encoding="utf-8", errors="ignore") as file:
        user_requirements = file.read()

    response = openai.ChatCompletion.create(
        model=model,
        stream=True,
        top_p=0.3,
        messages=[
            {"role": "system", "content": """You are an expert python programmer who writes excellent python programs in full given user requirements and high level outline. 
            Fully complete each function and method, do not leave any empty or blank with a "pass" statement or only a comment, you write the full funtioning python program code. 
            
            you will always return the entire python code and Fully complete each function and method within ```python ... ``` markdown. give an  explanation of how the code works as docstrings in the beginning. """},
            {"role": "user", "content": f"""
                Original user requirements are:

                {user_requirements}

                ####

                high level outline is:

                {pseudo_code}

                """},
        ]
    )

    responses = ''

    # Define colors for each concurrent process
    colors = ['blue', 'magenta', 'cyan', 'red', 'green', 'yellow']

    
    for chunk in response:
        if "role" in chunk["choices"][0]["delta"]:
            continue
        elif "content" in chunk["choices"][0]["delta"]:
            r_text = chunk["choices"][0]["delta"]["content"]
            responses += r_text
            print(colored(r_text, colors[file_number % len(colors)]), end='', flush=True)

    response_code = re.search(r"(```python\s*.*?```|```\s*.*?```)", responses, re.DOTALL).group(0)

    # Remove the code block formatting from the response
    response_code = re.sub(r"```python", "", response_code)
    response_code = re.sub(r"```", "", response_code)

    # Write the response to the "response.py" file in Final_Code folder. create it if it doesn't exist
    if not os.path.exists("Final_Code"):
        os.makedirs("Final_Code")
    with open(f"Final_Code/response_{file_number}.py", "w", encoding="utf-8", errors="ignore") as f:
        f.write(response_code)    


if __name__ == "__main__":
    with open("winning_pseudo_competitors.json", "r") as f:
        # load the nth competitor to generate high level outline for
        pseudo_Code_object = json.load(f)
        with ThreadPoolExecutor() as executor:
            # submit each competitor's high level outline as a separate task
            tasks = [executor.submit(final_code_responses, pseudo_code=competitor, file_number=int(i)) for i, competitor in pseudo_Code_object.items()]
            # wait for all tasks to complete
            for task in tasks:
                task.result()
