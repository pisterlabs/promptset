import openai
import os
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

system = "System: "
user = "You: "
welcome_instruction = """Welcome! I am Jupyter Bot. Your personal Jupyter Notebook assistant. 
To get started. Please provide me the context of your notebook and the datasets.
Here is an example:
The notebook I am writing is about .... These are the given dataset:
games.csv - metadata about games
turns.csv - all turns from start to finish of each game
To end type '$'.
"""
persona = "You are an experienced data scientist named Sarah, specializing in Jupyter Notebook. " \
"Your role is to expertly guide users in creating their notebooks, providing step-by-step assistance " \
"while carefully following their instructions to ensure a smooth and successful notebook creation process."
log = [
    {"role": "system", "content": persona}
]
custom_prompts = []
more_detail_prompt = """\nAsk the user to provide you with more detail about each dataset e.g., Can you please provide more detail about each dataset and their variables? Do not generate extra informations."""
goal_prompt = """\nAsk the user what their goal is e.g., What is your goal for this task based on the data provided?"""
general_step_prompt = """\nGenerate the required step to achieve the user's goal as detail as possible. Do not generate any code."""
custom_prompts.append(more_detail_prompt)
custom_prompts.append(goal_prompt)
custom_prompts.append(general_step_prompt)

start_instruction = """\nTo start. Type the step number. Or type 'start' to start from step 1 and type 'continue' to keep going.\n"""

def run():
    current_step = 0
    num_prompts = 0
    print(f"{system}{welcome_instruction}")
    while True:
        user_prompt = input(f"{user}")
        # stop the program
        if user_prompt == '$':
            break
        elif num_prompts < len(custom_prompts):
            user_prompt += custom_prompts[num_prompts]
        elif type(user_prompt) == int:
            user_prompt = f"Generate the code for step {user_prompt}. Then ask me for the result before moving on to the next step to assure correctness."
        elif user_prompt == 'start' or user_prompt == 'continue':
            user_prompt = f"Generate the code for step {current_step}. Then ask me for the result before moving on to the next step to assure correctness."
            current_step += 1
        # append the user input to the log
        log.append({"role": "user", "content": f"{user_prompt}"},)
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=log
        )
        answer = response.choices[0].message.content
        # append the gpt answer to the log
        log.append({"role": "assistant", "content": f"{answer}"},)
        if num_prompts == len(custom_prompts)-1:
            answer += start_instruction
        num_prompts += 1
        print(f"{system}{answer}\n")

run()