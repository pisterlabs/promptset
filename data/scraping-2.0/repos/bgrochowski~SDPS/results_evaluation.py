from openai import OpenAI, RateLimitError
import numpy as np
client = OpenAI(api_key='YOUR_API_KEY')
import base64
import os
import random
import time
import pickle

results = [[] for _ in range(16)]
tasks = ["Move the orange box to the lower right corner",
         "Move the red book from the left shelf to the right one",
         "Arrange blocks in a square pattern on the table",
         "Sort colored boxes into respective containers",
         "Draw a simple star pattern on the whiteboard",
         "Stack boxes in a vertical pyramid",
         "Arrange the boxes in a diagonal",
         "Navigate through a simple slalom on the floor",
         "Move a red-colored object to another bowl",
         "Move the series of boxes from horizontal to vertical line",
         "Follow the path drawn on the ground",
         "Move an apple to another bowl",
         "Write 'robot' on the whiteboard",
         "Sort the objects based on size",
         "Go in a square around the stool as many times as needed, check each step if there is a coca cola can, only stop moving when you see it"]

# load prompt_env and encode to base64 from assets/envs/ex_env.png
with open('assets/envs/ex_env.png', 'rb') as file:
    image_data = file.read()
    prompt_env = base64.b64encode(image_data).decode('utf-8')
# load prompt code from assets/prompts/ex_code.py
with open('assets/prompts/ex_code.py', 'r') as file:
    prompt_code = file.read()
prompt_task_description = "Move the blue-colored block in between the red block and the second block from the left"

def evaluate_gpt(code, task_description, env_base64):
    messages = [
            {
            "role": "system",
            "content": "Based on the high level code and image of the robot environment I provide you, tell me if these correspond to the same task I describe. By correspond, I mean if this code is likely to have the desired effect of completing the task, despite some assumptions and simplifications or lack of exact precision. The goal is to determine whether the code is likely to be ment to execute the prompted task."
            }
    ]
    messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": "Based on the high level code I provide you tell me if these correspond to the same task I described. Answer only 'yes' or 'no'."},
                # {
                # "type": "image_url",
                # "image_url": {
                #     "url": f"data:image/jpeg;base64,{prompt_env}",
                #     "detail": "low"
                # }
                # },
                {
                "type": "text",
                "text": prompt_code
                },
                {
                "type": "text",
                "text": prompt_task_description
                }
            ],
            }
        )
    messages.append({
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": "yes"
            }
        ]
    })
    messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": "Based on the high-level, general robot code I provide you and the environment image, tell me if these correspond to the same task I described. By correspond, I mean if this code is likely to have the desired effect of completing the robot task, given its environment. Elaborate on your answer, but start with 'Yes' or 'No'."},
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{env_base64}",
                    "detail": "low"
                }
                },
                {
                "type": "text",
                "text": code
                },
                {
                "type": "text",
                "text": task_description
                }
            ],
            })
    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=messages,
        max_tokens=4096,
    )

    # Return the response
    return response

for task in range(1, 16):
    # open results/{task} folder, read all files
    # for each file, read the results, use gpt4 to evaluate
    # add the results to the results list
    # save the results list to results_evaluation.txt
    with open(f'assets/envs/{task}.png', 'rb') as file:
        image_data = file.read()
        env_base64 = base64.b64encode(image_data).decode('utf-8')
    task_description = tasks[task-1]
    print(task_description)
    base_path = 'results'
    base_path = os.path.join(base_path, str(task))
    print(base_path)
    example_files = [f for f in os.listdir(base_path) if (f.endswith('.pkl'))]
    for file in example_files:
        file_path = os.path.join(base_path, file)
        with open(file_path, 'rb') as file:
            # print filename
            print(file_path)
            code = pickle.load(file).choices[0].message.content
            print(code)
        done = False
        while not done:
            try:
                response = evaluate_gpt(code, task_description, env_base64)
                print(response.choices[0].message.content)
                results[task-1].append(response.choices[0].message.content)
                with open('results_evaluation.pkl', 'wb') as file:
                    pickle.dump(results, file)
                done = True
            except RateLimitError as e:
                print(e)
                time.sleep(5)
        time.sleep(25)

with open('results_evaluation.pkl', 'wb') as file:
    pickle.dump(results, file)

