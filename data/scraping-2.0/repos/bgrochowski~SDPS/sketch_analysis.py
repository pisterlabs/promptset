from openai import OpenAI, RateLimitError
import numpy as np
client = OpenAI(api_key='YOUR_API_KEY')
import base64
import os
import random
import time
import pickle

def encode_example_images(excluded_task=None):
    base_path = 'assets'
    base_path = os.path.join(base_path, 'prompts')
    example_files = [f for f in os.listdir(base_path) if (f.endswith('.png') or f.endswith('.jpg') or f.endswith('.py'))]
    encoded_images = ['' for _ in range(16)]
    prompts = ['' for _ in range(16)]
    
    for file in example_files:
        file_path = os.path.join(base_path, file)
        if file.endswith('.py'):
            with open(file_path, 'r') as file:
                # print(file.name)
                if file.name == 'assets/prompts/ex_code.py':
                    example_code = file.read()
                    prompts[15] = example_code
                else:
                    index = int(file.name.split('/')[-1].split('.')[0]) - 1
                    example_code = file.read()
                    prompts[int(index)] = example_code
        else:
            with open(file_path, 'rb') as file:
                # print(file.name)
                if file.name == 'assets/prompts/ex_sketch.jpg':
                    image_data = file.read()
                    encoded_images[15] = base64.b64encode(image_data).decode('utf-8')
                else:
                    index = int(file.name.split('/')[-1].split('.')[0]) - 1
                    image_data = file.read()
                    encoded_images[int(index)] = base64.b64encode(image_data).decode('utf-8')

    base_path = 'assets'
    base_path = os.path.join(base_path, 'envs')
    example_files = [f for f in os.listdir(base_path) if (f.endswith('.png') or f.endswith('.jpg'))]
    encoded_envs = ['' for _ in range(16)]
    for file in example_files:
        file_path = os.path.join(base_path, file)
        with open(file_path, 'rb') as file:
            if file.name == 'assets/envs/ex_env.png':
                image_data = file.read()
                encoded_envs[15] = base64.b64encode(image_data).decode('utf-8')
            else:
                index = int(file.name.split('/')[-1].split('.')[0]) - 1
                image_data = file.read()
                encoded_envs[int(index)] = base64.b64encode(image_data).decode('utf-8')
    taskset1 = [1, 3, 4, 6, 7, 9, 10, 12]
    taskset2 = [5, 13]
    taskset3 = [8, 11, 15]
    taskset4 = [2, 14]
    if excluded_task is not None:
        # include only the tasks from the taskset of excluded task, without the excluded task
        if excluded_task in taskset1:
            taskset = taskset1
        elif excluded_task in taskset2:
            taskset = taskset2
        elif excluded_task in taskset3:
            taskset = taskset3
        elif excluded_task in taskset4:
            taskset = taskset4
        
        taskset.remove(excluded_task)

        encoded_images = [encoded_images[i-1] for i in taskset]
        prompts = [prompts[i-1] for i in taskset]
        encoded_envs = [encoded_envs[i-1] for i in taskset]

        # encoded_images = encoded_images[:excluded_task-1] + encoded_images[excluded_task:]
        # prompts = prompts[:excluded_task-1] + prompts[excluded_task:]
        # encoded_envs = encoded_envs[:excluded_task-1] + encoded_envs[excluded_task:]
    return encoded_images, prompts, encoded_envs

def select_and_encode_images():
    base_path = 'assets'
    envs_path = os.path.join(base_path, 'envs')
    encoded_images = []

    # Select a random image from 'envs'
    envs_files = [f for f in os.listdir(envs_path) if f.endswith('.png')]
    selected_env = random.choice(envs_files)
    envs_image_path = os.path.join(envs_path, selected_env)

    # Encode the selected image from 'envs'
    with open(envs_image_path, 'rb') as file:
        image_data = file.read()
        encoded_images.append(base64.b64encode(image_data).decode('utf-8'))

    # Select and encode an image from the folder with the same number as the selected 'envs' image
    folder_number = selected_env.split('.')[0]
    print(f"Selected environment number: {folder_number}")
    target_folder_path = os.path.join(base_path, folder_number)
    if os.path.exists(target_folder_path):
        target_files = [f for f in os.listdir(target_folder_path) if f.endswith('.png')]
        if target_files:
            selected_image = random.choice(target_files)
            target_image_path = os.path.join(target_folder_path, selected_image)
            print(f"Selected target image: {selected_image}")

            with open(target_image_path, 'rb') as file:
                image_data = file.read()
                encoded_images.append(base64.b64encode(image_data).decode('utf-8'))

    return encoded_images

def analyze_sketches(example_images_b64, example_code, example_envs_b64, images_b64):
    # Make a call to the OpenAI API with the images
    messages = [
            {
            "role": "system",
            "content": "Based on the image of the robot environment and the sketch as a robot policy, you are going to generate robot policy code."
            }
    ]
    if len(example_images_b64) > 2:
        selected_indices = random.sample(range(len(example_images_b64)), 2)
    elif len(example_images_b64) == 2:
        selected_indices = [0, 1]
    else:
        selected_indices = [0]
    for i in selected_indices:
        # print(i)
        messages.append(
            {
            "role": "user",
            "content": [
                {"type": "text", "text": "Given the environment image, give me policy code for the robot action represented on the sketch."},
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{example_images_b64[i]}"
                }
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{example_envs_b64[i]}"
                }
                },
            ],
            }
        )
        messages.append({
                "role": "assistant",
                "content": example_code[i]
                })
    messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": "Given the environment image, give me policy code for the robot action represented on the sketch. Provide only the code, with no extra explanation."},
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{images_b64[0]}"
                }
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{images_b64[1]}"
                }
                },
            ],
            })
    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=messages,
        max_tokens=4096,
    )

    # Return the response
    return response

tasks = range(1, 16) 

for task in tasks:
    print(f"Task {task}")
    example_images_b64, example_code, example_envs_b64 = encode_example_images(task)
    with open(f'assets/envs/{task}.png', 'rb') as file:
        image_data = file.read()
        env_b64 = base64.b64encode(image_data).decode('utf-8')
    task_files = [f for f in os.listdir(f'assets/{task}') if (f.endswith('.png') or f.endswith('.jpg') or f.endswith('.jpeg'))]
    # task_images_b64 = []
    for file in task_files:
        with open(f'assets/{task}/{file}', 'rb') as file:
            imagename = file.name.split('/')[-1].split('.')[0]
            print(imagename)
            # if exists in results, skip
            if os.path.exists(f'results/{task}/analysis_{imagename}.pkl'):
                print(f"Skipping {imagename}")
                continue
            image_data = file.read()
            task_images_b64 = base64.b64encode(image_data).decode('utf-8')
            images_b64 = [env_b64, task_images_b64]
        done = False
        while not done:
            try:
                analysis = analyze_sketches(example_images_b64, example_code, example_envs_b64, images_b64)
                done = True
            except RateLimitError as e:
                # print(f"Rate limit error. Sleeping for 10 seconds.")
                print(e)
                time.sleep(120)
        print(analysis)
        os.makedirs(os.path.dirname(f'results/{task}/'), exist_ok=True)
        with open(f'results/{task}/analysis_{imagename}.pkl', 'wb') as pkl_file:
            pickle.dump(analysis, pkl_file)
        time.sleep(10)
    

# images_b64 = select_and_encode_images()
# analysis = analyze_sketches(example_images_b64, example_code, example_envs_b64, images_b64)
# print(analysis)

