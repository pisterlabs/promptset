import os
import fnmatch
import csv
import base64
import random   
import re
import json
from openai import OpenAI

client = OpenAI()

FIND_IMAGES = True
LABEL_IMAGES = True
CREATE_FOLDER_STRUCTURE = True
IMGS_TO_LABEL_COUNT = 10

DIRS_TO_INDEX = ['/Users/bmachado/Desktop', '/Users/bmachado/Downloads', '/Users/bmachado/Documents', '/Users/bmachado/Pictures']

def list_images(directories, extensions=['*.png', '*.jpg', '*.jpeg', '*.gif']):
    matches = []
    blacklist = ['/.', '/com.', '/Library']
    for directory in directories:
        for root, dirnames, filenames in os.walk(directory):
            if any(blacklisted in root for blacklisted in blacklist):
                continue
            for extension in extensions:
                for filename in fnmatch.filter(filenames, extension):
                    matches.append(os.path.join(root, filename))
                    print(os.path.join(root, filename))
    return matches

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def label_my_image(img_base64):
    IMG_PROMPT = """
    I need to make a really goood file name for this image right now its very hard to search for it!

    I need you to output JSON that looks like this:

    ```json
    {
        "file_name": "<new_semantically_relevant_file_name>.png"
    }
    ```\
    
    We are going to put this in a REPL so only output the JSON and nothing else.
    """

    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": IMG_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": f"data:image/jpeg;base64,{img_base64}",
                    },
                ],
            }
        ],
        max_tokens=300,
    )

    resp_text = response.choices[0].message.content
    
    code_match = re.search(r'```json(.*?)```', resp_text, re.DOTALL)
    code_content = code_match.group(1) if code_match else ""

    return code_content

def create_folder_structure(labels_list):
    labels_string = "\n".join(str(label) for label in labels_list)

    FOLDER_PROMPT = f"""\
    I have all of these files and I need you to organize them for me by putting them in folders.

    Here are the files:
    {labels_string}

    suggest a folder structure for me to put them in, by coming up with a new folder structure and outputting the new folder structure in JSON.

    I need to make a really goood file name for this image right now its very hard to search for it!

    I need you to output JSON that looks like this:

    ```json
    <new folder structure>
    ```\
    
    We are going to put this in a REPL so only output the JSON and nothing else.
    """

    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": FOLDER_PROMPT},
                ],
            }
        ],
        max_tokens=300,
    )

    resp_text = response.choices[0].message.content
    
    code_match = re.search(r'```json(.*?)```', resp_text, re.DOTALL)
    code_content = code_match.group(1) if code_match else ""

    return code_content


def create_directories_and_move_files(structure, parent_dir=''):
    for key, value in structure.items():
        if isinstance(value, list):  # If the value is a list, then the key is a directory
            for file in value:
                for item in data:
                    if item['new_label'] == file:
                        src = item['file_path']
                        dst = os.path.join(parent_dir, key, file)
                        os.makedirs(os.path.dirname(dst), exist_ok=True)
                        os.symlink(src, dst)
        else: 
            create_directories_and_move_files(value, os.path.join(parent_dir, key))

if __name__ == "__main__":

    if FIND_IMAGES:
        print("GPT: Finding images...")
        images = list_images(DIRS_TO_INDEX)

        with open('image_paths.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            for image in images:
                writer.writerow([image])

    if LABEL_IMAGES:
        print("GPT: Labeling images...")
        with open('image_paths.csv', newline='') as f:
            reader = csv.reader(f)
            images = list(reader)

        labeled_images = []
        labeled_image_paths = []
        if os.path.exists('labeled_images.json'):
            with open('labeled_images.json', 'r') as f:
                labeled_images = json.load(f)
                labeled_image_paths = [img['file_path'] for img in labeled_images]

        unlabeled_images = [img for img in images if img[0] not in labeled_image_paths]
        sample_images = random.sample(unlabeled_images, IMGS_TO_LABEL_COUNT)

        for image in sample_images:
            image_path = image[0]
            encoded_image = encode_image(image_path)
            new_file_name = json.loads(label_my_image(encoded_image))['file_name']
            labeled_images.append({"file_path": image_path, "new_label": new_file_name})

        with open('labeled_images.json', 'w') as json_file:
            json.dump(labeled_images, json_file)

        with open('labeled_images.json', 'w') as json_file:
            json.dump(labeled_images, json_file)

    with open('labeled_images.json', 'r') as f:
                data = json.load(f)

    if CREATE_FOLDER_STRUCTURE:
        print("GPT: Creating folder structure...")

        new_file_names = [item['new_label'] for item in data]
        
        structure = json.loads(create_folder_structure(new_file_names))
        
        with open('structure.json', 'w') as f:
            json.dump(structure, f)

    print("Doing it!")
    with open('structure.json', 'r') as f:
        structure = json.load(f)

    create_directories_and_move_files(structure)
