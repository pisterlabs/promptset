import requests
from openai import OpenAI
import json
import os
from requests.models import Response
import os
import shutil
import pandas as pd
import time
from api_key import api_key
from utils import *




default_prompt = "Please describe this design in a way that would \
            allow DALL-E 3 to recreate it if used as a prompt. Only include the prompt and nothing else"

headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }


def convert_sketch_to_text(image, additional_info=""):
    payload = {
    "model": "gpt-4-vision-preview",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": default_prompt
          },
          {
            "type": "image_url",
            "image_url": {
              "url": f""
            }
          }
        ]
      }
    ],
    "max_tokens": 300
    }
    if additional_info:
        payload["messages"][0]["content"][0]["text"] += additional_info
    payload["messages"][0]["content"][1]["image_url"]["url"] = f"data:image/jpeg;base64,{image}"
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    json_data = response.json()
    return json_data


class ImageResponse:
    def __init__(self):
        self.data = []
        self.created = ""
    
    def add_image(self, image_response):
        if not self.created:
            self.created = image_response.created

        self.data.extend(image_response.data)

def convert_text_to_image(text, num_images, model="dall-e-3"):
    client = OpenAI(api_key=api_key)
    if model == "dall-e-3":
        response = ImageResponse()
        for i in range(num_images):
            response.add_image(client.images.generate(
                model=model,
                prompt=text,
                size="1024x1024",
                quality="standard",
                n=1
            ))
    else:
        response = client.images.generate(
            model=model,
            prompt=text,
            size="1024x1024",
            quality="standard",
            n=num_images
        )
    return response

def save_images(images, output_dir):
    image_dir = f"{output_dir}/images"
    os.mkdir(image_dir)
    for i, url in enumerate(images):
        response = requests.get(url)
        with open(f"{image_dir}/image_{i}.png", "wb") as image_file:
            image_file.write(response.content)
    return True
    


def create_json_from_image_response(image_response):
    obj = {"data" : []}
    for data in image_response.data:
        obj["data"].append({
            "prompt" : data.revised_prompt,
            "url" : data.url
        })

    return obj



def load_prompt(img_name, csv_file_path):
    df = pd.read_csv(csv_file_path)
    row = df.loc[df["Image_ID"] == img_name]
    if not row["text"].values:
        return ""
    return row["text"].values[0]


def sketch_to_images(input_path, output_dir, num_samples=4):
    """
    Given path to a sketch, populates output_dir with num_samples of images
    along with the text prompt used to generate the images
    """
    shutil.copyfile(input_path, f"{output_dir}/original.png")

    sketch = load_sketch(input_path)
    image_name = os.path.splitext(os.path.basename(input_path))[0]
    additional_prompt_info = load_prompt(image_name, "data/sketch_drawing.csv")
    print ("Prompt info:", additional_prompt_info)
    additional_info = "" if not additional_prompt_info else f" Here is some information about the sketch: {additional_prompt_info}"
    prompt_response = convert_sketch_to_text(sketch, additional_info)
    save_json(prompt_response, f"{output_dir}/prompt_response.json")
    print ("Prompt response", prompt_response)

    dalle_prompt = prompt_response['choices'][0]['message']['content']
    print("Generating image")
    image_response = convert_text_to_image(dalle_prompt, num_images=num_samples)
    save_json(create_json_from_image_response(image_response), f"{output_dir}/dalle_response.json")


    images = [image_response.data[i].url for i in range(len(image_response.data))]
    save_images(images, f"{output_dir}")
    return True
    
def check_valid_directory(dirpath):
    """
    Checks if dirpath must only contain png files
    """
    all_png = True
    for dirpath, _, filenames in os.walk(dirpath):
        for f in filenames:
            if not f.endswith(".png"):
                print (f)
                all_png = False

    return all_png
    pass



def sketches_to_dataset(input_dir, output_dir):
    if not check_valid_directory(input_dir):
        raise Exception("Directory contains non-png files")
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    for dirpath, _, filenames in os.walk(input_dir):
        print ("filenames", filenames)
        for f in filenames:
            dataset_dir = f"{output_dir}/{os.path.splitext(f)[0]}"
            print ("Processing", f)
            if os.path.isdir(dataset_dir):
                print ("directory already exists")
                continue
            os.mkdir(dataset_dir)
            res = None
            while res is None:
                try:
                    res = sketch_to_images(os.path.abspath(os.path.join(dirpath, f)), dataset_dir)
                    print ("Generated image")
                    time.sleep(5)
                except Exception as error:
                    print("An error occured: ", error)
                
            
            print ("Finished processing", f, "\n\n")




if __name__ == "__main__":
    sketches_to_dataset("milk_frother_dataset", "dataset_full")