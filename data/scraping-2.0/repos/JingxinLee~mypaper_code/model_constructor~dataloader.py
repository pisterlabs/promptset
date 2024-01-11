import os
from openai import OpenAI
from langchain.document_loaders import UnstructuredMarkdownLoader
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
import numpy as np
from torchvision import transforms
from transformers import AutoTokenizer
from PIL import Image
import torchaudio

from preprocess import augment_audio, augment_image, augment_text, augment_video, infer_data_modality, infer_folder_modality, infer_modality, normalize_text, normalize_image, normalize_audio, normalize_video


_ = load_dotenv(find_dotenv())  # read local .env file
client = OpenAI()
OpenAI.api_key = os.getenv("OPENAI_API_KEY")


# model = "gpt-3.5-turbo-1106" or "gpt-4-1106-preview"
def get_completion(prompt, model="gpt-3.5-turbo-1106"):
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model=model, messages=messages, temperature=0
    )
    return response.choices[0].message.content

# Preprocess 1. PandaGPT -> Text



# Preprocess 2. Stable Diffusion -> Picture
# Statble Diffusion API： bash webui.sh -f --api
def text2image(prompt, steps):
    url = "http://127.0.0.1:7860"
    payload = {
    "prompt": prompt,
    "steps": steps
    }
    response = requests.post(url=f'{url}/sdapi/v1/txt2img', json=payload)
    r = response.json()
    image = Image.open(io.BytesIO(base64.b64decode(r['images'][0])))
    image.save(f"{prompt}.png")
    


# Normalize Function  
def Normalize(modality):
    if modality == "text":
        normalize_function = normalize_text
    if modality == "image":
        normalize_function = normalize_image
    if modality == "audio":
        normalize_function = normalize_audio
    if modality == "video":
        normalize_function = normalize_video
    else:
        raise ValueError("The modality is not supported.")
    return normalize_function


# Augmentator
def Augmentator(modality):
    if modality == 'text':
        augment_function = augment_text
    if modality == 'image':
        augment_function = augment_image
    if modality == 'audio':
        augment_function = augment_audio
    if modality == 'video':
        augment_function = augment_video
    else:
        raise ValueError("The modality is not supported.")
    return augment_function


def main():
    file_path = "/home/ddp/nlp/github/paper/mypaper_code/model_constructor/MarkdownFiles"  # 这可以是文件或文件夹的路径
    modality = infer_modality(path)
    print(f"The modality of '{path}' is: {modality}")
    
    # 1. Preprocess 
    text2image_prompt = "Please generate an image based on the text of the file gave you" 
    text2image(prompt=text2image_prompt, steps=5)
    
    # 2. Normalize
    normalize_func = Normalize(modality)
    
    # 3. Augmentator 
    augmentator = Augmentator(modality)

    # 4. DataLoader
    dataloader_prompt = f"""
        Your task is to generate a dataloader code snippet for the following modality data. The MODALITY and FILE_PATH is enclosed in the triple backticks.
        You need to give a Huggingface dataloader code snippet that can load the data into a PyTorch/Tensorflow dataset.
        you should follow these steps when you write the code snippet:
        1. Import the necessary libraries,such as datasets, torchvision, torchaudio,transformers, etc.
        2. Use 'load_dataset' function to load the dataset using the file path I gave you.
        3. Augment the dataset using the 'augmentator' function.Then, normalize the dataset using the 'normalize_func' function. \ 
        Both augmentator and normalize_func are given to you and are enclosed in the triple backticks.
        3. Use 'map' function to preprocess the dataset. Transform the data into the format that the model can understand.
        4. Split the dataset at least into 'train' and 'test' sets.
        5. Use 'DataLoader' function to create a dataloader for the dataset.
        
        At the end, please return the dataloader code snippet with some usage code snippet.
        MODALITY: ```{modality}```
        FILE_PATH: ```{file_path}```
        augmentator: ```{augmentator}```
        normalize_func: ```{normalize_func}```
        
    """
    dataloader_response = get_completion(dataloader_prompt)
    print(dataloader_response)
    augment_text = Augmentator(modality='text')
    print(augment_text)


if __name__ == "__main__":
    main()
