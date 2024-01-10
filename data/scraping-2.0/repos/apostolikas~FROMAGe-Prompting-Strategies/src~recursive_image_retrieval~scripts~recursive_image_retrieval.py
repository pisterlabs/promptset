import os
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
import torch
import torch.nn.functional as F
import pandas as pd
from fromage import models
from visual_storytelling import story_in_sequence
import openai
import time 
import matplotlib.pyplot as plt
import numpy as np

openai.api_key = 'Your API key'



# Initialize the CLIP model and processor

model_dir = './fromage_model/'
model = models.load_fromage(model_dir)
model_clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to("cuda")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
image_directory = "datasets/Archive/Flicker8k_Dataset"

df_images = pd.read_csv('filtered_capt.csv')
df_images.drop_duplicates(subset='cap', inplace=True)


def encode_image(image_path):
    # Open the image file
    with Image.open(image_path) as img:
        # Convert the image to a tensor and move it to the GPU
        image = processor(images=img, return_tensors="pt").to("cuda")
        with torch.no_grad():
            image_features = model_clip.get_image_features(**image)
    return image_features



def caption_this(model, model_output):
    #Prompts a seperate Fromage instance to give a descriptive caption of the image it retrieved earlier
    
    prompt =  [model_output] + ['Q: Give a descriptive caption. \nA:']
    img_caption = model.generate_for_images_and_texts(prompt, max_img_per_ret=0, max_num_rets=0, num_words=15)
    #Print the caption it found for checking
    print('img_caption: ',img_caption[0])

    if ' A: A: A: A: A: A: A: A' in img_caption[0]:
        print("all A's")
        prompt =  [model_output] + ['Q: Caption this image. \nA:']
        img_caption = model.generate_for_images_and_texts(prompt, max_img_per_ret=0, max_num_rets=0, num_words=15)
        print('new caption:', img_caption )
    return str(img_caption)



def gpt_addition(original_prompt, retrieved_caption):
    #Prompts ChatGPT to find differences in the captions, these captions are used to continue our converstation with FROMAGe
    messages = [
    {"role": "system", "content": "You are an AI scientist who needs to come up with new prompts to put into a multimodal model. \
    You receive a string, which includes caption 1 and caption 2. \
    Caption 1 is the ground truth, caption 2 is a caption of a different image. We prompt a multimodal to get an image\
    as close as possible to the ground truth. Therefore we want to have a new prompt that highlights the differences between\
    The groundtruth and caption 2. Consider descriptive properties of words. So: a dog runs in a green field and a dog runs in the grass, contain the same elements.\
    Please return just the prompt that you come up with. No introduction, no explanation.\
    For example:\n\
    Caption 1: 'This is a photo of a dog'\n\
    Caption 2: 'This is a photo of a cat'\n\
    'Reply: 'The image should not contain a cat.'\n\
    Always return a prompt. If you can not find a new prompt, repeat Caption 1.  "
    }
    ]
    while True: 
        try: 
            if retrieved_caption:
                messages.append({"role": "user", "content":f'caption 1: {original_prompt}, caption 2: {retrieved_caption}'})
            chat = openai.ChatCompletion.create(
                model="gpt-3.5-turbo", messages=messages
            )
            reply = chat.choices[0].message.content
            return reply
        except: 
            time.sleep(1)


standard_path = 'visual_storytelling/images/fromage_dialogue/'
# Increase this hyperparameter to upweight the probability of FROMAGe returning an image.
ret_scale_factor = 1.5 
results = []
for index, row in df_images[19:].iterrows():
    #We set up the new variables for the new experiment. 
    p = "find a similar image: " +row['cap'] # Create a new initial prompt
    #empty all lists
    input_context = []
    text = ''
    all_outputs = []
    experiment_result = []
    #Get ground image features
    image_path = os.path.join(image_directory, row['image_id'])
    ground_features = encode_image(image_path)
    
    for experiment in range(5):
        # Add Q+A prefixes for prompting. This is helpful for generating dialogue.
        text += 'Q: ' + p + '\nA:'
        # Concatenate image and text.
        model_prompt = input_context + [text]
        print('=' * 30)
        print('modelPROMPT:' , model_prompt)
        print('=' * 30)
        while True:
            try:
                model_outputs = model.generate_for_images_and_texts(
                    model_prompt, num_words=32, ret_scale_factor=ret_scale_factor, max_num_rets=1)
                break
            except:
                print('exception at ', {index})
                continue

    
        text += ' '.join([s for s in model_outputs if type(s) == str]) + '\n'
        # Format outputs.
        if type(model_outputs[0]) == str:
            model_outputs[0] = 'FROMAGe:  ' + model_outputs[0]
            try: 
                #save images for inspection. 
                img_name = f'augmenting_set_{index}_{experiment}.png'
                output_path = standard_path + img_name
                img  = model_outputs[-1][0]
                img.save(output_path)
                print('img saved')
            except: 
                continue
        if type(model_outputs[-1][0]) == Image.Image:
            output_features = encode_image(output_path)
            cos_sim =  F.cosine_similarity(ground_features, output_features)
            experiment_result.append(float(cos_sim))
            differences = caption_this(model,model_outputs[-1][0])
            extended_dialogue = gpt_addition(row['cap'] ,differences) 
        else: 
            extended_dialogue = 'Please respond with an image that suits the previous prompt.'
        #show the new dialague: 
        print('extended dialogue:' , extended_dialogue)
        p = extended_dialogue

    #Some logic to store our results 
    result = {
        "original_image": row['image_id'],
        "original_caption": row['cap'],  
        "similarity_augmenting":  experiment_result
    }
    
    results.append(result)
    df_result = pd.DataFrame(results)
    df_result.to_csv('prompt_similarity_last.csv')

