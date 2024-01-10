import os, sys, re
import pickle as pkl
import pandas as pd
from openai import OpenAI
client = OpenAI()
import base64
import time


# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def get_gpt4v_response(text, base64_image):
    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Given the image as premise, and the following sentence as hypothesis, assign one of the three labels: entailment, contradiction, or neutral, based on the relationship conveyed by the image and the text. The definitions of the labels is provided below.\nentailment: if there is enough evidence in the image to conclude that the sentence is true.\ncontradiction: if there is enough evidence in the image to conclude that the sentence is false.\nneutral: if the evidence in the image is insufficient to draw a conclusion about the text.\n\nYou must respond only with one of the three words: entailment, contradiction, or neutral, and nothing else.\n\nsentence: {text}\nlabel:"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpg;base64,{base64_image}"
                        }
                    },
                ],
            }
        ],
        max_tokens=300,
    )
    return response.choices[0].message.content

# load the csv named sampled_data_images.csv
sampled_data = pd.read_csv('sampled_data_images.csv')
# print the head of the sampled data
print(sampled_data.head())

# open a pickle file in append mode to save the responses
fin = open('sampled_data_images_responses.pkl', 'ab')

for i in range(len(sampled_data)):
    print("starting to work with index: ", i)
    # encode the image
    base64_image = encode_image(sampled_data.iloc[i]['image_idx'])
    # get the response from gpt4v
    response_original = get_gpt4v_response(sampled_data.iloc[i]['hypothesis'].strip().replace('.', ''), base64_image)
    response_modified = get_gpt4v_response(sampled_data.iloc[i]['modified'].strip().replace('.', ''), base64_image)
    print("true label: ", sampled_data.iloc[i]['label'], " | response for original: ", response_original, " | response for modified: ", response_modified)
    # append the response to the pickle file
    pkl.dump([sampled_data.iloc[i]['label'], sampled_data.iloc[i]['hypothesis'], sampled_data.iloc[i]['modified'], sampled_data.iloc[i]['image_idx'], response_original, response_modified], fin)
    print("successfully dumped the response to the pickle file")
    time.sleep(10)
