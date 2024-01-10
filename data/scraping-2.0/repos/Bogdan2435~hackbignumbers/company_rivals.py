import os
import openai
import pandas as pd
import requests
from diffusers import StableDiffusionPipeline
from IPython.display import Image
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image

IMAGE_FIRST_COMPANY = ''
IMAGE_SECOND_COMPANY = ''

# openai.api_key = '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' #CHEIE NOUA
openai.api_key = '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' #CHEIE BOGDAN

df = pd.read_csv('data/csv/MERGED-from-0-100000-relevant-data-from-websites.csv')

# input1 = "ncoilheat.org"

print("Give the first company: ")
input1 = input() 

for iter, row in df.iterrows():
    if row[0] == input1:
        url = row[0]
        keywords = row[1]
        meta_description = row[2]
        page_type = row[3]
        relevant_paragraphs = row[4]
        break

company_prompt1 = {"COMPANY": url, "KEYWORDS": keywords, "DESCRIPTION": meta_description, "RELEVANT PARAGRAPHS": relevant_paragraphs}
company_prompt1 = str(company_prompt1)
# print(company_prompt1)

print("Give the second company: ") 
input2 = input()

print("Give the first company logo path: ") 
IMAGE_FIRST_COMPANY = input()

print("Give the second company logo path: ") 
IMAGE_SECOND_COMPANY = input()


for iter, row in df.iterrows():
    if row[0] == input2:
        url = row[0]
        keywords = row[1]
        meta_description = row[2]
        page_type = row[3]
        relevant_paragraphs = row[4]
        break

company_prompt2 = {"COMPANY": url, "KEYWORDS": keywords, "DESCRIPTION": meta_description, "RELEVANT PARAGRAPHS": relevant_paragraphs}
company_prompt2 = str(company_prompt2)


#IMAGE 2 TEXT

model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("mps")
model.to(device)



max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}
def predict_step(image_paths):
  images = []
  for image_path in image_paths:
    i_image = Image.open(image_path)
    if i_image.mode != "RGB":
      i_image = i_image.convert(mode="RGB")

    images.append(i_image)

  pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
  pixel_values = pixel_values.to(device)

  output_ids = model.generate(pixel_values, **gen_kwargs)

  preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
  preds = [pred.strip() for pred in preds]
  return preds


# predict_step(['data/logo_company.png'])
print("\n__________________Logo to text_______________\n")
img2txt1 = predict_step([IMAGE_FIRST_COMPANY]) # ['a woman in a hospital bed with a woman in a hospital bed']
print("%s logo description"%input1 + str(img2txt1))

img2txt2 = predict_step([IMAGE_SECOND_COMPANY])
print("%s logo description"%input2 + str(img2txt2))

# #DATA CORRELATION

with open("data/system-prompt-rivals.txt", "r") as f:
    content_system = f.read()

response1 = openai.ChatCompletion.create(
  # model="gpt-4",
  model="gpt-3.5-turbo",
  messages=[
    {
      "role": "system",
      "content": content_system
    },
    {
      "role": "user",
      "content": company_prompt1 + "LOGO:" + str(img2txt1)
    }
  ],
  temperature=0.9,
  max_tokens=500,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0
)

print("\n__________________Company %s_______________\n"%input1)
print(response1['choices'][0]['message']['content'])

response2 = openai.ChatCompletion.create(
  # model="gpt-4",
  model="gpt-3.5-turbo",
  messages=[
    {
      "role": "system",
      "content": content_system
    },
    {
      "role": "user",
      "content": company_prompt2 + "LOGO:" + str(img2txt2)
    }
  ],
  temperature=0.9,
  max_tokens=500,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0
)
print("\n__________________Company %s_______________\n"%input2)
print(response2['choices'][0]['message']['content'])

with open("data/system-prompt-rivals-comparison.txt", "r") as f:
    content_system_comparison = f.read()

response3 = openai.ChatCompletion.create(
  # model="gpt-4",
  model="gpt-3.5-turbo",
  messages=[
    {
      "role": "system",
      "content": content_system_comparison
    },
    {
      "role": "user",
      "content": "FIRST COMPANY: " + input1 + response1['choices'][0]['message']['content'] + "SECOND COMPANY: " + input2 + response2['choices'][0]['message']['content']
    }
  ],
  temperature=0.9,
  max_tokens=500,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0
)
print("\n__________________Comparison_______________\n")
print(response3['choices'][0]['message']['content'])