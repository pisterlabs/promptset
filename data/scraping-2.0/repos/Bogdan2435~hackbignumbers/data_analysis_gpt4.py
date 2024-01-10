# import os
# import openai
# import pandas as pd
# import requests
# from diffusers import StableDiffusionPipeline
# import torch

# from IPython.display import Image

# def open_journey(prompt, path):
#     model_id = "prompthero/openjourney"
#     pipe = StableDiffusionPipeline.from_pretrained(model_id)
#     pipe = pipe.to("mps")
#     pipe.enable_attention_slicing()

#     image = pipe(prompt).images[0]
#     image.save(path)

# def logo_diffusinon_checkpoint(prompt, path):
#     API_URL = "https://api-inference.huggingface.co/models/logo-wizard/logo-diffusion-checkpoint"
#     headers = {"Authorization": "Bearer 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"}

#     def query(payload):
#       response = requests.post(API_URL, headers=headers, json=payload)
#       return response.content
#     image_bytes = query({
#       "inputs": prompt,
#     })

#     import io
#     from PIL import Image
#     image = Image.open(io.BytesIO(image_bytes))
#     try:
#       image.save(path)
#     except:
#         pass

# def stbl_diffusion_1_5(prompt, path):
#     from diffusers import DiffusionPipeline

#     pipe = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
#     pipe = pipe.to("mps")

#     pipe.enable_attention_slicing()

#     _ = pipe(prompt, num_inference_steps=1)

#     image = pipe(prompt).images[0]

#     image.save(path)

# def dalle(prompt, path):
#     # openai.api_key = "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"
#     openai.api_key = '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' # CHEIE Bogdan 2

#     response = openai.Image.create(
#         prompt=prompt,
#         n=1,
#         size="512x512",
#     )

#     img_url = response["data"][0]["url"]
#     img_response = requests.get(img_url)

#     with open(path, "wb") as f:
#         f.write(img_response.content)

# def stbl_diffusion_1_2_base(prompt, path):
#     from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
#     import torch

#     model_id = "stabilityai/stable-diffusion-2-1-base"

#     scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
#     pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float32)
#     pipe = pipe.to("mps")

#     image = pipe(prompt).images[0]  
        
#     image.save(path)

# def vox2(prompt, path):
#     API_URL = "https://api-inference.huggingface.co/models/plasmo/vox2"
#     headers = {"Authorization": "Bearer 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"}

#     def query(payload):
#       response = requests.post(API_URL, headers=headers, json=payload)
#       return response.content
#     image_bytes = query({
#       "inputs": prompt,
#     })
#     # You can access the image with PIL.Image for example
#     import io
#     from PIL import Image
#     image = Image.open(io.BytesIO(image_bytes))
#     try:
#       image.save(path)
#     except:
#       pass

# # openai.api_key = os.getenv("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") CHEIE VECHE

# # openai.api_key = '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' #CHEIE BOGDAN
# # openai.api_key = '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' #CHEIE NOUA
# # openai.api_key = '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' # CHEIE NOUA 2

# openai.api_key = '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' # CHEIE Bogdan 2

# df = pd.read_csv('data/csv/MERGED-from-0-100000-relevant-data-from-websites.csv')

# # input = "ncoilheat.org" 
# # input = "numedical.com.au"
# # input = "kodingakademi.id"
# # input = "a1aautocenter.com"
# # input = "omgfin.com"
# # input = 'aire.es'

# # input = "saturnpower.com"
# input = input("Enter a website url root-domain: ")

# for iter, row in df.iterrows():
#     if row[0] == input:
#         url = row[0]
#         keywords = row[1]
#         meta_description = row[2]
#         page_type = row[3]
#         relevant_paragraphs = row[4]
#         break

# user_prompt = {"COMPANY": url, "KEYWORDS": keywords, "DESCRIPTION": meta_description, "RELEVANT PARAGRAPHS": relevant_paragraphs}
# user_prompt = str(user_prompt)


# with open("data/system-prompt-analyzer.txt", "r") as f:
#     content_analyzer_system = f.read()

# with open("data/system-prompt-creator.txt", "r") as f:
#     content_creator_system = f.read()

# # with open("user-prompt1.txt", "r") as f:
# #     content_role_user = f.read()

# response_analyzer = openai.ChatCompletion.create(
#   # model="gpt-4",
#   model="gpt-3.5-turbo",
#   messages=[
#     {
#       "role": "system",
#       "content": content_analyzer_system
#     },
#     {
#       "role": "user",
#       "content": user_prompt
#     }
#   ],
#   temperature=0.7,
#   max_tokens=500,
#   top_p=1,
#   frequency_penalty=0,
#   presence_penalty=0
# )
# print("________________________ANALYZER__________________________\n")
# print(response_analyzer['choices'][0]['message']['content'])

# #print("\n______________________CE INTRA IN CREATOR_______________\n")

# with open("data/logo-types.txt", "r") as f:
#     types = f.read()

# with open("data/styles.txt", "r") as f:
#     styles = f.read()

# user_intro_str = "You are an expert designer. You job is to take a succinct brief from a client such as this one:"
# user_final_str = "And turn it into a prompt for a hyper intelligent AI that converts text prompts to logos. You should offer 5 different prompts for 5 different ideas of logos. The prompts need to be described as concise as possible, include a logo type and style." + types + styles + "The prompts are not allowed to include any depictions of letters. A good example is the following: 'logo that illustrates a lighthouse (in white and blue) standing firm on a red base, modern, minimalism, vector art, 2d, best quality, centered'"
# #print(user_intro_str + "\n-----------\n" + response_analyzer['choices'][0]['message']['content'] + "\n-----------\n" + user_final_str)

# # print(response_analyzer['choices'][0]['message']['content'])

# response_creator = openai.ChatCompletion.create(
#   # model="gpt-4",
#   model="gpt-3.5-turbo",
#   messages=[
#     {
#       "role": "system",
#       "content": content_creator_system
#     },
#     {
#       "role": "user",
#       "content": user_intro_str + "\n-----------\n" + response_analyzer['choices'][0]['message']['content'] + "\n-----------\n" + user_final_str
#     }
#   ],
#   temperature=0.5,
#   max_tokens=500,
#   top_p=1,
#   frequency_penalty=0,
#   presence_penalty=0
# )

# prompts_split = response_creator['choices'][0]['message']['content'].split("\n\n")
# prompt1 = prompts_split[0][3:].strip()
# prompt2 = prompts_split[1][3:].strip()
# prompt3 = prompts_split[2][3:].strip()
# prompt4 = prompts_split[3][3:].strip()
# prompt5 = prompts_split[4][3:].strip()

# prompts = [prompt1, prompt2, prompt3, prompt4, prompt5]

# print("\n______________________CREATOR PROMPTS___________________________\n")
# print("CREATOR PROMPT 1: " + prompt1)
# print("CREATOR PROMPT 2: " + prompt2)
# print("CREATOR PROMPT 3: " + prompt3)
# print("CREATOR PROMPT 4: " + prompt4)
# print("CREATOR PROMPT 5: " + prompt5)

# logo_number = 1

# output_path = "logos-" + input + "/"

# os.mkdir(output_path)

# for prompt in prompts:
    
#     open_journey(prompt, output_path + "logo" + str(logo_number) + ".png")
#     logo_number += 1
#     logo_diffusinon_checkpoint(prompt, output_path + "logo" + str(logo_number) + ".png")
#     logo_number += 1
#     stbl_diffusion_1_5(prompt, output_path + "logo" + str(logo_number) + ".png")
#     logo_number += 1
#     dalle(prompt, output_path + "logo" + str(logo_number) + ".png")
#     logo_number += 1
#     stbl_diffusion_1_2_base(prompt, output_path + "logo" + str(logo_number) + ".png")
#     logo_number += 1
#     vox2(prompt, output_path + "logo" + str(logo_number) + ".png")
#     logo_number += 1


import os
import openai
import pandas as pd
import requests
from diffusers import StableDiffusionPipeline
import torch
import json
import time

from IPython.display import Image

def open_journey(prompt, path):
    model_id = "prompthero/openjourney"
    pipe = StableDiffusionPipeline.from_pretrained(model_id)
    pipe = pipe.to("mps")
    pipe.enable_attention_slicing()

    image = pipe(prompt).images[0]
    image.save(path)

def logo_diffusinon_checkpoint(prompt, path):
    API_URL = "https://api-inference.huggingface.co/models/logo-wizard/logo-diffusion-checkpoint"
    headers = {"Authorization": "Bearer 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"}

    def query(payload):
      response = requests.post(API_URL, headers=headers, json=payload)
      return response.content
    image_bytes = query({
      "inputs": prompt,
    })

    import io
    from PIL import Image
    image = Image.open(io.BytesIO(image_bytes))
    try:
      image.save(path)
    except:
        pass

def stbl_diffusion_1_5(prompt, path):
    from diffusers import DiffusionPipeline

    pipe = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
    pipe = pipe.to("mps")

    pipe.enable_attention_slicing()

    _ = pipe(prompt, num_inference_steps=1)

    image = pipe(prompt).images[0]

    image.save(path)

def dalle(prompt, path):
    openai.api_key = "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"

    response = openai.Image.create(
        prompt=prompt,
        n=1,
        size="512x512",
    )

    img_url = response["data"][0]["url"]
    img_response = requests.get(img_url)

    with open(path, "wb") as f:
        f.write(img_response.content)

def stbl_diffusion_1_2_base(prompt, path):
    from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
    import torch

    model_id = "stabilityai/stable-diffusion-2-1-base"

    scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
    pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float32)
    pipe = pipe.to("mps")

    image = pipe(prompt).images[0]  
        
    image.save(path)

def vox2(prompt, path):
    API_URL = "https://api-inference.huggingface.co/models/plasmo/vox2"
    headers = {"Authorization": "Bearer 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"}

    def query(payload):
      response = requests.post(API_URL, headers=headers, json=payload)
      return response.content
    image_bytes = query({
      "inputs": prompt,
    })
    # You can access the image with PIL.Image for example
    import io
    from PIL import Image
    image = Image.open(io.BytesIO(image_bytes))
    try:
      image.save(path)
    except:
      pass

def requesttt(prompt, key):

    url = "https://api.midjourneyapi.io/v2/imagine"

    payload = {"prompt": prompt}
    headers = {
    'Authorization': key
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    task_id = response.text[11:]
    task_id = task_id[:-2]  

    print("Task ID is: " + response.text)
    print("Task ID is: " + task_id)

    return task_id

def responseee(task, key, output):

    url = "https://api.midjourneyapi.io/v2/result"
    payload = {"taskId": task}

    headers = {
    'Authorization': key
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    json_response = json.loads(response.text)
    print(response.text)
    link = json_response.get('imageURL', '')
    print(link)

    if link != '':
        img_response = requests.get(link)

        with open(output, "wb") as f:
          f.write(img_response.content)
    return link

def split_image(image_path, output_dir, index):
    # Open the image file
    img = Image.open(image_path)
    # Calculate dimensions of each quarter
    width, height = img.size
    quarter_width = width // 2
    quarter_height = height // 2
    idx = 1
    # Split the image
    img1 = img.crop((0, 0, quarter_width, quarter_height))
    img2 = img.crop((quarter_width, 0, width, quarter_height))
    img3 = img.crop((0, quarter_height, quarter_width, height))
    img4 = img.crop((quarter_width, quarter_height, width, height))
    # Save the images
    img1.save(os.path.join(output_dir, str(index) + 'img' + str(idx) +'.png'))
    index += 1
    img2.save(os.path.join(output_dir, str(index) + 'img' + str(idx) + '.png'))
    index +=1
    img3.save(os.path.join(output_dir, str(index) + 'img' + str(idx) + '.png'))
    index +=1
    img4.save(os.path.join(output_dir, str(index) + 'img' + str(idx) + '.png'))
    idx +=1

# openai.api_key = os.getenv("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") CHEIE VECHE

# openai.api_key = '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' #CHEIE BOGDAN
# openai.api_key = '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' #CHEIE NOUA
# openai.api_key = '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' # CHEIE NOUA 2
mid_journey_key = 'af3416f8-9e61-4705-8fa2-8468bd0f762e'
openai.api_key = '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' # CHEIE Bogdan 2

df = pd.read_csv('data/csv/MERGED-from-0-100000-relevant-data-from-websites.csv')

# input = "ncoilheat.org" 
# input = "numedical.com.au"
# input = "kodingakademi.id"
# input = "a1aautocenter.com"
# input = "omgfin.com"
# input = 'aire.es'
# input = "saturnpower.com"
# input = "ocaseys.nl"

# output_path = "logos10/"

input = input("Enter a website url root-domain: ")
print("\n")
# output_path = "logos-" + input + "/"
# output_path = input("Enter a path to a folder to save the logos: ")

for iter, row in df.iterrows():
    if row[0] == input:
        url = row[0]
        keywords = row[1]
        meta_description = row[2]
        page_type = row[3]
        relevant_paragraphs = row[4]
        break

user_prompt = {"COMPANY": url, "KEYWORDS": keywords, "DESCRIPTION": meta_description, "RELEVANT PARAGRAPHS": relevant_paragraphs}
user_prompt = str(user_prompt)


with open("data/system-prompt-analyzer.txt", "r") as f:
    content_analyzer_system = f.read()

with open("data/system-prompt-creator.txt", "r") as f:
    content_creator_system = f.read()

# with open("user-prompt1.txt", "r") as f:
#     content_role_user = f.read()

response_analyzer = openai.ChatCompletion.create(
  # model="gpt-4",
  model="gpt-3.5-turbo",
  messages=[
    {
      "role": "system",
      "content": content_analyzer_system
    },
    {
      "role": "user",
      "content": user_prompt
    }
  ],
  temperature=0.7,
  max_tokens=500,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0
)
print("________________________ANALYZER__________________________\n")
print(response_analyzer['choices'][0]['message']['content'])

#print("\n______________________CE INTRA IN CREATOR_______________\n")

with open("data/logo-types.txt", "r") as f:
    types = f.read()

with open("data/styles.txt", "r") as f:
    styles = f.read()

user_intro_str = "You are an expert designer. You job is to take a succinct brief from a client such as this one:"
user_final_str = "And turn it into a prompt for a hyper intelligent AI that converts text prompts to logos. You should offer 5 different prompts for 5 different ideas of logos. The prompts need to be described as concise as possible, include a logo type and style." + types + styles + "The prompts are not allowed to include any depictions of letters. A good example is the following: 'logo that illustrates a lighthouse (in white and blue) standing firm on a red base, modern, minimalism, vector art, 2d, best quality, centered'"
#print(user_intro_str + "\n-----------\n" + response_analyzer['choices'][0]['message']['content'] + "\n-----------\n" + user_final_str)

# print(response_analyzer['choices'][0]['message']['content'])

response_creator = openai.ChatCompletion.create(
  # model="gpt-4",
  model="gpt-3.5-turbo",
  messages=[
    {
      "role": "system",
      "content": content_creator_system
    },
    {
      "role": "user",
      "content": user_intro_str + "\n-----------\n" + response_analyzer['choices'][0]['message']['content'] + "\n-----------\n" + user_final_str
    }
  ],
  temperature=0.5,
  max_tokens=500,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0
)

prompts_split = response_creator['choices'][0]['message']['content'].split("\n\n")
prompt1 = prompts_split[0][3:].strip()
prompt2 = prompts_split[1][3:].strip()
prompt3 = prompts_split[2][3:].strip()
prompt4 = prompts_split[3][3:].strip()
prompt5 = prompts_split[4][3:].strip()

prompts = [prompt1, prompt2, prompt3, prompt4, prompt5]

print("\n______________________CREATOR PROMPTS___________________________\n")
print("CREATOR PROMPT 1: " + prompt1)
print("CREATOR PROMPT 2: " + prompt2)
print("CREATOR PROMPT 3: " + prompt3)
print("CREATOR PROMPT 4: " + prompt4)
print("CREATOR PROMPT 5: " + prompt5)

logo_number = 1

# output_path = "logos10/"
output_path = "logos-" + input + "/"

os.mkdir(output_path)

for prompt in prompts:
    
    # # open_journey(prompt, output_path + "logo" + str(logo_number) + ".png")
    # logo_number += 1
    # try:
    #   logo_diffusinon_checkpoint(prompt, output_path + "logo" + str(logo_number) + ".png")
    # except:
    #    pass
    # logo_number += 1
    # stbl_diffusion_1_5(prompt, output_path + "logo" + str(logo_number) + ".png")
    # logo_number += 1
    # dalle(prompt, output_path + "logo" + str(logo_number) + ".png")
    # logo_number += 1
    # stbl_diffusion_1_2_base(prompt, output_path + "logo" + str(logo_number) + ".png")
    # logo_number += 1
    # try:
    #   vox2(prompt, output_path + "logo" + str(logo_number) + ".png")
    # except:
    #   pass
    # logo_number += 1
    task = requesttt(prompt, mid_journey_key)
    for i in range(100):
      time.sleep(10)
      main_output = output_path + "logo" + str(logo_number) + ".png"
      link = responseee(task, mid_journey_key, output = main_output)

      split_image = split_image(main_output, output_path, index = logo_number)
      if link != '':
          break
    logo_number += 1  