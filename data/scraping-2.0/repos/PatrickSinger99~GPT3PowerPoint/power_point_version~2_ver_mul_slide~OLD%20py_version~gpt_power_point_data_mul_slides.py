import openai
import gpt_power_point_creator_mul_slides
import wikipediaapi
import summarization_training
from google_images_download import google_images_download
import os
import shutil

max_len = 150  # Max length of the Bullet Points
max_slide_num = 5

# Set wikipedia language
wiki_wiki = wikipediaapi.Wikipedia(language='en', extract_format=wikipediaapi.ExtractFormat.WIKI)

# Get topic
while True:
    # Inputs
    prompt = input("Topic: ")

    # Define the wikipedia page
    p_wiki = wiki_wiki.page(prompt)

    # dictionary of the wikipedia page
    def create_dict(page):
        dict = {}
        dict[page.title] = page.summary
        for s in page.sections:
            dict[s.title] = s.text

            try:
                if s.text == "":
                    dict[s.title] = s.sections[0].text
            except:
                pass
        return dict

    wiki_dict = create_dict(p_wiki)
    try:
        wiki_dict.pop("See also")
    except:
        pass
    try:
        wiki_dict.pop("References")
    except:
        pass
    try:
        wiki_dict.pop("External links")
    except:
        pass
    try:
        wiki_dict.pop("Further reading")
    except:
        pass
    try:
        wiki_dict.pop("Notes")
    except:
        pass

    new_wiki_dict = {}
    slide_num = 0
    for key in wiki_dict:
        slide_num += 1
        if slide_num <= max_slide_num:
            new_wiki_dict[key] = wiki_dict[key]

    # Print Subtopics
    print("\nFound Subtopics: ")
    for key in new_wiki_dict:
        print(key)
    print("\n")

    verify = input("Do you want a power-point for these topics? (y/n): ")
    if verify.lower() == "y":
        break


# Openai key
with open("openai_key.txt") as file:
    key = file.read()
    openai.api_key = key

gpt_sum = summarization_training.create_sum_model()

dict_for_pptx = {}

for key in new_wiki_dict:
    # Get GPT output
    output = gpt_sum.submit_request(new_wiki_dict[key])
    output = output.choices[0].text[8:]

    # Crop if too long
    if len(output) > max_len:
        output = output[:max_len]
        to_cut = ""
        for i in reversed(range(1, len(output))):
            if output[i] == ".":
                output = output.replace(to_cut, "")
                break
            to_cut = output[i] + to_cut

    # Print Points
    print("\nSummarized points for " + key + ":")
    for sentence in output.split(". "):
        print("    - " + sentence)

    dict_for_pptx[key] = output

# Delete download folder if it exists
try:
    shutil.rmtree('downloads')
except FileNotFoundError:
    pass

# Download an image for the keyword
response = google_images_download.googleimagesdownload()
arguments = {"keywords": prompt, "limit": max_slide_num+2, "print_urls": True, format: "jpg"}
response.download(arguments)


img_list = os.listdir("downloads/" + prompt)
img_dict = {}

counter = 0
try:
    for key in dict_for_pptx:
        img_dict[key] = "downloads/" + prompt + "//" + img_list[counter]
        counter += 1
except:
    pass


gpt_power_point_creator_mul_slides.create_power_point_slides_from_gpt(dict_for_pptx, img_dict, prompt)
