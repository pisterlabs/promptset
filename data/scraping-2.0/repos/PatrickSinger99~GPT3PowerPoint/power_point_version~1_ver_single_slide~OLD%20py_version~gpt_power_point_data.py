import openai
from google_images_download import google_images_download
import gpt_power_point_creator
import os
import shutil
import wikipediaapi
import summarization_training

# Max length of the Bullet Points
max_len = 150

# Set wikipedia language
wiki_wiki = wikipediaapi.Wikipedia(language='en', extract_format=wikipediaapi.ExtractFormat.WIKI)

# Get topic
while True:
    # Inputs
    prompt = input("Topic: ")

    # Define the wikipedia page
    p_wiki = wiki_wiki.page(prompt)

    # Summary of the wikipedia page
    wiki_summary = p_wiki.summary
    wiki_summary = wiki_summary.replace("\n", " ")
    print(wiki_summary)
    verify = input("Do you want to summarize this text? (y/n): ")
    if verify.lower() == "y":
        break

# Openai key
with open("openai_key.txt") as file:
    key = file.read()
    openai.api_key = key

gpt_sum = summarization_training.create_sum_model()

# Get GPT output
output = gpt_sum.submit_request(wiki_summary)
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
print("\nSummarized points:")
for sentence in output.split(". "):
    print("    - " + sentence)


# Getting the keywords
keyword_response = openai.Completion.create(engine="davinci", prompt="Text: " + output + "Keywords:", temperature=0.3, max_tokens=60, top_p=1.0,
                                            frequency_penalty=0.8, presence_penalty=0.0, stop=["\n"])


# convert enumeration into list
keywords = keyword_response.choices[0].text
keywords = keywords.split(",")

print("\n Extracted keywords:")
for keyword in keywords:
    print("    - " + keyword)

# Delete download folder if it exists
try:
    shutil.rmtree('downloads')
except FileNotFoundError:
    pass

# Download an image to every keyword (currently just the first one)
for i in keywords[:1]:
    response = google_images_download.googleimagesdownload()
    arguments = {"keywords": i, "limit": 1, "print_urls": True, format: "jpg"}
    response.download(arguments)

# Create path to image
pic_path = "downloads/" + keywords[0] + "/" + os.listdir("downloads/" + keywords[0])[0]

gpt_power_point_creator.create_power_point_from_gpt(keywords[0], output, pic_path)
