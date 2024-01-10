import openai
from google_images_download import google_images_download
import gpt_power_point_creator
import os
import shutil
from gpt import GPT
import openai
from gpt import Example

# Max length of the Bullet Points
max_len = 150

prompt = input("Topic: ")


# Openai key
with open("openai_key.txt") as file:
    key = file.read()
    openai.api_key = key

gpt_point_creation = GPT(engine="davinci", temperature=.5, max_tokens=120)

gpt_point_creation.add_example(Example("Napoleon III",
                                       "Napoleon III was the first President of France. "
                                       "He founded the Second Empire, reigning until the defeat. "
                                       "He made the French merchant navy the second largest in the world."
                                       ))

gpt_point_creation.add_example(Example("mitochondrion",
                                       "A mitochondrion is a double-membrane-bound organelle. "
                                       "Mitochondria generate most of the cell's supply of adenosine triphosphate. "
                                       "The mitochondrion is often called the powerhouse of the cell."
                                       ))


gpt_point_creation.add_example(Example("blockchain",
                                       "A blockchain is a list of blocks, that are linked together. "
                                       "Blockchains are resistant to modification of their data. "
                                       "The data in any given block cannot be altered once recorded."
                                       ))

gpt_point_creation.add_example(Example("germany",
                                       "Germany is a country in Central Europe. "
                                       "A region named Germania was documented before AD 100. In the 10th century. "
                                       "It covers an area of 357,022 square kilometres. "
                                       "Germany has a population of over 83 million within its 16 constituent states."
                                       ))

# Get GPT output
output = gpt_point_creation.submit_request(prompt)
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


keyword = prompt

# Delete download folder if it exists
try:
    shutil.rmtree('downloads')
except FileNotFoundError:
    pass


# Download an image to the keyword
response = google_images_download.googleimagesdownload()
arguments = {"keywords": keyword, "limit": 1, "print_urls": True, format: "jpg"}
response.download(arguments)

# Create path to image
pic_path = "downloads/" + keyword + "/" + os.listdir("downloads/" + keyword)[0]

gpt_power_point_creator.create_power_point_from_gpt(keyword, output, pic_path)
