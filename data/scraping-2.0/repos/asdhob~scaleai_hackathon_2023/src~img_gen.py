import csv
import requests
import os
import openai
csv_data = '''Slide number,Slide title,Number of minutes to spend,High level bullet points for the slide
1,Introduction to Newton's Laws of Motion,2,Overview of Newton's Laws of Motion;Description of the three laws;Significance of the laws
2,Newton's First Law,3,Definition of Newton's First Law;Examples of Newton's First Law;Applications of Newton's First Law
3,Newton's Second Law,3,Definition of Newton's Second Law;Examples of Newton's Second Law;Applications of Newton's Second Law
4,Newton's Third Law,3,Definition of Newton's Third Law;Examples of Newton's Third Law;Applications of Newton's Third Law
5,Conclusion,2,Summary of Newton's Laws;Significance of Newton's Laws;Questions and Answers'''
slide_titles = []  # Empty list to store the slide titles
slides_prompt = []  # Empty list to store the responses
reader = csv.reader(csv_data.splitlines())
next(reader)  # Skip the header row
for row in reader:
    slide_title = row[1]
    slide_titles.append(slide_title)
print(slide_titles)
headers = {"Authorization": "Basic clk4dftv005t91ass5ncn78w5"}
for item in slide_titles:
    data = {
        "input": {
            "input": item
        }
    }
    response = requests.post(
        "https://dashboard.scale.com/spellbook/api/v2/deploy/ll13d4p",
        json=data,
        headers=headers
    )
    slides_prompt.append(response.json()["output"])
print(slides_prompt)
openai.api_key = "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"
slides_image = []  # Array to store the image URLs
for prompt in slides_prompt:
    response = openai.Image.create(
        prompt=prompt,
        n=1,
        size="1024x1024"
    )
    slides_image.append(response.data[0].url)
print(slides_image)