"""
GPT4 Contextual Diffusion

Module for LLM(GPT-4) Calls.

Copyright (c) 2023 Taehoon Hwang.
Licensed under the MIT License (see LICENSE for details)
Written by Taehoon Hwang
"""

import os
import json
import openai
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GPT4_APIKEY")
BASE_PROMPT = """
              I detected {} objects in my image generated with Stable Diffusion.
              The original prompt was {}, and the objects detected were {}.
              I need you to generate a creative story around this original prompt and these detected objects for this generated image.
              You are allowed to describe new themes, new backgrounds, etc, be as creative as you can.
              However, limit yourself to 1 paragraph to describe it.
              After you do so, with this story, you need to generate additional Stable Diffusion prompts for each detected object, following the rules below.
              The prompts should be formatted such to be generation tags splitted with commas like the  original prompt.
              DO NOT generate prompts with full natural text. The goal is to make Stable Diffusion understand the prompt, and the current model cannot process natural language that easily.
              Limit yourself to 10 tags per object prompt.
              Do not include the original prompt in the generated tags, ONLY include direct tags describing the object.
              The goal is to use your prompts for additional context to inpaint the existing image with more detail.
              Format your response as such: Describe your story at the top, with the text "Story: " in front of it.
              Following that, describe each of your new prompts for each object, with the text "[OBJECT NAME]: " in front of it.
              Replace the [OBJECT NAME] part with the actual name of the object given.
              """

def generate_prompt(detectedClasses, userPrompt):
    return BASE_PROMPT.format(len(detectedClasses), "'" + userPrompt + "'", "'" + ','.join(detectedClasses) + "'")

# Calls GPT-4 with the appropriate prompts, combined with the original user prompt and detected classes within the image.
# Returns the subprompts GPT-4 generated for each class.
def call_gpt4(detectedClasses, userPrompt):
    openai.api_key = API_KEY

    promptGPT = generate_prompt(detectedClasses, userPrompt)

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that generates context for the user's Stable Diffusion model with tag based prompts."
            },
            {
                "role": "user",
                "content": promptGPT
            },
        ],
        temperature=0.7
    )

    contentStr = response.choices[0].message.content
    contentListDoubleNewLine = contentStr.split('\n\n')
    contentList = []
    for item in contentListDoubleNewLine:
        contentList.extend(item.split('\n'))
    
    elementsTagsDict = {}

    for line in contentList:
        try:
            element, tags = line.split(': ')
        except:
            continue
        
        elementsTagsDict[element] = tags

    return elementsTagsDict

# print(call_gpt4(["Sky", "Buildings", "Alleyshops", "Jacket", "Bag", "Shoes"], "1girl, full body, background"))