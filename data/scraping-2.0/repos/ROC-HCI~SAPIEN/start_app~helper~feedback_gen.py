# Code authors: Masum Hasan, Cengiz Ozel, Sammy Potter
# ROC-HCI Lab, University of Rochester
# Copyright (c) 2023 University of Rochester

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN 
# THE SOFTWARE.


import openai
from openai import AzureOpenAI

client = AzureOpenAI(api_version=os.environ["api_version"],
api_key=os.environ["azure_openai_key"])
import random
import os
# from .keys import *
# openai.api_key = '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' # GPT-4 key for Cengiz's account, let's not use it right now.
# openai.api_key = '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' # GPT-3 key for Masum's account

# TODO: The 'openai.api_base' option isn't read in the client API. You will need to pass it when you instantiate the client, e.g. 'OpenAI(api_base=os.environ["api_base"])'
# openai.api_base = os.environ["api_base"]



def create_prompt(tags, meeting):
    transcript = meeting.get_transcript()
    if not meeting.goal:
        goal_prompt = f"What was the goal of {meeting.user.firstname} in the conversation? If the goal is unclear, say \"Undefined\". Don't say anything else.\n\n---\n{transcript}\n---\n"
        meeting.goal = ask_gpt(goal_prompt)
    prompt = f"{meeting.user.firstname} would like to achieve the following goal: {meeting.goal} \n\nPlease provide feedback to {meeting.user.firstname} on the conversation using the following {len(tags)} metrics:\n\n"
    for i, tag in enumerate(tags):
        prompt += f"{i+1}. {tag}\n"
    prompt += f"\nDo not provide feedback for {meeting.bot.firstname}\n\n"

    prompt += f"Conversation Transcript:\n{transcript}\n\n"
    
    prompt += "Please write your feedback below, using the metrics above to guide your evaluation:\n"

    return prompt

def generate_feedback(tags, meeting):
    # transcript = meeting.get_transcript()
    if tags[0] == "Insufficient meeting information." or not meeting or not meeting.get_transcript():
        return "Insufficient meeting information."
    
    prompt = [{"role": "user", "content": create_prompt(tags, meeting)}]

    try:
        response = client.chat.completions.create(model='Azure-ChatGPT',
        max_tokens=500,
        temperature=0.6,
        messages = prompt)

        response_text =  response.choices[0].message.content
    except:
        response_text = "Error. Please try again."

    return response_text

def ask_gpt(text):
    try:
        prompt = [{"role": "user", "content": text}]
        response = client.chat.completions.create(model='Azure-ChatGPT',
        max_tokens=100,
        temperature=0.6,
        messages=prompt)
        response_text = response['choices'][0]['message']['content']
        print(f"response: {response_text}")
    except:
        response_text = "None"
    return response_text

def get_feedback_keypoints(meeting):
    if not meeting or not meeting.get_transcript():
        return ["Insufficient meeting information."]
    premise = meeting.premise
    transcript = meeting.get_transcript()
    goal = meeting.goal
    if not premise:
        premise_prompt = f"What was the premise of the conversation? If the premise is not clear, say [Unclear] and don't say anything else.\n\n---\n{transcript}\n---\n"
        premise = ask_gpt(premise_prompt)
    if not goal:
        goal_prompt = f"What was the goal of {meeting.user.firstname} in the conversation? If the goal is not clear, say [Unclear] and don't say anything else.\n\n---\n{transcript}\n---\n"
        goal = ask_gpt(goal_prompt)
        meeting.goal = goal
    
    keypoints = ["Clear", "Engaging", "Concise", "Confident", "Empathetic", "Inspiring", "Inclusive", "Respectful", "Honest", "Open-minded", "Patient", "Positive", "Supportive", "Trustworthy", "Understanding"]

    if "[unclear]" in premise.lower() and "[unclear]" in goal.lower():
        keypoints = random.sample(keypoints, 3)
        return keypoints

    keypoints_prompt = f"{meeting.user.firstname} would like feedback on their conversation skill in the following premise and goal.\nPremise: {premise}\nGoal: {goal} \n\nWhat are the 3 most important traits of communication and conversation one should have in that premise? Don't say anything else. Do not list them as bullets. Only list the 3 traits separated by semicolon. Make it brief, each keypoint one or two words."

    max_attempt = 5
    keypoints_found = False

    while max_attempt > 0 and not keypoints_found:
        max_attempt -= 1
        print(f"Max attempt: {max_attempt}")
        response = ask_gpt(keypoints_prompt)
        if response[-1] == ".":
            response = response[:-1]
        keypoints = response.split(";")
        keypoints = [keypoint.strip() for keypoint in keypoints]

        if "language model" in response.lower():
            print("Language model detected")
            keypoints_found = False
            continue
        if len(response) > 100:
            print("Response too long")
            keypoints_found = False
            continue
        if "\n" in response:
            print("Newline detected")
            keypoints_found = False
            continue
        # Check if all keypoints are unique
        if len(keypoints) != len(set(keypoints)):
            print("Duplicate keypoints")
            keypoints_found = False
            continue
        # Check if 3 keypoints are found 
        if len(keypoints) != 3:
            print(len(keypoints), "keypoints found")
            keypoints_found = False
            continue
        # Check if all keypoints are not empty
        if all(keypoint == "" for keypoint in keypoints):
            print("Empty keypoints")
            keypoints_found = False
            continue
        # check if all keypoints are not too long
        if any(len(keypoint) > 100 for keypoint in keypoints):
            print("Keypoint too long")
            keypoints_found = False
            continue
        keypoints_found = True
        if keypoints_found:
            print("Keypoints found", keypoints)
            return keypoints

    keypoints = random.sample(keypoints, 3)
    return keypoints
