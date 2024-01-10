from datetime import datetime
from openai import OpenAI
from colorama import Fore
import numpy as np
import glob
import os

# SYSTEM FUNCTIONS
    
def get_most_recent_file(pattern):
    # Get a list of all files that match the pattern
    files = glob.glob(pattern)

    # Find the most recent file
    most_recent_file = max(files, key=os.path.getctime)

    return most_recent_file

def merge_files(file1, file2, output_file_prefix):
    # Check if both files are accessible and not empty
    if not os.path.isfile(file1) or os.path.getsize(file1) == 0:
        print(f"File {file1} is not accessible or is empty.")
        return
    if not os.path.isfile(file2) or os.path.getsize(file2) == 0:
        print(f"File {file2} is not accessible or is empty.")
        return

    with open(file1, 'r') as f1, open(file2, 'r') as f2:
        lines1 = f1.readlines()
        lines2 = f2.readlines()

    unique_lines = list(set(lines1 + lines2))

    # Get current date and time
    now = datetime.now()

    # Format as a string suitable for a filename
    filename_date = now.strftime("%Y-%m-%d_%H-%M-%S")

    # Create output filename
    output_file = f'{output_file_prefix}-{filename_date}.txt'

    with open(output_file, 'w') as output:
        for line in unique_lines:
            output.write(line)

# Read API key from a file
with open('secret_apikey.txt', 'r') as file:
    api_key = file.read().strip()

client = OpenAI(api_key=api_key)


# specify the OpenAI model to use:
# https://platform.openai.com/docs/models/gpt-3-5
# 16k token limit on "gpt-3.5-turbo-16k-0613"
# 16k token limit on "gpt-3.5-turbo-1106" (newest release as of dec-1-2023)
gpt_model = "gpt-3.5-turbo-1106"

RESEARCH_TOPIC      = "We are doing a deepdive into the 5 states of emotional control words can exhibit. The states are: Very Negative, Negative, Neutral, Positive, Very Positive."

PRIMARY_FOCUS       = "Words that feel emotionally Very Positive. Not simply Positive - but exceptionally so, a 5/5 on our scale. We are looking for words (or sentences of up to three words) that are Very emotionally positive or exhibit that kind of vibe."

CURRENT_TASK        = "We are looking to get at least 2000 words that are Very Strictly related to {PRIMARY_FOCUS}. Very important: words strictly related to {PRIMARY_FOCUS}. We also want to get the words in both singular and plural form, so a word like 'cat' would be 'cat' and 'cats' on separate lines. {DATA_PRESENTATION}."

PRIMARY_GOAL        = f"""Primary Goal: find, gather, research and explore everything available to you to find English words that are Very Strictly related to {PRIMARY_FOCUS}.
                          We are going to need at least 2000 words Very Strictly related to {PRIMARY_FOCUS}."""

SECONDARY_GOAL      = f"""Once we find words, we format them in a single word per line but both in singular and plural form, so a word like 'cat' would be 'cat' and 'cats' on separate lines.
                          IMPORTANT: do not add any other words than the words that are Very Strictly related to {PRIMARY_FOCUS}. I want you to consider each word carefully and only add the words that are in practice guaranteed to be related to {PRIMARY_FOCUS}."""

DATA_PRESENTATION   = f"You write file in ASCII: every single word or up to three word sentence on its own line. No line numbers, no indexes, no capital letters, ONLY a single word in ASCII, or a sentence of up to three words in ASCII per line"

sme1_specialization = "English language professor and doctor in psychology"
sme1_task           = f"""You are a {sme1_specialization}. you are one of the worlds top experts in psychology and communication. Your job, {RESEARCH_TOPIC}. Find words that are Very Strictly SPECIFICALLY emotionally related to ."""

sme2_specialization = "Thought-leader in language and psychology"
sme2_task           = f"""You are a {sme2_specialization}. Your focus is {PRIMARY_FOCUS} - always try to get as many and as accurate as possible words - do not forget to pluralize or the other
                        way around and to help us find words that are Very Strictly related to {PRIMARY_FOCUS}."""


user_request        = f"""
                      Always consider the goals. We need to make singularized and pluralized words as well as matching the 'vibe' of the word to the {PRIMARY_FOCUS}. This is going to be
                      a challenge but I believe that with enough iterations we can probably be able to find at least 2000 words for each of these 5 areas of focus for this research."""


# If i by chance happen to figure out a really nice way to set up a working team of resarchers,
# I just want to write here that this is stricly for fun and games, a learning experience, and
# I never went to school so I have no normal training and I have no idea who they usually do these things.
# I am just trying to figure out how to do it in a way that is fun and interesting and that I can learn from.
# one love :) 



###############################################
################# SME1 ########################
stream = client.chat.completions.create(
    model=gpt_model,
    messages=[
        {"role": "system", "content": RESEARCH_TOPIC + PRIMARY_FOCUS + PRIMARY_GOAL + SECONDARY_GOAL + DATA_PRESENTATION,
         "role": "assistant", "content": f"Your focus is on {PRIMARY_FOCUS}, {DATA_PRESENTATION}, and providing at least 2000 words that are Very Strictly related to {PRIMARY_FOCUS}.",
         "role": "user", "content": user_request + f"(IMPORTANT) - everytime you think of a word that is related to {RESEARCH_TOPIC}, please add it to the list both as plural and singular on separate lines."
         #"role": "assistant", "content": goal + agenda_for_the_moment
    }], 
    stream=True)
# Create a list to store the strings
sme1_output = []

for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")
        # Add the string to the list
        sme1_output.append(chunk.choices[0].delta.content)
        # Join the strings together to get the final result
        sme1_results = "".join(sme1_output)
with open('gpt3.5-dual-agent-sme1.txt', 'w') as file:
    file.write(sme1_results)


previous_research_material = sme1_results

###############################################
################# SME2 ########################
stream = client.chat.completions.create(
    model=gpt_model,
    messages=[
        {"role": "system", "content": RESEARCH_TOPIC + PRIMARY_FOCUS + PRIMARY_GOAL + SECONDARY_GOAL + DATA_PRESENTATION,
         "role": "assistant", "content": f"I will build upon {previous_research_material} and my focus is on {PRIMARY_FOCUS} and continuing to get more data.",
         "role": "user", "content": user_request + f"(IMPORTANT) - everytime you think of a word that is related to {RESEARCH_TOPIC}, please think twice or even three times about it and if you are sure it is Very Strictly related to {PRIMARY_FOCUS}",
         "role": "assistant", "content": previous_research_material + PRIMARY_FOCUS + CURRENT_TASK
    }],
    stream=True)
# Create a list to store the strings
sme2_output = []

for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")
        # Add the string to the list
        sme2_output.append(chunk.choices[0].delta.content)
        # Join the strings together to get the final result
        sme2_results = "".join(sme2_output)
with open('gpt3.5-dual-agent-sme2.txt', 'w') as file:
    file.write(sme2_results)


# Example usage:
merge_files('gpt3.5-dual-agent-sme1.txt', 'gpt3.5-dual-agent-sme2.txt', 'RESEARCH-LAB-DATA')























