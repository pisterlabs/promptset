#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  9 16:07:46 2022

@author: sohanjs, Mahdjoubi Bilal
"""

import json
import os
import re
import reprlib

import openai
# Import the libraries
import requests
from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.

API_KEY_CORE_API = os.environ.get('CORE_API_KEY')
API_ENDPOINT_CORE_API = "https://api.core.ac.uk/v3"

# Set initial Parameters
# Enter your OpenAI API keys to run GPT-3 model
# Remember to authorize the key before using it.
openai.api_key = os.environ.get('OPEN_AI_API_KEY')

# Character_limit is set in order to avoid the maxing token request
CHARACTER_LIMIT = 3000

# how many number of pdf downloads are needed ?
NUMBER_OF_PDF_DOWNLOADS = 3

MAX_NUMBER_OF_CHARACTERS_IN_PAPERS = 100000


def pretty_json(obj):
    return json.dumps(obj, indent=4)


# ---------------------- PART 1 ----------------------
# This function receive a topic and search for the papers in the topic
# It will download the papers and save them in the folder named "papers"
# The files will be downloaded for the CORE API that is an API that is used to get research papers


def get_papers(topic):
    results, elapsed = query_core_api("/search/works", topic)
    print(f"The search took {elapsed} seconds")
    print(f"The results are: ")
    print(reprlib.repr(results))
    papersTitle = []
    papersText = []
    papersUrl = []

    for result in results["results"]:
        print(len(result["fullText"]))
        if "fullText" in result is not None and len(result["fullText"]) <= MAX_NUMBER_OF_CHARACTERS_IN_PAPERS:
            # filename = generate_filename(result["title"])
            # download_success = download_pdf(result["downloadUrl"], filename)
            # if download_success:
            papersTitle.append(result["title"])
            papersUrl.append(result["downloadUrl"])
            papersText.append(result["fullText"])

    return papersTitle, papersText, papersUrl


def query_core_api(url_fragment, query, limit=NUMBER_OF_PDF_DOWNLOADS):
    headers = {"Authorization": "Bearer " + API_KEY_CORE_API}
    query = {"q": query, "limit": limit}
    response = requests.post(f"{API_ENDPOINT_CORE_API}{url_fragment}", data=json.dumps(query), headers=headers)
    if response.status_code == 200:
        return response.json(), response.elapsed.total_seconds()
    else:
        print(f"Error code {response.status_code}, {response.content}")
        handle_error(response.status_code)


# ---------------------- PART 2 ----------------------


# Shows the Paper Summary from GPT-3
def getPaperSummary(paperContent):
    tldr_tag = "\n Tl;dr"
    text = ""
    textBegin = None

    print("The paper content is:")
    print(reprlib.repr(paperContent))

    # For loop to read all the text from the array paperContent
    for page in paperContent:
        text += page

    try:
        textBegin = re.search("^[\s\S]*?(?=INTRODUCTION|INTRODUCTIONS)", text).group()
    except Exception as e:
        print("No introduction found")
        print(e)

    textEnd = text[-CHARACTER_LIMIT:]

    if textBegin is not None and textEnd is not None:
        text = textBegin + textEnd
    else:
        textBegin = text[0:CHARACTER_LIMIT]
        textEnd = text[-CHARACTER_LIMIT:]
        text = textBegin + textEnd
    if text is not None:
        text = cut(text)
        text += tldr_tag
    print("The AI will summarize the text below:", text)
    basePrompt = "Write me a summary of the following research paper:"
    try:
        response = openai.Completion.create(model="text-davinci-003",
                                            prompt=basePrompt + text,
                                            temperature=0,
                                            max_tokens=300,
                                            top_p=1,
                                            frequency_penalty=0,
                                            presence_penalty=0
                                            )
    except Exception as e:
        print("Exception thrown : " + e.__str__())
        return "The engine was not able to load, probably due to an overload of requests"

    print("The response is:")
    print(response["choices"][0]["text"])
    return response["choices"][0]["text"]


def cut(text):
    # Make sure the numbers of characters in text is under or equals the limited character
    if len(text) <= CHARACTER_LIMIT:
        # If it is, then we return the text
        return text
    else:
        # If it is not, then we cut the text and return the text
        return text[:CHARACTER_LIMIT]


def getSummariesForTopic(topic):
    # Get the papers from the topic
    papersTitle, papersTexts, papersUrl = get_papers(topic)
    print("The papers title are:")
    print(papersTitle)
    print("The papers text are:")
    print(papersTexts)
    print("The papers url are:")
    print(papersUrl)

    summaries = []
    for paper in papersTexts:
        summaryOfPaper = getPaperSummary(paper)
        # Cut the first 2 characters in the string summaryOfPaper
        summaryOfPaper = summaryOfPaper[2:]
        summaries.append(summaryOfPaper)

    # Merge the papersInfo array and the summaries array into one key value pair
    papersInfoAndSummaries = []
    for i in range(len(papersTitle) & len(summaries) & len(papersUrl)):
        papersInfoAndSummaries.append({"Title": papersTitle[i], "Url": papersUrl[i], "summary": summaries[i]})

    print("The papers info and summaries are:", papersInfoAndSummaries)
    # Transform the papersInfoAndSummaries array into a json for html return
    return papersInfoAndSummaries


def getSummariesForFile(fileContent):
    papersInfoAndSummaries = [{"Title": "Your paper", "Url": "", "summary": getPaperSummary(fileContent)}]
    return papersInfoAndSummaries


def handle_error(status_code):
    pass


def store_mail(mail):
    # Open the file in append mode to ensure that the mail is added to the end
    with open("mails.txt", "a") as file:
        # print the absolute path of the file
        print("Absolute path of the file: ", os.path.abspath("mails.txt"))
        # Write the mail to the file followed by a newline character
        file.write(mail + "\n")
        file.close()
    print("Mail stored successfully!")
