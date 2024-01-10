#!/usr/bin/env python
# coding: utf-8

import os
import sys, fitz
from bs4 import BeautifulSoup
import re
import json
import argparse
import openai
import platform

# your OpenAI API key
openai.api_key = 'sk_YOUR_OPENAI_API_KEY'

def get_arguments(argv):
    parser = argparse.ArgumentParser(description='PyBib')
    # Simulation settings
    parser.add_argument('-i', '--input_folder', default="")
    parser.add_argument('-n', '--name', default="")
    args = parser.parse_args(argv)

    return args

# Function to extract title from pdf
def extract_title(html_text):
    soup = BeautifulSoup(html_text)
    spans = soup.find_all('span',style=True)
    usedFontSize = []
    for span in spans:
        styleTag = span['style']
        fontSize = re.findall("font-size:(\d+)",styleTag)
        usedFontSize.append(int(fontSize[0]))
        
    usedFontSize = sorted(set(usedFontSize))
    title_text = []
    while True and len(usedFontSize)>0:
        max_font = usedFontSize[-1]
        usedFontSize = usedFontSize[:-1]
        title_text = []
        for span in spans:
            #print span['style']
            styleTag = span['style']
            fontSize = re.findall("font-size:(\d+)",styleTag)
            if int(fontSize[0]) == max_font:
                title_text.append(span.text)
        if((len(title_text[0]) < 10 or 'arXiv' in title_text[0]) and len(usedFontSize)>0): # Less than 10 letters, more likely to be journal name
            max_font = usedFontSize[-1]
        else:
            break
    
    return ''.join(title_text).strip()

# Remove the style tag from the pdf
def extract_text(html_text):
    soup = BeautifulSoup(html_text)
    spans = soup.find_all('span',style=True)

    res_text = []
    for span in spans:
        res_text.append(span.text)
    
    return truncate_text(''.join(res_text).strip())

def truncate_text(text: str, max_tokens=2500): # GPT has a max token limit of 4096, but we want to leave some room for more granular tokenization
    tokens = text.split()  # crude tokenization
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
    return " ".join(tokens)

# Function to get summary from gpt
def summarize_text(html_text):
    document = extract_text(html_text)
    response = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=[{"role": "system", "content": "I want you summarize the following text."},
                {"role": "user", "content": f"{document}"}]
    )

    return response.choices[0].message.content.strip()

# Try to load the notebook in the traget directory
if __name__ == '__main__':
    # Check the system
    if platform.system() == "Windows":
        splitter = "\\"
    else:
        splitter = "/"
    # Configuration
    args = get_arguments(sys.argv[1:])

    input_folder = args.input_folder
    name = args.name

    print("Your input folder is " + input_folder)

    if input_folder == "":
        print("Input folder cannot be empty")

    if not input_folder.endswith(splitter):
        input_folder += splitter

    if platform.system() == "Windows": input_folder = input_folder.replace("\\\\", "\\")

    output_folder = splitter.join(input_folder.split(splitter)[:-4])

    local_dir =  '/'.join(input_folder.split(splitter)[-4:])  # local dir need not to be changed accroding to the system

    if name == "":
        name = input_folder.split(splitter)[-2]

    print("Your output file is " + output_folder + splitter+ name + ".ipynb")

    # Get all the files in the folder
    res_links = []
    if(os.path.isfile(output_folder + splitter + name+".ipynb")):
        with open(output_folder + splitter + name+".ipynb", 'r', encoding="utf8") as f:
            res = json.load(f)
        cell_id = int(res['cells'][-1]['id']) + 1
        res_links = [cell['source'][1][9:-10] for cell in res['cells'] if cell['source'][0].startswith("###")]
    else:
        res = { "cells": [], 
           "metadata": {
              "kernelspec": {
               "display_name": "Python 3 (ipykernel)",
               "language": "python",
               "name": "python3"
               },
              "language_info": {
               "codemirror_mode": {
                "name": "ipython",
                "version": 3
               },
               "file_extension": ".py",
               "mimetype": "text/x-python",
               "name": "python",
               "nbconvert_exporter": "python",
               "pygments_lexer": "ipython3",
               "version": "3.8.8"
              }
            },
            "nbformat": 4,
            "nbformat_minor": 5
        }
        cell_id = 0
    
    for paper in os.listdir(input_folder):
        local_link = local_dir + paper
        if local_link not in res_links:
            doc = fitz.open(input_folder + paper)
            html_text = ''
            count = 0
            for page in doc:
                html_text += page.get_text('html')
                count += 1
                if(count > 3): #Only extract the first 3 pages
                    break
            title = extract_title(html_text)

            try:
                summary = summarize_text(html_text)
            except:
                summary = ""
            
            # post process
            title = title.replace("Transportation Research Part E", "")
            title = title.replace("Transportation Research Part D", "")
            title = title.replace("Transportation Research Part C", "")
            title = title.replace("Transportation Research Part B", "")
            title = title.replace("Transportation Research Part A", "")
            title = title.replace("TransportationResearchPartB", "")
            title = title.replace("ScienceDirect", "")
            if(len(title) > 200 or len(title) < 5):
                title  = "Need manual check"

            print(title)
            remote_link = "https://scholar.google.com/scholar?q="+title.replace(" ", "%20")

            one_cell = {
               "cell_type": "markdown",
               "id": str(cell_id),
               "metadata": {},
               "source": [
                "### "+title+"\n",
                "<a href=\""+local_link+"\">PDF</a>" +"\n",
                "<a href=\""+remote_link+"\">Google Scholar</a>"
               ]
              }
            cell_id+=1
            two_cell = {
                "cell_type": "markdown",
               "id": str(cell_id),
               "metadata": {},
               "source": [
                summary
               ]
            }
            cell_id+=1
            res['cells'].append(one_cell)
            res['cells'].append(two_cell)
    
    # Save the notebook
    with open(output_folder+"/" + name+".ipynb", 'w') as f:
        json.dump(res, f)


# In[ ]:




