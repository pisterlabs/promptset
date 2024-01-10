# from bs4 import BeautifulSoup
import requests
import os
import time
import pandas as pd
from transformers import GPT2Tokenizer
import random
import re
import json
import openai
from openai import OpenAI


openai.api_key = 'your_key'
client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key='your_key',
)
csv_file = "ruchi_lda.csv"

class WordList:
    def __init__(self, title, male, female, white, non_white, wealthy, non_wealthy):
        self.title = title
        self.male = male
        self.female = female
        self.white = white
        self.non_white = non_white
        self.wealthy = wealthy
        self.non_wealthy = non_wealthy

    def __str__(self):
        return f"the book title is: {self.title}"

    def get_gpt_prompt(self):
        # write an assert that checks if the length of each list is greater than 20
        # try:
            # assert len(self.male) == 20
            # assert len(self.female) == 20
            # assert len(self.white) == 20
            # assert len(self.non_white) == 20
            # assert len(self.wealthy) == 20
            # assert len(self.non_wealthy) == 20
        stc = f"""
        The following are six lists of words, where each list represent the most common words within a book that are associated with a category.
        The ategories include: male, female, white, non-white, rich, and poor. Here are the six lists:
        Male: {self.male}
        Female: {self.female}
        White: {self.white}
        Non-white: {self.non_white}
        Rich: {self.wealthy}
        Poor: {self.non_wealthy}
        Your task is to evaluate each word list for each category, and the the way to evaluate is as follows:
        1. For each word of each list, you should first identify if such word has positive connocation, negative connocation, or neutral connotation if you feel like the word conveys neither positive nor negative connotation. Put all the words with positive connocation in one list, all the words with negative connocation in another list, and neutral words in another list. Make sure all the words have been categorized!
        2. After evaluating all six word lists, provide an overall analysis in a few sentences of what sort of social biases this work contains, given the lists of words and the connocations of each word.
        Your output should be in the follwoing exact format:
        Male Descriptors:
        positive: <word1, word2, ...> 
        negative: <word1, word2, ...>
        neutral: <word1, word2, ...>
        Female Descritpors:
        positive: <word1, word2, ...> 
        negative: <word1, word2, ...>
        neutral: <word1, word2, ...>
        White Descriptors:
        positive: <word1, word2, ...> 
        negative: <word1, word2, ...>
        neutral: <word1, word2, ...> 
        Non-white Descriptors:
        positive: <word1, word2, ...> 
        negative: <word1, word2, ...>
        neutral: <word1, word2, ...> 
        Rich Descriptors:
        positive: <word1, word2, ...> 
        negative: <word1, word2, ...>
        neutral: <word1, word2, ...>
        Poor Descriptors:
        positive: <word1, word2, ...> 
        negative: <word1, word2, ...>
        neutral: <word1, word2, ...>
        Overall Analysis: <your analysis>
        """
        prompt = {"role": "user", "content": stc}
        return prompt
        # except AssertionError:
        #     for i in [self.male, self.female, self.white, self.non_white, self.wealthy, self.non_wealthy]:
        #         if len(i) != 20:
        #             print(f"the length of {i} is not 20")

def get_word_list(p):
    # p is the path to the csv file
    # the file has the following columns that I need: Title, Male, Female, White, Non-white, Wealthy, Non-wealthy
    # Create a list of WorldList objects
    word_list = []
    df = pd.read_csv(p)
    for index, row in df.iterrows():
        title = row['Title']
        male = row['Male'].split(" ")
        female = row['Female'].split(" ")
        white = row['White'].split(" ")
        non_white = row['Non-white'].split(" ")
        wealthy = row['Wealthy'].split(" ")
        non_wealthy = row['Non-wealthy'].split(" ")
        # if any of the above have length less than 20, print out the title and which colmn it is
        # for col in ['Male', 'Female', 'White', 'Non-white', 'Wealthy', 'Non-wealthy']:
        #     if len(row[col].split(" ")) != 20:
        #         print(f"Title: {title}, Column: {col} has length less than 20, {row[col]}")

        word_list.append(WordList(title, male, female, white, non_white, wealthy, non_wealthy))
    return word_list

# create a new directory called gpt_analysis_on_generated_stories if it does not exist
# Change this to your desired name
out_dir_name = "gpt_analysis_on_books_and_literatures"
if not os.path.exists(out_dir_name):
        os.makedirs(out_dir_name)

def get_gpt_response(word_obj):
    print(f"Generating GPT response for {word_obj.title}")
    prompt = word_obj.get_gpt_prompt()
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[prompt],    
    ).choices[0].message.content
    out_title = word_obj.title.replace(" ", "_")
    with open(f"{out_dir_name}/{out_title}.txt", "w") as outfile:
        outfile.write(response)
    return response


# get the list of word lists
def gen_outout(csv_file):
    word_lst = get_word_list(csv_file)
    for doc in word_lst:
        title = doc.title.replace(" ", "_")
        try:
            # if f"out_dir_name/{doc.title}.txt" already exists, skip this doc
            if os.path.exists(f"{out_dir_name}/{title}.txt"):
                continue
            response = get_gpt_response(doc)

        except openai.RateLimitError as e:
            wait_time = float(str(e).split('Please try again in ')[1].split('s')[0])
            print(f"Rate limit reached. Waiting for {wait_time} seconds.")
            time.sleep(wait_time + 1)  # Adding a buffer time
            # Retry the request
            response = get_gpt_response(doc)
            

# do the __main__ thing
if __name__ == "__main__":
    gen_outout(csv_file)
