import os
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import wordnet
import os
import pickle
import openai

openai.api_key = "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"

def get_keywords_with_definitions(textfile, num_keywords=12):
    with open(os.getcwd() + "\\" + "\\pickle_notes\\" + textfile + "_pickle", "rb") as fp:  
         text = pickle.load(fp)

    text = ' '.join(text)
    text = text[:4000]
    short_prompt = "Extract keywords thatt are less than 3 words and not a sentence from this text:\n\n"

    keywords = []

    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=short_prompt + text,
        temperature=0.5,
        max_tokens=50,  
        top_p=1.0,
        frequency_penalty=0.8,
        presence_penalty=0.0
    )

    keywords = response.choices[0].text.strip().split("\n")

    print("Keywords:")
    for keyword in keywords:
        print(keyword)




# def get_keywords_with_definitions(textfile, num_keywords=6):

#     with open(os.getcwd() + "\\" + "\\pickle_notes\\" + textfile + "_pickle", "rb") as fp:  
#         text = pickle.load(fp)

#     text = ' '.join(text)
#     # Extract the first 4000 words from the text
#     text = text[:4000]

#     keywords = []

#     while len(keywords) < num_keywords:
#         short_prompt = "Extract keywords thatt are less than 3 words and not a sentence from this text:\n\n"
#         response = openai.Completion.create(
#             model="text-davinci-003",
#             prompt=short_prompt + text,
#             temperature=0.5,
#             max_tokens=50,  
#             top_p=1.0,
#             frequency_penalty=0.8,
#             presence_penalty=0.0
#         )

#         new_keywords = response.choices[0].text.strip().split("\n")

#         # Filter out duplicate keywords and add the new ones
#         for keyword in new_keywords:
#             if keyword not in keywords:
#                 keywords.append(keyword)

#     print("Keywords:")
#     for keyword in keywords:
#         print(keyword)