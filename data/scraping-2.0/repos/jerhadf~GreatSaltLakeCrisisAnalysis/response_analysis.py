################################################# SETUP #################################################

# Import libraries
import spacy
import requests
import pandas as pd
from bs4 import BeautifulSoup
import PyPDF2
from io import BytesIO
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import openai
import os
from urllib.parse import urlparse
from requests.exceptions import RequestException
import configparser
import json

# config API keys
config = configparser.ConfigParser()
config.read('config.ini')
openai_key = config['openai']['key']
openai.api_key = openai_key

# Initialize an empty DataFrame to store the data
"""
DataFrame Columns:
    - source: The origin of the response as a name/type (e.g., 'website', 'youtube', 'reddit', etc.).
    - content: The filepath to the text content of the response. This is the main body of the response.
    - url: The URL or source link of the original response. This provides a reference to the original content.
    - stakeholder_type: The type of stakeholders in the response. This represents the groups or individuals who have a stake in the Great Salt Lake crisis, as identified in the response.
    - values A list of the major value types in the response. This represents the main themes or values that the response is promoting or discussing.
    - keywords: A list of the top 5 significant non-stop-words used in the response. These are the words that are most relevant to the content of the response, excluding common stop words like 'the', 'and', 'is', etc.
    - methods: A list of the research methods, techniques, or ways of analyzing the problem that are mentioned in the response. This could include scientific research methods, policy analysis techniques, or other methods of understanding and addressing the crisis.
    - solutions: A list of the solutions to the crisis proposed in the response. These are the specific actions or strategies suggested to address the Great Salt Lake crisis.
    - facts: A list of the facts, numbers, results, or takeaways in the response. This includes any specific data or factual information presented in the response, such as the cost of a proposed solution or the amount of water it could save.
"""

responses_df = pd.DataFrame(columns=["source", "content", "url", "author_names",
                           "value_types", "stakeholder_types", "keywords",
                           "methods", "solutions", "facts"])

# ################################################# DATA ANALYSIS ################################################# 

# Define value types
value_types = ["personal_finance", "sense_of_place", "aesthetics", "health_safety",
               "government_finance", "autonomy", "social_cohesion", "economic_growth", "environmental_protection", 
               "place_attachment", "loss_material_or_emotional", "other"] 

# Initialize a list to store the value types for each response
df["value_types"] = [[] for _ in range(len(df))]

# For each response, classify the content into value types
for i, row in df.iterrows():
    for question_answer in row["content"]:
        # Generate a prompt to classify the content
        prompt = f"Given the following statement, identify the main value types: {question_answer['answer']}"
        
        # Use OpenAI to classify the content
        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=prompt,
            max_tokens=60
        )

        # Parse the model's response to extract the identified value types
        identified_value_types = [value_type for value_type in value_types if value_type in response.choices[0].text.strip().lower()]
        
        # Add the identified value types to the dataframe
        df.at[i, "value_types"].extend(identified_value_types)



# ################################################# AI STUFF ################################################# 

# response = openai.Completion.create(
#   engine="text-davinci-002",
#   prompt="Translate the following English text to French: '{}'",
#   max_tokens=60
# )

# ################################################# DATA VISUALIZATION ################################################# 

# # Create a word map
# text = " ".join(df["keywords"])
# wordcloud = WordCloud().generate(text)

# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis("off")
# plt.show()
