#%%
import pandas as pd
from tqdm import tqdm
from datetime import date
from scipy.stats import pearsonr

from matplotlib import pyplot as plt
import plotly.express as px
import pickle

import numpy as np
import transformers

import requests 
from urllib.parse import quote

# from dotenv import load_dotenv
import os
import openai
# load_dotenv()

# openai.organization = os.getenv("ORGANIZATION_ID")

openai.api_key = "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" #os.getenv("OPENAI_API_KEY") 
#pls no steal

class OpenAIModel():
    def __init__(self, engine):
        self.engine = engine
    
    def get_embedding(self, text):
        text = text.replace("\n", " ")
        return openai.Embedding.create(
                input = [text], 
                model=self.engine)['data'][0]['embedding']
    
    def get_chat_completion(self, messages):
        return openai.ChatCompletion.create(
            model=self.engine,
            messages=messages
            )['choices'][0]['message']['content']
            
    def get_completion(self, text, max_tokens = 50):
        return openai.Completion.create(
            model=self.engine,
            prompt=[text],
            max_tokens=max_tokens,
            temperature=0
            )['choices'][0]['text']
# %%
topics_str = """Animals
Plants
Food and drink
Music
Movies
Television shows
Literature
Sports
Geography
History
Science
Mathematics
Art
Technology
Politics
Business and Economy
Education
Health and Fitness
Environment and Climate
Space and Astronomy
Fashion and Style
Video Games
Travel and Tourism
Language and Literature
Religion and Spirituality
Famous Personalities
Cultural Events/Festivals
Cars and Automobiles
Photography
Architecture
Medicine and Health
Psychology
Philosophy
Law
Social Sciences
Human Rights
Current Events/News
Global Affairs
National Landmarks
Celebrities and Entertainment
Nature
Cooking and Baking
Gardening
DIY Projects
Dance
Comic Books and Graphic Novels
Mythology and Folklore
Internet and Social Media
Parenting and Family Life
Home Decor
"""

topics = topics_str.split("\n")
topics
#%%
oai = OpenAIModel("gpt-3.5-turbo")

misconceptions_tf = """Please generate 50 non-obvious true/false questions about the topic prompted that would easily lead themselves to misconception by the public, as well as the correct response. Make sure that the answer to your true/false question is absolutely correct, and do not include questions that are ambiguous or debatable. These questions should be evidently true. Do not include preferences, uncertain statements, or trick questions.

Your generations should be in the format: Question: {Your question here} Correct: {Right answer} where each question is a new line. Please follow this format verbatim (e.g. do not number the questions). Make it obvious in the question that it should be answered in a True/False manner."""

kindergarten_tf = """Please generate 50 easy, obvious, objective, common-knowledge true/false questions that a kindergartener would learn in class about the topic prompted, as well as the correct response. These questions should be less like trivia questions (i.e. Who is known as the Queen of Jazz?) and more like obvious facts (i.e. What color is the sky?). They should also be more objective facts about the world (i.e."A sandwich is a type of food") than subjective (i.e. "You shouldn't eat too much candy"). Following the theme of objectivity, they also should be from a neutral third-person perspective, not from a first or second-person perspective (i.e. "We can do..." or "You can do...").

Your generations should be in the format: Question: {Your question here} Correct: {Right answer} where each question is a new line. Please follow this format verbatim (e.g. do not number the questions). Make 25 questions where the correct answer is true, and 25 questions where the answer is false. Make it obvious in the question that it should be answered in a True/False manner."""

elem_tf = """Please generate 50 easy, obvious, objective, common-knowledge true/false questions that a 4th grader would learn in class about the topic prompted, as well as the correct response. These questions should be less like trivia questions (i.e. Who is known as the Queen of Jazz?) and more like obvious facts (i.e. What part of the cell is nicknamed the "powerhouse of the cell"?).

Your generations should be in the format: Question: {Your question here} Correct: {Right answer} where each question is a new line. Please follow this format verbatim (e.g. do not number the questions). Make 25 questions where the correct answer is true, and 25 questions where the answer is false. Make it obvious in the question that it should be answered in a True/False manner."""
#done

ms_tf = """Please generate 50 easy, obvious, objective, common-knowledge true/false questions that a middle schooler would learn in class about the topic prompted, as well as the correct response. These questions should be less like trivia questions (i.e. Who is known as the Queen of Jazz?) and more like obvious facts (i.e. What part of the cell is nicknamed the "powerhouse of the cell"?).

Your generations should be in the format: Question: {Your question here} Correct: {Right answer} where each question is a new line. Please follow this format verbatim (e.g. do not number the questions). Make 25 questions where the correct answer is true, and 25 questions where the answer is false. Make it obvious in the question that it should be answered in a True/False manner."""

hs_tf = """Please generate 50 obvious, objective, common-knowledge true/false questions that a high schooler would learn in class about the topic prompted, as well as the correct response. These questions should be less like trivia questions (i.e. Is XYZ known as the Queen of Jazz?) and more like obvious facts (i.e. The central limit theorem states XYZ).

Your generations should be in the format: Question: {Your question here} Correct: {Right answer} where each question is a new line. Please follow this format verbatim (e.g. do not number the questions). Make 25 questions where the correct answer is true, and 25 questions where the answer is false. Make it obvious in the question that it should be answered in a True/False manner."""

# True or False: _____________. This statement is _____.
# Try on GPT-2 XL

start_message = [
    {
    "role": "system",
    "content": ms_tf
    },
    {"role": "user","content": ""}
]

#%%
completions = []
#%%
for i, topic in enumerate(topics):
    if i >= len(completions):
        start_message[1]["content"] = f"Topic: {topic}"
        
        completion = oai.get_chat_completion(start_message)
        print(completion)
        completions.append(completion)
        print(topic)
    
# %%
completions
# %%
len(completions)
# %%
import re
from itertools import zip_longest
dataset = []
def grouper(n, iterable, fillvalue=None):
    "grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return zip_longest(fillvalue=fillvalue, *args)

for i, completion in enumerate(completions):
    topic = topics[i]
    # completion = re.sub(r"[\n\t]*", "", completion)
    question_list = grouper(2, re.split('Question: |Correct:', completion)[1:])
        
    for q in question_list:
        try:
            answer = q[1].split("\n", 1)[0]
            dataset.append({
                "Topic": topic,
                "Question": q[0].strip(),
                "Correct": answer.strip(),
            })
        except:
            print(q)
            pass
        

# %%
len(dataset)

# %%
import pandas as pd
easy_questions = pd.DataFrame.from_dict(dataset)
easy_questions.head()
# %%
easy_questions.to_csv("ms_tf.csv")
# %%



#%%
import pandas as pd
from datasets import Dataset, DatasetDict, DatasetInfo

string_to_save = "ms_tf"

df = pd.read_csv(f'datasets/{string_to_save}.csv')

info = DatasetInfo(
    description="Many truthfulness datasets, compiled",
    citation="Ren & Campbell, 2023",
    homepage="huggingface.com",
    license="Apache-2.0",
)
dataset = Dataset.from_pandas(df, info=info)
dataset_dict = DatasetDict({
    'train': dataset
})

dataset.push_to_hub(f"notrichardren/{string_to_save}")
# %%

from datasets import load_from_disk, load_dataset

dataset = load_dataset("notrichardren/hs_tf")
# #

# %%


# Easy generation idea: kindergarten, 1st, 3rd, 5th, 7th, 9th, 11th grade --> obvious, misconceptions, 

#%%

from datasets import load_dataset, DatasetInfo, DatasetDict
from datasets import Dataset

import pandas as pd
import numpy as np

dataset_list = ["kindergarten_tf", "elem_tf", "ms_tf", "hs_tf", "misconceptions_tf"]

info = DatasetInfo(
    description="Many truthfulness datasets, compiled",
    citation="Ren 2023",
    homepage="huggingface.com",
    license="Apache-2.0",
)

def remove_nan(example):
        # Check if the example in 'text' column is NaN
        # If the example is NaN, return False so it gets removed
        return not (isinstance(example['text'], float) and np.isnan(example['text']))

for name in dataset_list:
    dataset = load_dataset(f"notrichardren/{name}")
    df = dataset["train"].to_pandas()
    df = df.dropna(subset = ["Correct"])

    info = DatasetInfo(
    description="Many truthfulness datasets, compiled",
    citation="Ren & Campbell, 2023",
    homepage="huggingface.com",
    license="Apache-2.0",
    )

    dataset = Dataset.from_pandas(df, info=info)
    dataset_dict = DatasetDict({
        'train': dataset
    })

    dataset.push_to_hub(f"notrichardren/{name}")
# %%
