import os
import pandas as pd
import datetime
from dotenv import load_dotenv, find_dotenv
import openai
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain

_ = load_dotenv(find_dotenv())
openai.api_key = os.environ['OPENAI_API_KEY']

# Simple exercises to understand how Sequentials chains and  Router chains works

df = pd.read_csv('spotify-2023.csv', sep=";", encoding='latin-1')
songs = df.head()

'''Sequential Chain'''
llm = ChatOpenAI(temperature=0.9, model='gpt-3.5-turbo-16k')

# 1 Chain
first_prompt = ChatPromptTemplate.from_template(
    'What is the most listened song of all in Spotify?'
    '\n\n{songs}'
)
chain_one = LLMChain(llm=llm, prompt=first_prompt, output_key='song')

# 2 Chain
second_prompt = ChatPromptTemplate.from_template(
    'Who is the artist of that song?'
    '\n\n{song}'
)

chain_two = LLMChain(llm=llm, prompt=second_prompt, output_key='artist')

# 3 Chain
third_prompt = ChatPromptTemplate.from_template(
    'What is his least listened song on Apple Music?'
    '\n\n{artist}'
)

chain_three = LLMChain(llm=llm, prompt=second_prompt, output_key='least_song')

# 4 Chain
fourth_prompt = ChatPromptTemplate.from_template(
    'In how many Apple Music playlists is the song listed?'
    '\n\n{least_song}'
)

chain_four = LLMChain(llm=llm, prompt=second_prompt, output_key='listed_am')

# Create the Sequential Chain
overall_chain = SequentialChain(
    chains=[chain_one, chain_two, chain_three, chain_four],
    input_variables=["songs"],
    output_variables=["song", "artist","least_song", "listed_am"],
    verbose=True
)

overall_chain(songs)
