import streamlit as st
from key_generator.key_generator import generate
from openai import OpenAI
import numpy as np
import pandas as pd
import time
import yaml
from dotenv import load_dotenv
from helper_functions import refresh
import os

config = yaml.load(open('./configs/config.star.yaml', 'r'),
                   Loader=yaml.FullLoader)

load_dotenv()

API_KEY = os.getenv('OPENAI_API_KEY')

client = OpenAI(
    api_key=API_KEY
)

df = pd.read_csv('./data/star_questions.csv')
tech = pd.read_excel('./data/Data_analyst_question.xlsx')


def get_evaluation(content: str) -> dict:
    """
    Evaluate if the provided answer follows the STAR methodology

    :param content: {'question': the question, 'answer': the answer}
    :return: {"eval": your detailed evaluation of the answer}
    """
    key = generate()

    st.session_state.messages.append(
        {"role": "assistant", "content": "", "key": "assistant-"+key.get_key()})

    st.session_state.response = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system",
              "content": config['prompts']['evaluation_prompt']},
            {"role": "user", "content": content}
        ],
        stream=True
    )

    st.session_state.rateAnswer = False

    st.session_state.responding = True

    refresh("chatcontainer")


def get_random_question():
    """
    get a random question from the csv file

    :return: question 
    """
    random_question_idx = np.random.randint(0, 60)
    data = df.iloc[random_question_idx]

    return data['Question']


def messageFromChatBot():
    """
    get each chunk streamed from the API and add it to the message then refresh except for the first
    4 and last 2 chunks which are these token {eval:" "}

    :return: nothing 
    """
    
    for chunk in st.session_state.response:
        if chunk.choices[0].delta.content is not None:
            if chunk.choices[0].delta.content not in '{\"eval\":\"':
                st.session_state.messages[-1]["content"] += chunk.choices[0].delta.content
                time.sleep(0.01)
                refresh("chatcontainer")
            else:
                st.session_state.skip += chunk.choices[0].delta.content
    st.session_state.messages[-1]["content"] = st.session_state.messages[-1]["content"][:-1]
    st.session_state.messages[-1]["content"] = st.session_state.messages[-1]["content"][:-1]
    # st.session_state.skip=""


def get_technichal_question(selected_category):
    """
    get a random question from the csv file

    :return: question 
    """
    category_data = tech[tech["Category"] == selected_category]
    category_data = category_data.sample(frac=1).reset_index(drop=True)

    return category_data.loc[0, "Question"]


def get_technichal_evaluation(content: str) -> dict:
    """
    Evaluate if the provided answer follows the STAR methodology

    :param content: {'question': the question, 'answer': the answer}
    :return: {"eval": your detailed evaluation of the answer}
    """
    key = generate()

    st.session_state.messages.append(
        {"role": "assistant", "content": "", "key": "assistant-"+key.get_key()})

    st.session_state.response = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system",
              "content": config['prompts']['tech_evaluation_prompt']},
            {"role": "user", "content": content}
        ],
        stream=True
    )

    st.session_state.rateAnswer = False

    st.session_state.responding = True

    refresh("chatcontainer")
