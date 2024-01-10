import streamlit as st
import tweepy
import requests
import os
from dotenv import load_dotenv

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from pymongo.mongo_client import MongoClient
from pages.database import ChatStore
from pages.chat import agent

load_dotenv("./key.env")
password = os.getenv("MONGO_PWD")
user = os.getenv("MONGO_USER")

uri = f"mongodb+srv://{user}:{password}@qndb.fdshmnw.mongodb.net/?retryWrites=true&w=majority"


def phrase_for_socials(query):
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    system_prompt = """You are a social media influencer on Twitter. You cover cutting-edge tech topics such as AI.
    You will take scientifically accurate information and rephrase it to be attractive to social media.
    You will NOT alter the information given, you will only rephrase. The original information should be kept.
    
    Abide by these ruls while completeing the objective above:
    1/ You will make the post engaging to an audience on Twitter.
    2/ You will include all the sources originally provided.
    3/ You will NOT alter any of the content given to you.
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt), ("user", "{input}")
    ])

    output_parser = StrOutputParser()

    chain = prompt | llm | output_parser
    output = chain.invoke({"input": query})

    return output


def main():
    st.header("Post to Socials")
    storage = ChatStore(uri, "qndb", "qna")

    query = st.text_input("Research Goal")
    answered = False

    if query:
        st.write(f"Researching {query}...")
        result = agent({"input": query})
        result = phrase_for_socials(result)
        st.info(result)
        answered = True

    if answered:
        edit = st.button("Edit Response")

        if edit:
            result = st.text_area("Edit your response", value=result)
            save_edits = st.button("Save Edits")

            if save_edits:
                st.info(result)


if __name__ == '__main__':
    main()
