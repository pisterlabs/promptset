import streamlit as st
from pathlib import Path

from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from utils import get_credentials
from database import load_db

# st.title("MovieMuse")

# st.text_input("Enter the kind of movie you want to watch (You can also enter the kind of mood you are in):")
# st.button("Search")

def recommend(user_input: str):
    OPEN_AI_API, ACTIVELOOP_TOKEN = get_credentials()

    prompt = PromptTemplate(
        input_variables=["user_input"],
        template = Path("prompts/app_prompt.prompt").read_text()
    )

    chat_model = ChatOpenAI(openai_api_key=OPEN_AI_API, model="gpt-3.5-turbo", temperature=0.3)
    chain = LLMChain(llm=chat_model, prompt=prompt)

    chain.run(user_input)

    db = load_db("movie-db")

    matches = db.similarity_search_with_score(user_input, k=10)

    movies = []

    for match in matches:
        print(match[0].metadata["movie_name"], match[1])

        movies.append([match[0].metadata["movie_name"],
                       match[0].metadata["movie_id"], 
                        match[0].metadata['year'], 
                        match[0].metadata['genres'], 
                        match[0].metadata['avg_rating'], 
                        match[0].metadata['votes'],
                        match[1]])
        
    return movies






