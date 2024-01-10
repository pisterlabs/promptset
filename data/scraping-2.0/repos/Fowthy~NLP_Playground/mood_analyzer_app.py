######################################################################
#                                                                    #
#                AI Mood Analyzer from user input                    #
#                                                                    #
######################################################################

import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain.schema import HumanMessage, BaseOutputParser
from typing import List
import os


# Define a class for parsing the output
class ParseOutput(BaseOutputParser[List[str]]):
    def parse(self, text: str) -> List[str]:
        return text

st.title('🎭 Mood Analyzer')

# Define a function to generate responses and analyze mood
# The function accepts only the input text and outputs the chatbot response and the mood analysis.
# It also uses the vector store to store information about the conversation, therefore as long as the session storage is not cleared, the conversation will continue from where it was left off.

# The flow of the model is the following:
# 1. The user enters a prompt for the chatbot
# 2. The model generates a response based on the user prompt
# 3. The model analyzes the mood of the conversation so far
# 4. The model outputs the response and the mood analysis
def generate_response_and_analyze_mood(input_text):
    
    # Define the model that will generate the response based on the user prompt and the mood analysis
    prompt_model = ChatOpenAI(temperature=0.7, openai_api_key=os.environ['OPENAI_API_KEY'], model='gpt-3.5-turbo-1106')
    mood_model = ChatOpenAI(temperature=0.3, openai_api_key=os.environ['OPENAI_API_KEY'], model='gpt-3.5-turbo-1106')
    recommendation_model = ChatOpenAI(temperature=0.3, openai_api_key=os.environ['OPENAI_API_KEY'], model='gpt-3.5-turbo-1106')

    vector_store.setdefault('moodanalyzer_history', []).append(f"User's input: {input_text}")
    
    chat_prompt = ChatPromptTemplate.from_messages([
        "You are an AI bot that responds to human text and nothing else. You just respond with text or question.",
        *vector_store.get('moodanalyzer_history')
    ])


    # Invoke the model chain
    chain = chat_prompt | prompt_model | ParseOutput()
    response = chain.invoke(vector_store)

    # Define the template for the model that will analyze the mood of the conversation so far. It uses the whole conversation as input.
    # The model will output only the emoji, no quotes or other text.
    # The model will output text only when the user's input is concerning and there is something really wrong that must be pointed out.
    chat_mood_prompt = ChatPromptTemplate.from_messages([
        "You are an AI bot that analyzes the mood of the conversation so far. Use only one of three colorful emojis to describe the mood. Green, yellow or red., where green is friendly, yellow is neutral and red is angry. You output only the emoji and also short analysis on the text. Few bullet points about the most important analysis. More concerning the text is, more explanation you provide.",
        *vector_store.get('moodanalyzer_history')
    ])

    # Model for analyzing mood

    chain_mood = chat_mood_prompt | mood_model | ParseOutput()
    mood_analysis = chain_mood.invoke(vector_store)

    vector_store.setdefault('moodanalyzer_history', []).append(f"Model's output: {response}. Mood Analysis: {mood_analysis}. ")

    recomendadion_prompt = ChatPromptTemplate.from_messages([
        "You are an AI bot that gives recommendations based on the mood analysis. You output only the recommendation, no quotes or other text.",
        *vector_store.get('moodanalyzer_history')
    ])

    chain_recommendation = recomendadion_prompt | recommendation_model | ParseOutput()
    recommendation = chain_recommendation.invoke(vector_store)


    # Output the response and the mood analysis to streamlit
    st.info(response)
    st.header('Mood Analysis')
    st.info(mood_analysis)
    st.header('Recommendation')
    st.info(recommendation)

    # Update the vector_store for future interactions

        # Update the vector_store for future interactions
    st.session_state.moodanalyzer_store = vector_store

with st.form('my_form'):
    text = st.text_area('Enter text:')
    submitted = st.form_submit_button('Submit')
    vector_store = st.session_state.get('moodanalyzer_store', {})

    # Clean the vector store on new session
    vector_store['moodanalyzer_history'] = []

    if submitted and os.environ['OPENAI_API_KEY'].startswith('sk-'):
        generate_response_and_analyze_mood(text)