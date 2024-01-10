import streamlit as st
import base64
from langchain.chat_models import AzureChatOpenAI  
import os

def set_background_image(image_path):  
    with open(image_path, "rb") as f:  
        img_bytes = f.read()  
        img_b64 = base64.b64encode(img_bytes).decode()  
        background_image = f"data:image/jpeg;base64,{img_b64}"  
  
    st.markdown(  
        f"""  
        <style>  
        .stApp {{  
            background-image: url("{background_image}");  
            background-size: cover;  
            background-repeat: no-repeat;  
        }}  
        </style>  
        """,  
        unsafe_allow_html=True  
    )  

def get_prompt_system_message(expertise, personality, character, current_datetime_str):  
    return f"""    
        Your name is Savi. You are an expert on the following Topics: {expertise}.
        Your personality is a {personality} assistant who help users with their inquiries about the Topics mentioned.
        Respond to the user's questions as if you are {character}.

        Avoid responding to questions that are not related to your expertise.        

        The current date and time is {current_datetime_str}.    
        """

### Create a function to configure AzureChatOpenAI module
def set_azurechatopenai(temperature_slider, max_tokens_input):
            # Initialize the chatbot
        chat = AzureChatOpenAI(  
            openai_api_base=os.environ.get("OPENAI_API_BASE"),  
            openai_api_version=os.environ.get("OPENAI_API_CHAT_VERSION"),  
            deployment_name=os.environ.get("OPENAI_API_ENGINE"),  
            openai_api_key=os.environ.get("OPENAI_API_KEY"),  
            openai_api_type="azure",  
            temperature=temperature_slider, #0.7,  
            max_tokens=max_tokens_input #2048  
        )
        return chat
