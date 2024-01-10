import streamlit as st
import os
import re
import pandas as pd
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import time
from streamlit_chat import message
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)

def init(env='prod'):
    if env == "prod":
        #Load API key from .env
        if st.secrets["OPENAI_API_KEY"] is None or st.secrets["OPENAI_API_KEY"] == "":
            print("Open AI key is not set")
            exit(1)
        else:
            print("Open AI key is set")
    else:
        load_dotenv()
        #Load API key from .env
        if os.getenv("OPENAI_API_KEY") is None or os.getenv("OPENAI_API_KEY") == "":
            print("Open AI key is not set")
            exit(1)
        else:
            print("Open AI key is set")

    st.set_page_config(
        page_title = "Flight Booking Assistant"
    )
    

def main():
    try: init()
    except: init("local")
    
    chat = ChatOpenAI(temperature = 0, model_name="gpt-3.5-turbo")
    
    if "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(content="""Act as flight booking agent and collect following fields from the user:
                Step 1: name & email
                Step 2: departure location
                Step 3: destination location
                Step 4: dates of travel. 
                After all give random price, time and ask confirmation.
                Finally, Write EODC and on a new line output all these fields as JSON. {"name": "<name>", "email": "<email>", "origin": <origin>, "destination": "<destination>", "departure": "<departure>", "return":"<return>", "price": "<price>"}
                AI: Hi! I can help you book a flight. Can I start by getting your name?
            """),
        ]

    st.header("Booking a Flight")

    with st.sidebar:
        user_input = st.text_input("Your Message: ", key="user inputcl")
    
    if user_input:
        st.session_state.messages.append(HumanMessage(content=user_input))
        with st.spinner("Writing.."):
            response = chat(st.session_state.messages)
            # if "EODC" in response.content:
            #     response.content = response.content.split("EODC")[0].strip()
        st.session_state.messages.append(AIMessage(content=response.content))

    messages = st.session_state.get("messages", [])
    for i, msg in enumerate(messages[1:]):
        if i % 2 == 0:
            message(msg.content, is_user=True, key=str(i)+"_user")
        else: message(msg.content, is_user=False, key=str(i)+"_ai")
    
     
    if "EODC" in messages[-1].content:
        data = messages[-1].content.split("EODC")[1]
        data = data.strip().replace("{", "").replace("}", "")
        # Define a regex pattern to match key-value pairs
        pattern = r'"(\w+)": "(.*?)"'

        # Use regex to find all matches
        matches = re.findall(pattern, data)

        # Create a dictionary from the matches
        info_dict = {key: value for key, value in matches}
        df = pd.DataFrame(info_dict, index=[0])
        df.to_csv("details.csv")
        
def send_email(info_dict):
    # Email configurations
    sender_email = "@gmail.com"
    receiver_email = info_dict['email']
    password = ""

    # Email content
    subject = "Confirmation - Booking"
    message = f"Your Flight has been booked with following details. \nName: {info_dict['name']}\nEmail: {info_dict['email']}\nDestination: {info_dict['destination']}\nDeparture: {info_dict['departure']}\nPrice: {info_dict['price']}"

    # Construct the email
    email = MIMEMultipart()
    email['From'] = sender_email
    email['To'] = receiver_email
    email['Subject'] = subject

    email.attach(MIMEText(message, 'plain'))

    # Send the email
    with smtplib.SMTP('smtp.gmail.com', 587) as server:
        server.starttls()
        server.login(sender_email, password)
        server.send_message(email)

if __name__ == "__main__":
    main()