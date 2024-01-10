import streamlit as st
from pymongo import MongoClient
from openai import OpenAI

openaiclient = OpenAI(api_key='<OPEN AI KEY>')
import datetime

# OpenAI API setup

# MongoDB connection setup
mongoclient = MongoClient("<mongodb_uri>")
db = mongoclient["community_fundraiser"]
fundraisers_collection = db["community"]

def generate_description(title):
    try:
        response = openaiclient.chat.completions.create(model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Write a brief description for a fundraiser titled '{title}'."}
        ])
        print(response)
        # Accessing the content of the message from the first choice
        generated_text = response.choices[0].message.content
        return generated_text.strip()
        # return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        print(e)
        return "Description not available"


# Streamlit UI
st.title('Welcome to HelpYourSchool app. Create a New Fundraiser')

title = st.text_input('Fundraiser Title')
generated_description = ''
if title:
    generated_description = generate_description(title)

description = st.text_area('Fundraiser Description', value=generated_description)
start_date = st.date_input('Start Date', min_value=datetime.date.today())
end_date = st.date_input('End Date', min_value=start_date)
goal_amount = st.number_input('Goal Amount', min_value=0.0, format="%.2f")
current_amount = st.number_input('Current Amount (Initial)', min_value=0.0, max_value=goal_amount, format="%.2f")
# Dropdown for wallet selection
wallet_options = [
    'Wallet - ##', 
    'Wallet - ##', 
    'Create New Fund Wallet'
]

selected_wallet = st.selectbox('Select a Wallet', wallet_options)
# Check if 'Create New Fund Wallet' is selected
if selected_wallet == 'Create New Fund Wallet':
    # Generate a new wallet address (or implement your wallet creation logic here)
    new_wallet_id = ###
    selected_wallet = f'New Wallet - {new_wallet_id}'


if st.button('Create Fundraiser'):
    # Save to MongoDB
    new_fundraiser = {
        "_id":6,
        "title": title,
        "description": description,
        "start_date": start_date,  
        "end_date": end_date,
        "goal_amount": goal_amount,
        "current_amount": current_amount,
        "wallet_address": selected_wallet
    }
    fundraisers_collection.insert_one(new_fundraiser)
    st.success('Fundraiser Created Successfully!')

# Close the MongoDB client connection
mongoclient.close()

