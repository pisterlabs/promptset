#!/usr/bin/env python

import os
import openai
from products_data import products_data
from dotenv import load_dotenv, find_dotenv

# Load the OpenAI API key from the environment variable
load_dotenv(find_dotenv())
openai.api_key = os.environ['API_KEY']

def get_completion_from_messages(messages, model="gpt-3.5-turbo", temperature=0, max_tokens=500):
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message["content"]

def process_user_query(user_message):
    system_message = f"""
    Follow these steps to answer the customer queries. The customer query will be delimited with four hashtags, i.e. ####.

    Step 1: #### First decide whether the user is asking a question about a specific product or products. Product category doesn't count.

    Step 2: #### If the user is asking about specific products, identify whether the products are in the following list. ```{products_data}```

    Step 3: #### If the message contains products in the list above, list any assumptions that the user is making in their message.

    Step 4: #### If the user made any assumptions, figure out whether the assumption is true based on your product information.

    Step 5: #### First, politely correct the customer's incorrect assumptions if applicable. Only mention or reference products in the list of 5 available products. Answer the customer in a friendly tone.
    """

    messages = [
        {'role': 'system', 'content': system_message},
        {'role': 'user', 'content': f"####{user_message}####"},
    ]

    response = get_completion_from_messages(messages)
    try:
        final_response = response.split("#### Response to user: ####")[-1].strip()
    except Exception as e:
        final_response = "Sorry, I'm having trouble right now. Please try asking another question."

    return final_response

def main():
    user_message1 = "by how much is the BlueWave Chromebook more expensive than the TechPro Desktop"
    user_message2 = "do you sell TVs"

    response1 = process_user_query(user_message1)
    response2 = process_user_query(user_message2)
    print("User message 1:", user_message1)
    print("Response to user 1:", response1)
    print("User message 2:", user_message2)
    print("Response to user 2:", response2)

if __name__ == "__main__":
    main()
    