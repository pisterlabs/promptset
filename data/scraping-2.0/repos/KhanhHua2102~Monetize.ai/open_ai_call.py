import time

import openai
from flask import request

import sql
from config import Config


openai.api_key = Config.OPENAI_API_KEY


def update_openai_key(email, key):
    """
    Update OpenAI key

    Args:
        key (string): new OpenAI key
    """
    sql.update_api_key(email, key)
    Config.OPENAI_API_KEY = key

    openai.api_key = key

    # test if key is valid
    if davinci_003("test api key") is not None:
        return True
    else:
        raise ValueError("API key not working!")


def davinci_003(query, temperature=0):
    print("Starting text-davinci-003...\n")

    start_time = time.time()

    response = openai.Completion.create(
    model="text-davinci-003",
    prompt=query,
    temperature=temperature,
    max_tokens=100,
    top_p=0,
    frequency_penalty=0.0,
    presence_penalty=0.0,
    stop=["|"])

    end_time = time.time()
    elapsed_time = end_time - start_time

    print("\nTime elapsed: " + str(elapsed_time) + " seconds\n")

    response = response.choices[0].text.strip("'").strip(' ')
    return response

def gpt_3(message_list, temperature=0.2):
    """Uses OpenAI's GPT-3.5-turbo API to generate a response to a query

    Args:
    query (str): The query to be sent to the API
    temperature (float): The temperature of the response, which controls randomness. Higher values make the response more random and vice versa.
        (default is 0.2)

    Returns:
        response: The response from the API
    """

    print("Starting GPT-3.5-turbo...\n")

    start_time = time.time()

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=message_list,
        temperature=temperature,)
    

    end_time = time.time()
    elapsed_time = end_time - start_time
    print("\nTime elapsed: " + str(elapsed_time) + " seconds\n")

    response = completion.choices[0].message.content
    return response

def gpt_with_info(message_list, temperature=1):
    """Uses OpenAI's GPT-3 API to generate a response to a query, with user information and portfolio information appended to the query

    Args:
    query (str): The query to be sent to the API
    temperature (float): The temperature of the response, which controls randomness. Higher values make the response more random and vice versa.
        (default is 1)

    Returns:
        response: The response from the API
    """

    email = request.cookies.get('email')
    user_data = sql.get_user_data(email)[1]
    user_info = "User information: Username: " + user_data[0] + ", Email: " + user_data[1] + ", Phone number: " + str(user_data[2]) + "."

    portfolio_data = sql.get_stock_data(email)[1]
    portfolio_info = f"User's risk tolerance: {user_data[3]}. "
    if portfolio_data == []:
        portfolio_info += "User's portfolio is empty."
    else:
        portfolio_info += "User's portfolio information:"
        for stock in portfolio_data:
            portfolio_info += " Date added: " + stock[0] + ", Ticker: " + stock[1] + ", Quantity: " + str(stock[2]) + ", Start price: " + str(stock[3]) + ", End price: " + str(stock[4]) + ", Return percent: " + str(stock[5]) + ", Return amount: " + str(stock[6]) + ", Total: " + str(stock[7]) + "."

    message_list[0] = {"role": "system", "content": "You are a friendly financial chatbot named Monetize.ai. The user will ask you questions, and you will provide polite responses. " + user_info + ' ' + portfolio_info + ". If user ask to change risk tolerance, response that risk tolerance has been changed. If user bought or sold a stock, response that their portfolio has been updated and answer detail about their profit."} 
    
    response = gpt_3(message_list, temperature=temperature)
    return response