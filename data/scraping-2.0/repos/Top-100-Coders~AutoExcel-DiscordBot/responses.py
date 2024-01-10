from openai import OpenAI
import os
import requests
import dotenv

url = "http://localhost:8000/openai"
url_row = "http://localhost:8000/addrow"






def handle_request(message) -> str:
    print(message.author.roles)
    p_message = message.content.lower()
    data = {
    "context": os.environ.get('B'),
    "columns": os.environ.get('C'),
    "data": p_message
}
    print(data)
    print(os.environ.get('A'))
    print(os.environ.get('B'))
    
    response = requests.post(url, json=data)
    print(response)
    print(response.json())
    response_data = response.json()
    print(response)
    row_data= requests.post(url_row+f"/{os.environ.get('A')}", json=response_data)
    print(row_data)
    
    return   format_message(row_data.json())


def format_message(data):
    formatted_message = "A new row was added:\n"
    for key, value in data.items():
        formatted_message += f"{key.capitalize()} - {value}\n"
    return formatted_message.rstrip('\n')