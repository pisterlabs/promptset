from openai import OpenAI
import requests
import json
import gradio as gr
from dotenv import load_dotenv
import os

load_dotenv()
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
OPENAI_API_KEY="OPENAI_API_KEY"

PLACES_API_KEY = os.getenv("PLACES_API_KEY")
PLACES_API_KEY="PLACES_API_KEY"

client = OpenAI(api_key="OPENAI_API_KEY")

def find_places_to_eat(location_query):
    places_api_key = "PLACES_API_KEY"
    url = "https://maps.googleapis.com/maps/api/place/textsearch/json?"

    params = {
        'query': location_query,
        'key': places_api_key
    }
    
    response = requests.get(url, params=params)
    results = response.json().get('results', [])
    
    places = []
    for place in results[:5]:
        places.append({
            'name': place.get('name'),
            'address': place.get('formatted_address'),
            'rating': place.get('rating'),
            'user_ratings_total': place.get('user_ratings_total')
        })

    places_info = "\n\n".join([f"{place['name']} - {place['address']}" for place in places])
    return places_info

# Gradio function that will interact with the OpenAI API
def run_conversation(query):
    messages = [{"role": "user", "content": query}]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "find_places_to_eat",
                "description": "Find places to eat near a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location_query": {
                            "type": "string",
                            "description": "Location to find places to eat near, e.g., 'restaurants near Waterlily Adani'",
                        }
                    },
                    "required": ["location_query"],
                },
            },
        }
    ]
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )

    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls
    
    if tool_calls:
        available_functions = {
            "find_places_to_eat": find_places_to_eat,
        }
        
        messages.append(response_message)
        
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_to_call = available_functions[function_name]
            function_args = json.loads(tool_call.function.arguments)
            function_response = function_to_call(
                location_query=function_args.get("location_query"),
            )
            messages.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool", 
                    "name": function_name,
                    "content": function_response,
                }
            )
        
        second_response = client.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            messages=messages,
        )
        
        #Format the final response
        formatted_response = "\n".join([choice.message.content for choice in second_response.choices])
        
        return formatted_response

demo = gr.Interface(fn=run_conversation, inputs="text", outputs="text").launch(share=True)