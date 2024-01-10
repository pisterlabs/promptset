import os
import openai
import tiktoken
import json
from datetime import datetime
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) 

def classify(input_string: str) -> str:
    functions = [{
        "name": "print_sentiment",
        "description": "A function that prints the given sentiment",
        "parameters": {
            "type": "object",
            "properties": {
                "sentiment": {
                    "type": "string",
                    "enum": ["positive", "negative", "neutral"],
                    "description": "The sentiment to print",
				},
			},
            "required": ["sentiment"],
		}
	}]
    messages = [{"role": "user", "content": input_string}]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=messages,
        functions=functions,
        function_call={"name": "print_sentiment"},
	)
    function_call = response.choices[0].message["function_call"]
    argument = json.loads(function_call["arguments"])
    return argument

def main():
	positive = "I love your hair!"
	negative = "I hate your hair!"
    
	print(classify(positive))
	print(classify(negative))
        
if __name__ == "__main__":
	main()