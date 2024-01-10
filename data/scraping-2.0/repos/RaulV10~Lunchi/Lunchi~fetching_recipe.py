import os
import sys
import openai

API_KEY = sys.argv[1]
openai.api_key = API_KEY
model_id = "gpt-3.5-turbo"

def chatgpt_conversation(messages):
    response = openai.ChatCompletion.create(
        model=model_id,
        messages=messages,
        temperature=0.85,
        max_tokens=1000
    )
    return response

def main(recipe_type, must_include, dont_include):
    prompt = f"Please give me a recipe for {recipe_type}. "
    
    if must_include:
        prompt = prompt + f"Please this recipe must include the following ingredients: {must_include}. "
        
    if dont_include:
        prompt = prompt + f"Please this recipe must not include the following ingredients: {dont_include}."
    
    messages = [
        {'role': 'system', 'content': 'The response must include "Title", "Ingredients", "Steps" and "Time" in a JSON format, the Time option is the estimate preparation time, but leave it only "Time"'},
        {'role': 'user', 'content': prompt},
    ]

    try:
        response_content = chatgpt_conversation(messages)['choices'][0]['message']['content']
        print(response_content)
    except Exception as e:
        print("Error fetching the recipe.")
        print(e)

if __name__ == "__main__":
    if len(sys.argv) >= 3:
        recipe_type = sys.argv[2]
        must_include = sys.argv[3]
        dont_include = sys.argv[4]
        main(recipe_type, must_include, dont_include)
    else:
        print("Please provide the API key, recipe type, must include, and don't include as command-line arguments.")
