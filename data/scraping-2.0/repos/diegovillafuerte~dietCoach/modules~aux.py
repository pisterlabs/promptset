import openai
import os
import sys
import json

api_key = os.getenv("OPENAI_API_KEY")
if api_key is None:
    sys.exit("Error: OPENAI_API_KEY environment variable not set.")

prompt = [
    {
        "role": "system",
        "content": (
            "You are a nutritional information calculator. The user will give a description of a meal and you will"
            " reply with your best estimate for its nutritional information."
        ),
    },
    {
        "role": "user",
        "content": (
            f"For the meal description delimited by --- give your best estimate for calories, carbohydrates, protein, fat, sodium and explain your thought process"
            f" \n--- A large caesar salad with chicken  ---\n"
            f" Only if the text describes food you can recognize, explain your calculation process, and calculate the values. If the text does not describe food you can recognize, respond with 'error' for all fields."
        ),
    }
]  
functions = [
        {
            "name": "get_meal_info",
            "description": "For a given meal, give your best estimate for calories, carbohydrates, protein, fat, sodium and explain your process to be informative about this particular meal.",
            "parameters": {
                "type": "object",
                "properties": {
                    "explanation": {"type": "string","description": "Explain your thought process by going ingredient by ingredient of the given meal. Be detailed an informative."},
                    "calories": {"type": "number","description": "The number of calories in the meal"},
                    "carbohydrates": {"type": "number","description": "The number of carbohydrates in the meal"},
                    "protein": {"type": "number","description": "The number of protein in the meal"},
                    "fat": {"type": "number","description": "The number of fat in the meal"},
                    "sodium": {"type": "number","description": "The number of sodium in the meal"}
                },
                "required": ["explanation", "calories", "carbohydrates", "protein", "fat", "sodium"],
            },
        }
]

def get_nutinfo(prompt,functions, model="gpt-3.5-turbo-0613", temperature=0):

    response = openai.ChatCompletion.create(
        model=model,
        messages=prompt,
        functions=functions,
        temperature=temperature)
    resp = response["choices"][0]["message"]
    if not resp["content"]:
        correct_answer = json.loads(resp["function_call"]['arguments'])
        return correct_answer
    else:
        empty_answer = json.loads('{"explanation": "None", "calories": "0", "carbohydrates": 0, "protein": 0, "fat": 0, "sodium": 0}')
        return empty_answer