import openai
import json
from pprint import pprint

openai.api_key = "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"

number_of_dishes = 3
number_of_ingredients = 5

with open('preprompt.json', 'r', encoding='utf-8') as f:
    pre_prompt = json.load(f)


def get_dishes(ingredient_list):
    
    example_user_input = f"How do I make a dish with the following ingredients in my fridge: {ingredient_list}"
    pre_prompt.append({"role": "user", "content": f"{example_user_input}"})
    print(example_user_input)

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=pre_prompt,
            functions=[
            {
                "name": "generate_food_dishes",
                "description": f"Using the ingredients from user input, generate the a dictionary, with the key being the name of {number_of_dishes} different gourmet food dishes that may or may not include those ingredients, and the value being the list of at least {number_of_ingredients} ingredients needed to make each dish",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "food_name": {
                            "type": "array",
                            "keys":{
                                "type": "string",
                                "description": "a name of a gourmet food dish"
                            },
                            "values":{
                                },
                            "items": {
                                "type": "array",
                                "items": {
                                    "type": "string",
                                    "description": "an ingredient needed to make the dish"
                                },
                                "description": f"the array contains {number_of_ingredients} ingredients needed to make the dish" 
                            },
                            "description": f"List of {number_of_dishes} gourmet food dishes that can be made with the ingredients from user input, and at least {number_of_ingredients} ingredients needed to each the dish"
                        }
                    },
                    "required": ["food_name"]
                }
            }
            ],
            function_call={"name": "generate_food_dishes"},
    )

    reply_content = completion.choices[0].message


    funcs = reply_content.to_dict()['function_call']['arguments']
    funcs = json.loads(funcs)
    dish_list = funcs['food_name']
    # print(dish_list, '\n')
    # return dish_list
    
    food_dict = {}
    for dish in dish_list:
        dish_name = dish[0]
        ingredient_list = dish[1:len(dish)]
        food_dict[dish_name] = ingredient_list
    
    return food_dict


