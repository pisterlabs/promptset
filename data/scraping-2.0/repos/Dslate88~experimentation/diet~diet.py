# %!jq
import ast
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import json
import argparse
import re

from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory

meal_schema = {
    "{meal_name}": {
        "ingredients": "{ingredients}",
        "total_calories": 0,
        "total_protein": 0,
        "protein_calories": 0,
        "carb_calories": 0,
        "vegetable_calories": 0,
        "fat_calories": 0,
        "fiber": 0,
        "sodium": 0,
        "saturated_fat": 0,
        "sugars": 0,
        "vitamin_d": 0,
        "calcium": 0,
        "iron": 0,
        "potassium": 0,
        "cholesterol": 0,
        "omega_3": 0,
        "omega_6": 0,
    }
}

def load_meals(filename):
    with open(filename, 'r') as f:
        meals = json.load(f)
    return meals

def update_meals(filename, meal_name, nutritional_values):
    with open(filename, 'r') as f:
        meals = json.load(f)

    meals[meal_name] = nutritional_values

    with open(filename, 'w') as f:
        json.dump(meals, f)

def parse_meal_name_from_input(user_input):
    meal_names = re.findall(r'\[(.*?)\]', user_input)
    return meal_names[0] if meal_names else None

def get_nutritional_values(meals, meal_name):
    if meal_name in meals:
        return meals[meal_name]
    return None

def approximate_nutritional_values(meal_name, additional_info, meal_schema):
    llm = OpenAI(temperature=0.4)
    prompt = PromptTemplate.from_template(
        "What are the nutritional values for {meal_name} given the following information: {additional_info}? "
        "You must return a JSON object with the following schema: {schema}. "
        "If the additional_info does not provide enough information then infer as much as you can to make approximations. "
    )

    chain = LLMChain(llm=llm, prompt=prompt)
    resp = chain.run(
        meal_name=meal_name, additional_info=additional_info, schema=meal_schema
    )

    # Use ast.literal_eval to convert the string into a dictionary
    resp_dict = ast.literal_eval(resp)

    return resp_dict

def start_chat(meals):
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            "This is a conversation between a human and an AI focused on converting the humans input into nutritional values. "
            "The values of importance are calories and protein, however calories should have a total number and a number broken down into "
            "sub-categories of protein, vegetables, fat and carbs."
        ),
        MessagesPlaceholder(variable_name="history"),
        HumanMessagePromptTemplate.from_template("{input}")
    ])

    llm = ChatOpenAI(temperature=0)
    memory = ConversationBufferMemory(return_messages=True)
    conversation = ConversationChain(memory=memory, prompt=prompt, llm=llm)

    print('Start logging your meals (type "quit" to exit):\n')

    while True:
        user_input = input('> ')
        if user_input.lower() == 'quit':
            break

        meal_name = parse_meal_name_from_input(user_input)
        if not meal_name:
            resp = conversation.predict(input=user_input)
            print(resp, '\n')
            continue

        nutritional_values = get_nutritional_values(meals, meal_name)
        if not nutritional_values:
            # Ask for more information
            print('Please provide as much information about this meal as you can.')
            additional_info = input('> ')
            nutritional_values = approximate_nutritional_values(meal_name, additional_info, meal_schema)

        # Here nutritional_values is a dictionary
        update_meals('meals.json', meal_name, nutritional_values)


def main():
    parser = argparse.ArgumentParser(description="A tool for logging meals and calculating nutritional values.")
    parser.add_argument('--type', required=True, choices=['daily_log'], help="Type of action to perform. Currently supports 'daily_log' only.")

    args = parser.parse_args()

    if args.type == 'daily_log':
        meals = load_meals('meals.json')
        start_chat(meals)

if __name__ == "__main__":
    main()

