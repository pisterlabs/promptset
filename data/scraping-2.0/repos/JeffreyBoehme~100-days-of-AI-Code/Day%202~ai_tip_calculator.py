from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv("../.env")
api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)

system_prompt = "You are a tip calculator, you take the total bill, the number of people splitting the bill and the tip percetage and calculate the amount each person should pay. You then also add a snarky comment about how tipping is stupid and tell them to pay just the amount for the meal."

total = input("How much was the total bill?\n")

people = input("How many peeople are splitting the bill?\n")

tip_amount = input("What percentage would you like to tip?\n")

user_message = f"Total = {total}, People = {people}, Tip percentage = {tip_amount}"


chat_completion = client.chat.completions.create(
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ],
    model="gpt-4-0613",
    temperature=0.6,
    max_tokens=100,
    top_p=1,
)

response = chat_completion.choices[0].message.content
print(response)
