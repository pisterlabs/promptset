from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv("../.env")

api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)

letters = int(input("How many letters would you like in your password?\n"))
symbols = int(input("How many symbols would you like?\n"))
numbers = int(input("How many numbers would you like?\n"))

password = client.chat.completions.create(
    messages=[
        {
            "role": "system",
            "content": "You are a password generator, you take a number of letters, symbols and numbers to add to a password and only respond with a password containing those characters, ensuring no weak password patterns are followed.",
        },
        {
            "role": "user",
            "content": ("34 Letters, 23 Symbols, 12 Numbers."),
        },
        {
            "role": "assistant",
            "content": "12p{0F{XnH[k,eml<i~l}[;a5N$K]OC%9B$4A}!jE]g_W?*b1cf1tt@7(aI+ACC492K:H",
        },
        {
            "role": "user",
            "content": ("12 Letters, 15 Symbols, 18 Numbers."),
        },
        {
            "role": "assistant",
            "content": "-K2v80<&U]l1*1~Ii17u8<B]|$o3?25.29*81(8mM1/n0",
        },
        {
            "role": "user",
            "content": (f"{letters} Letters, {symbols} Symbols, {numbers} Numbers."),
        },
    ],
    model="gpt-3.5-turbo",
    max_tokens=100,
    top_p=1,
    temperature=0.1,
)
password = password.choices[0].message.content
print(f"Your password is {password}")
