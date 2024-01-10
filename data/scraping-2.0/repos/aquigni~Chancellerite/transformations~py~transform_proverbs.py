import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# API key, set it in .env (remember to add .env to .gitignore)
client = OpenAI()

def transform_proverb(proverb, index, total):
    print(f"Processing {index}/{total}")  # Only print the counter

    try:
        completion = client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[
                {"role": "system", "content": "Вы работаете в роли переводчика, который переформулирует поговорки в максимально бюрократический канцеляритный стиль. Пример: «Цыплят по осени считают» превратится в «Подсчет прироста домашней птицы производится после завершения сезона сельскохозяйственных работ»."},
                {"role": "user", "content": f"Переформулируйте поговорку, без заключения её в кавычки и без печати в ответе исходной: '{proverb}'"}
            ]
        )

        transformed = completion.choices[0].message.content.strip()
        return transformed

    except Exception as e:
        print(f"Error with proverb '{proverb}': {e}")
        return None

# Reading proverbs from file
with open("../txt/proverbs.txt", "r") as file:
    proverbs = [line.strip() for line in file if line.strip()]

total_proverbs = len(proverbs)
transformed_proverbs = [transform_proverb(proverb, index+1, total_proverbs) for index, proverb in enumerate(proverbs)]

# Writing result to file
with open("../txt/transformed_proverbs.txt", "w") as file:
    for proverb in transformed_proverbs:
        if proverb is not None:
            file.write(proverb + "\n")
        else:
            file.write("Transformation Failed\n")
