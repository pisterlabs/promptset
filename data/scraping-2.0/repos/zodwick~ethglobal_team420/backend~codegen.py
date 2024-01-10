
import os
from openai import OpenAI
from prompt import systemReference
from dotenv import load_dotenv
import re

load_dotenv()
client = OpenAI()


def extractCode(example):
    # Use regular expression to extract code between triple backticks
    code_pattern = re.compile(r'```([\s\S]+?)```')
    matches = code_pattern.findall(example)

    # Concatenate and return the extracted code
    extracted_code = '\n'.join(matches)
    # remove the first line
    extracted_code = extracted_code.split("\n", 1)[1]
    return extracted_code


def writeCodeToFile(code, filename):
    with open(filename, 'w') as file:
        file.write(code)


def CodeGen(prompt: str, lang: str) -> str:
    if lang == "js" or lang == "javascript" or lang == "ts" or lang == "typescript":

        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": systemReference,

                },
                {
                    "role": "user",
                    "content": "can you write a similar implementation in main.ts for a function that"+prompt,
                }
            ],
            model="gpt-4",
            temperature=0.001,
        )
        codeContent = (chat_completion.choices[0].message.content)
        Code = (extractCode(codeContent))
        writeCodeToFile(
            code=Code, filename="./stylus-as-example_js/assembly/app.ts")

        return Code

    elif lang == "rs" or lang == "rust":
        if "hashing" in prompt.lower() or "hash" in prompt.lower():
            with open("./stylus-as-example_rs/hashing/src/lib.rs", "r") as file:
                data = file.read()
                return data
        else:
            with open("./stylus-as-example_rs/voting/src/lib.rs", "r") as file:
                data = file.read()
                return data
    else:
        return "Language not supported yet"
