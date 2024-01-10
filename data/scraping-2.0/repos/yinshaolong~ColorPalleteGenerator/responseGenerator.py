import openai
from dotenv import dotenv_values
import json

config = dotenv_values(".env")
openai.api_key = config["OPENAI_KEY"]
def get_prompt():
    with open("promptColor.txt") as f:
        return f.read()
    
def response_gen(query):
    prompt = f"{get_prompt()}{query}\nA: "
    response = openai.completions.create(
        prompt=prompt,
        model="gpt-3.5-turbo-instruct",
        max_tokens=200,
        stop="11."
    )
    # return json.loads(response['choices'[0]['text'])

    #Completion is a dictionary. Choices is a dictionary with a list as an object. The first index of the list is a dictionary with text as a key. 
    return json.loads(response.choices[0].text)

def main():
    query = input("Enter a color: ")
    print(response_gen(query))
if __name__ == "__main__":
    main()