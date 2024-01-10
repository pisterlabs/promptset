import openai
import os
from dotenv import find_dotenv, load_dotenv

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)


def flashCards (input_texts): # returns list[dict]
    api_key = os.getenv('OPEN_AI_API_KEY', default=None)
    notes = str(input_texts)
    # print("notes: ", notes)
    prompt = "Give me only a json array of 5 objects with a prompt field and an answer field where prompt has the question and answer has the response to the question. Use this input as the material: " + notes

    response = openai.Completion.create(
        engine="gpt-3.5-turbo-instruct",
        prompt=prompt,
        max_tokens=3800,
        api_key=api_key
    )

    answer = response.choices[0].text
    # print(answer)
    return answer