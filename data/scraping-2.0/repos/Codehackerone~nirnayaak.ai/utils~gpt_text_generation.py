import os
import openai
import json
from dotenv import load_dotenv


load_dotenv()


openai.api_key = f"{os.getenv('OPENAI_APIKEY')}"


def get_judgement(search_text):
    try:
        prompt = f"Give a judgment on the basis of Indian Constitution. Add sections in response. Do not reply something controversial. Include similar keywords related to the judgement. \n{search_text}"

        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt,
            temperature=0.7,
            max_tokens=256,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        return response["choices"][0]["text"].strip()
    except:
        return "cannot reply"


def get_title_date_parties(doc_text) :
    try :
        doc_text = doc_text.split()
        prompt = f"Give a title, parties involved, court name, and date from the below text in JSON format only. Keys are title, date, parties and court only. Dont change key names\n{doc_text}"
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt,
            temperature=0,
            max_tokens=256,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        string_json = response["choices"][0]["text"].strip()
        return_json = json.loads(string_json)
        return return_json
    except :
        return "cannot reply"