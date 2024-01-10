import io
import json
import time
from openai import OpenAI
import sys
import openai
import requests
import os
if os.path.dirname(os.path.realpath(__file__)) not in sys.path:
    sys.path.append(os.path.dirname(os.path.realpath(__file__)))
import keys
import apiDataHandler
import exceptions
from contextlib import ExitStack

class AiAssistantManager:
    def __init__(self, content_text, files):
        self.content_text = content_text
        self.files = files

    def __enter__(self):
        client = OpenAI(api_key=keys.openAI_APIKEY)

        self.textFileAssistant = client.beta.assistants.create(
        instructions="You are a helpful robot who extracts flight details from email.",
        model="gpt-4-1106-preview",
        tools=[{"type": "retrieval"}],
        file_ids=[]
        )

        self.thread = client.beta.threads.create(
        messages=[
            {
            "role": "user",
            "content": self.content_text,
            "file_ids": self.files
            }
        ]
        )
        
        return self.textFileAssistant, self.thread

    def __exit__(self, exc_type, exc_value, traceback):
        apiDataHandler.delete_assistant(self.textFileAssistant.id, keys.openAI_APIKEY)
        for file_ in self.files:
            apiDataHandler.delete_file(file_, keys.openAI_APIKEY)

def askGPT(emailText, files, imageInfo=[]):
    client = OpenAI(api_key=keys.openAI_APIKEY)

    for index, file_ in enumerate(files):
        files[index] = client.files.create(
        file=file_,
        purpose='assistants'
        ).id
    

    content_text = """Extract ALL flight details from the text which I will give you. Extract ALL of the following data:
        - currency
        - number of passangers (MUST ALWAYS include in output)
        - maximum number of connections
        - requested airlines with codes
        - whether near airports should be included as departure options
        - amount of checked bags per person (MUST ALWAYS include in output)
        - insurance for the risk of cancellation (say "no" if not specified otherwise)
        - changeable ticket (say "no" if not specified otherwise)

    In the text which you will be given, person is asking for offers for one or more flight options that are usually round-trip if not specified otheriwse. (Keep in mind that each flight segment can have multiple connection points - you must NOT ignore them if they are specified)
    Select only one flight option and extract data for each segment of this specific flight option. There should be only 2 segments. One for outbound and one for return. Use connection points.
    For each flight segment extract the following data:
        - origin location names and IATA 3-letter codes
        - alternative origin locations names and IATA 3-letter codes (only for this specific segment)
        - destination locationname and IATA 3-letter code
        - alternative destination locations names and IATA 3-letter codes (only for this specific segment)
        - included connection points names and IATA 3-letter codes
        - travel class
        - departure date
        - exact departure time
        - earliest departure time
        - latest departure time
        - exact arrival time
        - earliest arrival time
        - latest arrival time
    \n\n"""

    """Timeframe definitions: 
        - morning: from 06:00:00 to 12:00:00
        - evening: from 18:00:00 to 23:59:59
        - afternoon: from 12:00:00 to 18:00:00
        - middle of the day: from 10:00:00 to 14:00:00
    \n\n"""
    #content_text += emailText

    filesPromptText = ""
    if len(files) > 0:
        filesPromptText = "Also, if there are any documents attached, read them too, they provide aditional information. You MUST read every single one of the attached documents (if they are any), as they all include critical information."

    print("filesPromptText", filesPromptText)
    content_text += "Extract ALL flight details from the text which I will give you. Extract data like origin, destionation, dates, timeframes, requested connection points (if specified explicitly) and ALL other flight information. " + filesPromptText + "\n\nProvide an answer without asking me any further questions. Ignore files if you didn't find any.\n\nText to extract details from:\n\n" + emailText
        #content_text = "Extract ALL flight details from the email which I will give you. Extract data like origin, destionation, dates, timeframes, requested connection points (if specified explicitly) and ALL other flight information. Also, if there are any documents attached, read them too, they provide aditional information. You MUST read every single one of the attached documents, as they all include critical information.\n\nProvide an answer without asking me any further questions.\n\nEmail (in text format) to extract details from:\n\n" + emailText
    if len(imageInfo) > 0:
        content_text += "\n\nAlso take this important extra information into consideration:\n" + imageInfo

    #content_text += "\n\nIf there is a specific flight written, say that it is a preffered option."
    
    with AiAssistantManager(content_text, files) as (assistant, thread):
        answer = None
        try:
            answer = runThread(assistant, thread, client)
        except exceptions.stuck as e:
            print(f"Caught an exception: {e}")
            try:
                answer = runThread(assistant, thread, client)
            except exceptions.stuck as e:
                return None
        print("Extracted non-structured data:\n", answer)
        return answer

def runThread(assistant, thread, client):
    assistant_id=assistant.id

    run = client.beta.threads.runs.create(
    thread_id=thread.id,
    assistant_id=assistant_id
    )

    i = 0
    while True:
        time.sleep(3)
        run = client.beta.threads.runs.retrieve(
        thread_id=thread.id,
        run_id=run.id
        )
        print(run)
        print(run.status)

        if run.status == "failed":
            return "There was an error extracting data."
        if run.status == "completed":
            break

        if i >= 100:
            raise exceptions.stuck("no response")

        i += 1

    print("Done")

    messages = client.beta.threads.messages.list(
    thread_id=thread.id
    )
    #print("Extracted non-structured data:\n", messages.data[0].content[0].text.value)
    answer = messages.data[0].content[0].text.value

    return answer

def extractCities(emailText):
        openai.api_key = keys.openAI_APIKEY
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": ("Lets think step by step.\nI will provide you an email text about flight tender inquiry. There will be origin and destination locations of the flight mentioned. Tell me what the initial origin location and final destination of the flight is. (connection flight locations dont count)\n\nEmail text:\n" + emailText)}
            ],
            temperature=0.0
        )

        if response.choices:
            res = (response.choices[0].message.content).lower()
            return res
        else:
            return False
        
def isIntercontinentalFlight(emailText):
        locations = extractCities(emailText)
        print(locations)

        openai.api_key = keys.openAI_APIKEY
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": ("Lets think step by step.\nI will give you 2 locations. Think about in which continents each of these locations are. I want you to tell me if these 2 locations are on different continents. \n\nEmail text:\n" + locations + "\n\nOutput should be ONLY yes/no and NO other text!!. (yes if they are on different continents and no if they are on the same continent)")}
            ],
            temperature=0.0
        )

        if response.choices:
            res = (response.choices[0].message.content).lower()
            print(res)
            if "yes" in res:
                return True
            else:
                return False
        else:
            return False
        
def getTravelClass(emailText):
        openai.api_key = keys.openAI_APIKEY
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": ("I will provide you an email text about flight tender inquiry. I want you to tell me what travel class customer is requesting. Choose one of these options: [ECONOMY, BUSINESS, FIRST]\n\nEmail text:\n" + emailText + "\n\nOutput should be ONLY travel class inside brackets and NO other text - like this: {ECONOMY}")}
            ],
            temperature=0.0
        )

        if response.choices:
            res = (response.choices[0].message.content).upper()
            print(res)
            return res
        else:
            return "ECONOMY"