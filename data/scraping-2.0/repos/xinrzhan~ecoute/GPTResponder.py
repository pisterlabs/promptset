import openai
from keys import OPENAI_API_KEY
from keys import GOOGLE_API_KEY
from prompts import create_prompt, INITIAL_RESPONSE
import requests
import time

openai.api_key = OPENAI_API_KEY

def generate_response_from_transcript(transcript):
    try:
        response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0301",
                messages=[{"role": "system", "content": create_prompt(transcript)}],
                temperature = 0.0
        )
    except Exception as e:
        print(e)
        return ''
    full_response = response.choices[0].message.content
    try:
        return full_response.split('[')[1].split(']')[0]
    except:
        return ''

def translate_text(text, target_language, api_key):
    url = "https://translation.googleapis.com/language/translate/v2"
    params = {
        "q": text,
        "target": target_language,
        "key": api_key
    }
    response = requests.get(url, params=params)
    translation = response.json()["data"]["translations"][0]["translatedText"]
    return translation

class GPTResponder:
    def __init__(self):
        self.response = INITIAL_RESPONSE
        self.response_interval = 2

    def respond_to_transcriber(self, transcriber):
        while True:
            if transcriber.transcript_changed_event.is_set():
                start_time = time.time()

                transcriber.transcript_changed_event.clear() 
                transcript_string = transcriber.get_transcript()
                response = generate_response_from_transcript(transcript_string)
            
                
                if response != '':
                    api_key = GOOGLE_API_KEY.__str__()
                    text = response.__str__()
                    target_language = "zh-CN" 
                    try:
                        translation = translate_text(text, target_language, api_key)
                    except KeyError:
                        print("Failed to retrieve translation. Check the API response structure.")
                        
                        
                    response =  response + "\n" +target_language.__str__()+": " + translation
                
                end_time = time.time()  # Measure end time
                execution_time = end_time - start_time  # Calculate the time it took to execute the function
                
                if response != '':
                    self.response = response

                remaining_time = self.response_interval - execution_time
                if remaining_time > 0:
                    time.sleep(remaining_time)
            else:
                time.sleep(0.3)

    def update_response_interval(self, interval):
        self.response_interval = interval