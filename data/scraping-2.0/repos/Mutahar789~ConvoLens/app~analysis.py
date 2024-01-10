import os
import json
import openai
import util

os.environ["OPENAI_API_KEY"] = util.OPENAI_API_KEY
model_name = "gpt-3.5-turbo"

system_prompt_script = '''
You are an AI bot that splits the given transcription of a Customer and a CSR done by an ASR system into a dialogue script in the example format:
CSR: dialogue 1
Customer: dialogue 2
CSR: dialogue 3
Customer: dialogue 4

If the first dialogue is of Customer then the dialogue script will start with Customer.
'''

system_prompt_features = '''
You are an AI bot that analyzes the conversation between a Customer and a CSR and returns a list features.
You should ALWAYS give your output in the following JSON format:
{
    "Incomplete Conversation": "Yes/No",
    "Customer Emotion Progression": "Customer left in a worse mood/Customer left in a better mood/No change in the mood",
    "Reason for Customer Emotion Progression": "(30 to 40 characters)",
    "Customer Satisfied": "Yes/No",
    "Query Resolved": "Yes/No",
    "Unprofessional CSR": "Yes/No",
    "Call Purpose": "(30 to 40 characters)",
    "Quality of User Interaction": "integer between 1 and 10"
}
ALWAYS give valid values for each field.

Each field above has options seperated by "/" make sure to use these options ONLY.
Make sure to take the whole conversation in account before analysing.
Adjust the analysis based on sarcasm if present.
'''

def get_transcription(audio_file):
    audio = open(audio_file, "rb")
    transcription = openai.Audio.transcribe("whisper-1", audio)
    return transcription["text"]

def get_script(input):
    messages = [
        {"role": "system", "content": system_prompt_script},
        {"role": "user", "content": input}
    ]
    response = openai.ChatCompletion.create(
        model=model_name,
        messages=messages
    )
    return response["choices"][0]["message"]["content"]

def get_features(input):
    messages = [
        {"role": "system", "content": system_prompt_features},
        {"role": "user", "content": input}
    ]
    response = openai.ChatCompletion.create(
        model=model_name,
        messages=messages
    )
    return json.loads(response["choices"][0]["message"]["content"])