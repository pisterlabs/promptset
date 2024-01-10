import os
from pathlib import Path
import re
from openai import OpenAI
from langchain.memory import MongoDBChatMessageHistory
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


client = OpenAI(api_key=(os.environ.get("OPENROUTER_API_KEY")), base_url="https://openrouter.ai/api/v1")
oai = OpenAI(api_key=(os.environ.get("OPENAI_API_KEY")))

def get_recipient_chat_history(recipient):
    try:
        history = MongoDBChatMessageHistory(
            connection_string="mongodb://mongo:xQxzXZEzUilnKKhrbELE@containers-us-west-114.railway.app:6200",
            database_name="test",
            collection_name="message_store",
            session_id=str(recipient),
        )
        return history

    except Exception as e:
        return str(e)

def clean_history(history):
    """does string operations to clean the history therefore reducing the size of the prompt sent to the llm"""
    clean_history = str(history.messages[-5:]).replace(
        ", additional_kwargs={}, example=False", ""
    )
    clean_history = clean_history.replace("content=", "")
    clean_history = clean_history.replace(r"(lc_kwargs={", "")
    clean_history = clean_history.replace(r", 'additional_kwargs': {}", "")
    return clean_history

def extract_messages(cleaned_history):
    # Define regex patterns for AIMessage and HumanMessage
    ai_message_pattern = re.compile(r'AIMessage\(\'(.*?)\'\)', re.DOTALL)
    human_message_pattern = re.compile(r'HumanMessage\(\'(.*?)\'\)', re.DOTALL)

    # Find all matches for AIMessages and HumanMessages
    ai_messages = re.findall(ai_message_pattern, cleaned_history)
    human_messages = re.findall(human_message_pattern, cleaned_history)

    # Map each AI message with the role of assistant
    ai_messages_mapped = [{"role": "assistant", "content": message} for message in ai_messages]

    # Map each human message with the role of user
    human_messages_mapped = [{"role": "user", "content": message.replace('\n', ' ')} for message in human_messages]

    # Combine the lists and sort by their appearance order in the cleaned history
    all_messages = sorted(ai_messages_mapped + human_messages_mapped, key=lambda x: cleaned_history.find(x["content"]))

    return all_messages


class Chipoko:
    '''this class should expose the following methods:
    1) create_message_and_get_response(message):
        - creates a message, retrieves history, cleans history, packages up the new message along with the history, sends to LLM and returns the response from LLM & finally updates history'''
        
    def __init__(self, recipient, name):
        self.client = oai
        self.recipient = recipient
        self.name = name
        self.history = get_recipient_chat_history(recipient)
        self.clean_history = clean_history(self.history)
        
        
    def create_message_and_get_response(self, message):
        system_dict = {"role": "system", "content": f"Your name is Winter, an AI developed by Tarmica Chiwara (Zimbabwean male computer engineering, aged 24) as an experiment of Mistral's AI technology. You are talking to {self.name}. Admit when you don't know something."}
        history_dict = extract_messages(self.clean_history)
        live_message_dict = {"role": "user", "content": message}
        all_messages = [system_dict] + history_dict + [live_message_dict]
        completion = client.chat.completions.create(
        model="mistralai/mistral-7b-instruct",
        messages=all_messages,
        )
        response = completion.choices[0].message.content
        self.history.add_user_message(message)
        self.history.add_ai_message(response)
        return response
    
    def create_audio(self, script):
        '''takes in a script and returns an audio file path'''
        try:
            speech_file_path = Path(__file__).parent / "speech.aac"
            response = self.client.audio.speech.create(
            model="tts-1",
            voice="nova",
            response_format="aac",
            input=script
            )
            response.stream_to_file(speech_file_path)
            return speech_file_path
        except Exception as e:
            logging.error("ERROR OCCURED======================================================================%s", e)
            return e