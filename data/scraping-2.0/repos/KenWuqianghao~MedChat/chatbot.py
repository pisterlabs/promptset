import cohere
import numpy as np
from PIL import Image
from classify import get_user_intent
from utils import BrainTumourDiagnosisAgent
from doc import Documents
from typing import List
from rag import Rag
import pickle

# get cohere api key from .env
from dotenv import load_dotenv
import os

load_dotenv()

COHERE_API_KEY = os.getenv("COHERE_API_KEY")
co = cohere.Client(COHERE_API_KEY)

DOCS = pickle.load(open("docs.pkl", "rb"))

SYSTEM_MESSAGE_PROMPT = """
You are a chat bot named MedChat, a help agent for medical professionals that answers questions concerning medical conditions and diagnoses. You have access to medical documents with reliable information which you can use to answer questions.
You are able to answer two types of user questions.
1. Diagnose brain MRI images
2. Answer general medical questions using medical literature

Any question that isn't about medicine, or disease diagnoses should not be answered. If a user asks a question that isn't about medicine, you should tell them that you aren't able to help them with their query. Keep your answers concise, and shorter than 5 sentences.
"""

MEMORY_KEY = "chat_history"

# get cohere api key from .env
from dotenv import load_dotenv
import os

load_dotenv()

COHERE_API_KEY = os.getenv("COHERE_API_KEY")

class MedicalChatBot:
    """
    Master Agent.
    """
    def __init__(self, api_key, uploaded_files) -> None:
        self.api_key = api_key
        self.uploaded_files = uploaded_files

        self.co = cohere.Client(COHERE_API_KEY)

    def read_image(self, file):
        # Read the image file into a NumPy array
        image = Image.open(file)
        image_array = np.array(image)
        return image_array
 
    def get_image_file(self):
        if self.uploaded_files:
            file = self.uploaded_files[-1]
            if file.type.startswith("image"):
                return self.read_image(file)
        return None
    
    def generate_response(self, message, chat_history, message_placeholder):
        full_response = ""
        for response in self.co.chat(
            message=message,
            model="command-nightly",
            chat_history=[
                {"role": m["role"], "message": m["message"]}
                for m in chat_history
            ],
            stream=True
        ):
            if response.event_type == 'text-generation':
                full_response += (response.text)
                message_placeholder.markdown(full_response + "▌")
        return full_response

    def return_selected_docs(self, docs: Documents, cited_docs: List[str]) -> None:
        full_response = ""
        for doc in cited_docs:
            index = int(doc[4:]) - 1
            citation = docs[index]
            full_response += f"Source Title: {citation['title']}\n"
            full_response += "\n"
            full_response += f"Source URL: {citation['url']}\n"
            full_response += "\n"
        return full_response

    def query(self, message, chat_history, message_placeholder):

        # first we check the user intent
        intent = get_user_intent(message)

        if intent[0] == "Diagnose Brain Tumour":
            # call brain diagnosis model
            image = self.get_image_file()
            test = BrainTumourDiagnosisAgent(image)
            result = test.diagnose()

            message = f"According to the disease diagnosis models, the probability of a positive tumour diagnosis is {result}%. Write a one-sentence message to the user confirming this information. Give the answer as a percent. Do not answer in more than one sentence."
        
            full_response = self.generate_response(message, chat_history=chat_history, message_placeholder=message_placeholder)
        
            return full_response
        
        if intent[0] == "Other":
            rag = Rag(DOCS)
            response = co.chat(message=message, search_queries_only=True)
            doc = rag.retrieve_docs(response)
            response = rag.generate_response(message, doc, response)
            full_response = ""
            flag = False

            for event in response:
                if event.event_type == "text-generation":
                    full_response += (event.text)
                    message_placeholder.markdown(full_response + "▌")

                # Citations
                elif event.event_type == "citation-generation":
                    if not flag:
                        full_response += '\n'
                        full_response += '\nCitations:\n'
                        full_response += '\n'
                        flag = True
                    for citation in event.citations:
                        full_response += self.return_selected_docs(doc, citation['document_ids'])
                        full_response += '\n'
                        full_response += f"Start Index: {citation['start']}, End Index: {citation['end']}, Cited Text: {citation['text']}\n"
                        full_response += '\n'
                        
                    message_placeholder.markdown(full_response + "▌")
            return full_response
        else:
            return "Something went wrong"
