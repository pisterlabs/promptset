from langchain.document_loaders import DirectoryLoader
from langchain.indexes import VectorstoreIndexCreator
from deep_translator import (GoogleTranslator)
from langchain.memory import ConversationBufferMemory

class ConstitutionAI:
    index = object
    memory = object

    def __init__(self):
        pass

    def load_data(self):
        loader = DirectoryLoader('_data', '*.txt')
        self.index = VectorstoreIndexCreator().from_loaders([loader])
        self.memory = ConversationBufferMemory()
    
    def  get_response(self, query):
        translated = GoogleTranslator(source='auto', target='en').translate(text=query)
        translated_response = GoogleTranslator(source='auto', target='pl').translate(text=self.index.query(translated))
        self.memory.chat_memory.add_user_message(translated)
        self.memory.chat_memory.add_ai_message(translated_response)
        return translated_response

    def clear_memory(self):
        self.memory.clear()