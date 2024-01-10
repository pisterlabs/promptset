import time
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from llama_index import (
    LLMPredictor,
    ServiceContext,
    StorageContext,
    load_index_from_storage,
)
from llama_hub.file.unstructured.base import UnstructuredReader
from llama_index.tools import QueryEngineTool
from llama_index.agent import OpenAIAgent
from gtts import gTTS
from openai import OpenAI

import speech_recognition as sr

# Initializing the speech recognition engine
recognizer = sr.Recognizer()

from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent
from langchain.chains.conversation.memory import ConversationBufferMemory
import os
from dotenv import load_dotenv

env_found = load_dotenv(dotenv_path=os.path.join(os.getcwd(), ".env"))
STREAMING = True

try:
    # rebuild storage context
    storage_context = StorageContext.from_defaults(persist_dir="./storage")
    # load index
    index = load_index_from_storage(storage_context)
except:
    from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader

    documents = SimpleDirectoryReader(
        input_dir="./data",
        file_extractor={".pdf": UnstructuredReader()}
    ).load_data()
    index = GPTVectorStoreIndex.from_documents(documents)
    index.storage_context.persist()


def query_engine_tool_factory():
    llm_predictor = LLMPredictor(
        llm=ChatOpenAI(
            temperature=0,
            model_name="gpt-3.5-turbo",
            streaming=STREAMING,
        ),
    )
    service_context = ServiceContext.from_defaults(
        llm_predictor=llm_predictor,
        chunk_size=512,
    )

    query_engine = index.as_query_engine(
        service_context=service_context,
        streaming=STREAMING,
    )

    query_engine_tool = QueryEngineTool.from_defaults(
        query_engine=query_engine,
        name="query_tool",
        description=(
            "Useful for questions needing context from the documents. input is a natural language question."
        ),
    )
    return query_engine_tool

memory = ConversationBufferMemory(memory_key="chat_history")
llm = ChatOpenAI(temperature=0)
query_engine_tool = query_engine_tool_factory()
# agent_executor = initialize_agent(
#     [query_engine_tool.to_langchain_tool], llm, agent="conversational-react-description", memory=memory
# )
message_history = [
        {"system": "You are a helpful QA assistant. you are specialised to know extra context about local files through query tool."}]
agent = OpenAIAgent.from_tools(
    tools=[query_engine_tool], verbose=True, chat_history=message_history)

def ai_response(question):
    return agent.chat(question).response

# Function to convert speech to text and display it on the UI
def speech_to_text():
    file_path = filedialog.askopenfilename(filetypes=[("Audio files", "*.wav")])
    
    # with sr.AudioFile(file_path) as source:
    #     audio = recognizer.record(source)
    #     progress['value'] = 50  # Update progress bar
    #     root.update_idletasks()  # Force update of GUI
    client = OpenAI()
    start_time = time.time()
    with open(file_path, "rb") as audio_file:
        response = client.audio.transcriptions.create(
            model="whisper-1", 
            file=audio_file
        )
        text = response.text
    whisper_time = time.time() - start_time
    print(f"Whisper took {whisper_time} seconds")
    try:
        # text = recognizer.recognize_google(audio, )
        status_label.config(text=f"Recognized Speech: {text}")
        progress['value'] = 100  # Update progress bar
        root.update_idletasks()  # Force update of GUI
        start_time = time.time()
        chat_response = ai_response(text)
        response_label.config(text=f"AI Response: {chat_response}")  # Display AI response in the UI
        ai_time = time.time() - start_time
        print(f"AI took {ai_time} seconds")
        import pygame
        # Convert response text back into speech
        start_time = time.time()
        
        audio_response = client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=chat_response,
        response_format="mp3",
        )

        audio_response.stream_to_file("response.mp3")

        # tts = gTTS(text=response, lang='es', )
        # tts.save("response.mp3")
        tts_time = time.time() - start_time
        
        print(f"TTS took {tts_time} seconds")
        
        pygame.mixer.init()
        pygame.mixer.music.load("response.mp3")
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy() == True:
            continue
        pygame.mixer.quit()

    except sr.UnknownValueError:
        status_label.config(text="Could not understand audio")
    except sr.RequestError:
        status_label.config(text="Could not request results; check your network connection")


# Creating the main window
root = tk.Tk()
root.geometry("500x300")
root.title("Speech to Text Converter")
# Creating and configuring widgets
browse_button = tk.Button(root, text="Select Audio File", command=speech_to_text)
browse_button.pack()
progress = ttk.Progressbar(root, length=200, mode='determinate')
progress.pack()
progress['value'] = 0
status_label = tk.Label(root, text="", wraplength=250)
status_label.pack()
response_label = tk.Label(root, text="", wraplength=250)  # New label to display AI response
response_label.pack()

res = ai_response("Hello")
print(res)
root.mainloop()