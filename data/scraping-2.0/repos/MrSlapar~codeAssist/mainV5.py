from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from dotenv import load_dotenv, find_dotenv
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
import textwrap
import os
import getpass
import PyPDF2
import speech_recognition as sr
from gtts import gTTS
import playsound
import time

os.environ['OPENAI_API_KEY'] = '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'
embeddings = OpenAIEmbeddings()


def create_db_from_pdfs(pdf_file_paths):
    all_pages = []
    for pdf_file_path in pdf_file_paths:
        loader = PyPDFLoader(pdf_file_path)
        pages = loader.load_and_split()
        all_pages.extend(pages)
    db = FAISS.from_documents(all_pages, embeddings)
    return db


def get_response_from_query(db, query, k=4):
    docs = db.similarity_search(query, k=k)
    docs_page_content = " ".join([d.page_content for d in docs])

    chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2)

    template = """
     You are a helpful assistant that can answer questions about Slovenia 
        based on the information in the provided PDF document: {docs}
        
        In your answers, do not mention that your knowledge is from the attached PDF document.

        Only use the factual information from the document to answer the question.
        
        If you feel like you don't have enough information to answer the question, say "I don't know".
        
        Your answers should be verbose and detailed.
    
    """

    system_message_prompt = SystemMessagePromptTemplate.from_template(template)

    human_template = "Answer the following question: {question}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    chain = LLMChain(llm=chat, prompt=chat_prompt)

    response = chain.run(question=query, docs=docs_page_content)
    response = response.replace("\n", "")
    return response, docs


# Example usage:
pdf_file_paths = [
    "./PDFS/STATOBOOK_2018.pdf",
    "./PDFS/2021_09_STO_TAM_WEB.pdf",
    "./PDFS/The_Regionalization_of_Slovenia_Regionalizacija_Sl.pdf",
    # Add more paths as needed
]
db = create_db_from_pdfs(pdf_file_paths)

print("Hello, I'm your AI guide. I'm here to answer your questions about tourism in Slovenia.")

# Vprašajte uporabnika za način vnosa
mode = input("Please select your input method:\n1. Text to speech\n2. Speech to speech\n")

if mode == '2':
    # Vprašajte uporabnika, kateri mikrofon želi uporabiti
    mic_list = sr.Microphone.list_microphone_names()
    for i, microphone_name in enumerate(mic_list):
        print(f"{i+1}. {microphone_name}")
    mic_choice = input("Please select the microphone you wish to use: ")
    mic_index = int(mic_choice) - 1
    mic_name = mic_list[mic_index]
    print(f"You have chosen to use {mic_name} for audio input.")

r = sr.Recognizer()

exit_phrases = ["quit", "exit", "goodbye"]

while True:
    try:
        if mode == '1':
            # Text to speech
            query = input("Please type your question: ").lower()
            print("Your question:", query)
        else:
            # Speech to speech
            print("Speak your question:")
            with sr.Microphone(device_index=mic_index) as source:
                r.adjust_for_ambient_noise(source, duration=0.2)
                # Listen for audio with a timeout of 5 seconds
                audio = r.record(source, duration=5)
                # Check if speech is detected
                if audio:
                    query = r.recognize_google(audio)
                    query = query.lower()

                    # Print the speech-to-text output
                    print("Speech-to-Text:", query)

        if any(phrase in query for phrase in exit_phrases):
            break

        response, docs = get_response_from_query(db, query)
        print(textwrap.fill(response, width=10))

        # Convert response to speech
        tts = gTTS(text=response, lang='en')
        tts.save("response.mp3")

        # Play the audio file
        playsound.playsound("response.mp3", True)

    except sr.RequestError as e:
        print("Could not request results; {0}".format(e))

    except sr.UnknownValueError:
        print("Sorry, I did not understand that. Could you please repeat your question?")

    # Delay for a moment before listening again
    time.sleep(0.2)
