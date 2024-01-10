import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import openai
from langchain.document_loaders.unstructured import UnstructuredFileLoader 
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

from langchain.chains import VectorDBQA
from langchain.chat_models import ChatOpenAI
from PyPDF2 import PdfReader


#for voice process
import pyttsx3
import speech_recognition as sr
from bs4 import BeautifulSoup
from selenium import webdriver
engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)
#for voice outout
def speak(audio):
    engine.say(audio)
    engine.runAndWait()
#for voice input
def commandnow():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        # print("Listening...")
        r.pause_threshold=1
        audio = r.listen(source)
    try:
        #print("Recognizing...") 
        query = r.recognize_google(audio, language='en-in') #Using google for voice recognition.h
        print(f"{query}\n")

    except Exception as e:
        # print(e)
        return "None"
    return query    

#end

#pdf
# reader = PdfReader('raw-data/file2.pdf')
# num_pages = len(reader.pages)
# all_text = ''
# for page_num in range(num_pages):
#     page = reader.pages[page_num]
#     text = page.extract_text()
#     all_text += text
# with open('output.txt', 'w', encoding='utf-8') as text_file:
#     text_file.write(all_text)
#end
pdf_directory = 'raw-data/'
all_text = ''

# Iterate through all files in the directory
for filename in os.listdir(pdf_directory):
    if filename.endswith('.pdf'):
        pdf_path = os.path.join(pdf_directory, filename)
        reader = PdfReader(pdf_path)
        
        # Iterate through the pages of the current PDF
        for page in reader.pages:
            text = page.extract_text()
            all_text += text
with open('pdf-output.txt', 'w', encoding='utf-8') as text_file:
    text_file.write(all_text)



app = Flask(__name__)
CORS(app)  # CORS for all routes

# Set your OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')

# Load the document and create Langchain components

loader = UnstructuredFileLoader('raw-data/main-document.txt') #attach your document here that contains raw data
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()


db = Chroma.from_documents(texts, embeddings)

# Initialize Langchain for question answering
qa = VectorDBQA.from_chain_type(llm=ChatOpenAI(), chain_type="stuff", vectorstore=db, k=1)

def chat_with_gpt3(user_input):
    messages = [{"role": "user", "content": user_input}]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=1,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    assistant_response = response["choices"][0]["message"]["content"]
    speak(assistant_response)
    return assistant_response

@app.route("/ask", methods=["POST"])
def ask():
    user_input = request.json.get("user_input", "")
    question_answer_pairs = [
        {
            "question": "Who are you",
            "answer_items": [
                "Personalized Dynamic Chatbot",
                "Your personal chatbot"
                #add pre question answers
            ]
        },
        # Add more question-answer pairs as needed
    ]
    user_input_lower = user_input.lower()
    
    if user_input_lower == "exit":
        return jsonify({"assistant_response": speak("Goodbye!")})
    
    # Check if the user's input is in the question-answer pairs
    for pair in question_answer_pairs:
        if user_input_lower == pair["question"].lower():
            answer_items = pair["answer_items"]
            answer = "\n".join(answer_items)
            return jsonify({"assistant_response": answer})

    # If no match is found in the question-answer pairs, use Langchain for question-answering
    answer = qa.run(user_input)
    
    if answer:
        return jsonify({"assistant_response": answer})
    
    # If Langchain doesn't have an answer, use ChatGPT
    assistant_response = chat_with_gpt3(user_input)
    return jsonify({"assistant_response": assistant_response})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
