import sys
import os
import json
import speech_recognition as sr
import pyttsx3
from dotenv import load_dotenv
import openai
import torch
from transformers import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel
import pinecone
from urllib.parse import urlparse
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Initialize tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2Model.from_pretrained('gpt2')
generator = GPT2LMHeadModel.from_pretrained('gpt2')

# Initialize speech recognition and text-to-speech engines
r = sr.Recognizer()
engine = pyttsx3.init()

if os.path.exists("debug.env"):
    load_dotenv("debug.env")

openai.api_key = os.environ['OPENAI_API_KEY']

def create_pinecone_index(api_key, environment, index_name):
    pinecone.init(api_key=api_key, environment=environment)
    index = pinecone.Index(index_name=index_name)
    return index

def generate_embedding(text):
    inputs = tokenizer.encode_plus(text, return_tensors='pt', truncation=True, max_length=1024)
    
    with torch.no_grad():
        outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).numpy()[0]  # Average pooling

def retrieve_similar_texts(embedding, pinecone_index, num_matches=10):
    query_result = pinecone_index.query(queries=[embedding.tolist()], top_k=num_matches)
    return query_result

def generate_response(name, question, context):
    system = f"You're a sales associate of {name}. Your job is to pitch {name} to customers who will ask you questions about {name}. You will answer questions as accurately as possible given background context and ensure that your answers are never vague, giving specific, pointed information that only applies to {name}. Your answers MUST be conversational, as in they should be written as if you were speaking verbally to a customer."
    prompt = f"You've recieved the following question \"{question}\". Here's some background information: {context}."

    def chat(system, user_assistant, max_tokens=500):
        assert isinstance(system, str), "`system` should be a string"
        assert isinstance(user_assistant, list), "`user_assistant` should be a list"
        system_msg = [{"role": "system", "content": system}]
        user_assistant_msgs = [
            {"role": "assistant", "content": user_assistant[i]} if i % 2 == 0 else {"role": "user", "content": user_assistant[i]}
            for i in range(len(user_assistant))]

        msgs = system_msg + user_assistant_msgs
        response = openai.ChatCompletion.create(model="gpt-3.5-turbo",
                                                messages=msgs,
                                                max_tokens=max_tokens)
        status_code = response["choices"][0]["finish_reason"]
        assert status_code == "stop", f"The status code was {status_code}."
        return response["choices"][0]["message"]["content"]

    try:
        summary = chat(system, [prompt])
        return summary
    except Exception as e:
        print(f"An error occurred: {e}")
        return "Response Error"

def listen():
    with sr.Microphone() as source:
        print("Listening...")
        audio = r.listen(source)
    try:
        text = r.recognize_google(audio)
        print(f"You said: {text}")
        return text
    except Exception as e:
        print(f"Error: {e}")
        return ""

def speak(text):
    engine.say(text)
    engine.runAndWait()

def main(url):
    name = urlparse(url).netloc.split('www.')[-1].split('.')[0]
    print("SEARCHING", name, "IN", url)

    pinecone_index = create_pinecone_index(
        api_key=os.environ['PINECONE_API_KEY'],
        environment="us-west1-gcp-free",
        index_name=f"{name}-embeddings"
    )

    # Load the text data
    with open('text_data.json', 'r') as f:
        text_data = json.load(f)

    while True:
        print("\n\nListening for your question...")
        question = listen()

        if not question:
            print("I couldn't understand you. Please try again.")
            continue

        embedding = generate_embedding(question)
        query_result = retrieve_similar_texts(embedding, pinecone_index)
        
        if query_result:
            best_matches = query_result['results'][0]['matches'][:5]  # Get top 5 matches
            best_match_texts = []

            for match in best_matches:
                match_id = match['id']
                if match_id in text_data:
                    best_match_texts.append(text_data[match_id])
            
            if best_match_texts:
                # Combine all best match texts into one background information string
                context = "\n".join(best_match_texts)
                response = generate_response(name, question, context)
                print(response)
                speak(response)
            else:
                error_message = "No similar texts found. Please try asking a different question."
                print(error_message)
                speak(error_message)

if __name__ == "__main__":
    url = sys.argv[1]
    main(url)
