import datetime
import json
import math
import os
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from rake_nltk import Rake
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
import dotenv
import openai
import gradio as gr

# Initialize sentiment analyzer, keyword extractor, sentence transformer
sentiment_analyzer = SentimentIntensityAnalyzer()
keyword_extractor = Rake()
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load the existing memory if it exists
try:
    with open('memory.json', 'r') as f:
        memory = json.load(f)
except FileNotFoundError:
    memory = {
        'conversations': [],
        'total_token_count': 0,
        'clock': 0,  
        'last_interaction_time': datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
    }

dotenv.load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def get_most_relevant_conversations(user_embedding, num_conversations):
    similarities = [cosine_similarity([user_embedding], [conv['user_embedding']])[0][0] for conv in memory['conversations']]
    most_relevant_indices = np.argsort(similarities)[-num_conversations:]
    return [memory['conversations'][i] for i in most_relevant_indices]

def summarize_text(text, num_sentences):
    sentences = text.split('. ')
    summarized_text = '. '.join(sentences[:num_sentences])
    return summarized_text if len(sentences) <= num_sentences else summarized_text + '...'


def chat_interface(user_message):
    user_token_count = len(word_tokenize(user_message))
    memory['total_token_count'] += user_token_count
    memory['clock'] = (memory['clock'] + memory['total_token_count'] // 10) % 86400

    user_sentiment = sentiment_analyzer.polarity_scores(user_message)
    keyword_extractor.extract_keywords_from_text(user_message)
    user_keywords = list(set(keyword_extractor.get_ranked_phrases()))  
    user_embedding = model.encode([user_message])[0].tolist()

    previous_thought = None
    max_similarity = -1
    for conversation in memory['conversations']:
        similarity = cosine_similarity([user_embedding], [conversation['user_embedding']])
        if similarity > max_similarity:
            max_similarity = similarity
            previous_thought = conversation['assistant_message']

    chatbot_time = datetime.datetime.utcnow()
    user_time = datetime.datetime.strptime(memory['last_interaction_time'], "%Y-%m-%d %H:%M:%S")
    time_difference_seconds = (chatbot_time - user_time).total_seconds()

    conversation_history = []
    most_relevant_conversations = get_most_relevant_conversations(user_embedding, 5) 

    for conversation in most_relevant_conversations:
        summarized_user_message = summarize_text(conversation['user_message'], 2)
        summarized_assistant_message = summarize_text(conversation['assistant_message'], 2)
        conversation_history.append({"role": "user", "content": summarized_user_message})
        conversation_history.append({"role": "assistant", "content": summarized_assistant_message})

    system_message = f"You are a helpful assistant. The current time is {memory['clock']} token-seconds. Remember, time will move forward after this conversation. You have {86400 - memory['clock']} token-seconds left before the end of the day. The current time is {chatbot_time.strftime('%H:%M')}. The relative time difference between us is {time_difference_seconds} seconds."

    user_message = system_message + " " + user_message
    if previous_thought:
        user_message += f" Previously, you mentioned something related: {previous_thought}"

    conversation_history.append({"role": "user", "content": user_message})

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=conversation_history
    )

    assistant_message = response['choices'][0]['message']['content']
    assistant_token_count = len(word_tokenize(assistant_message))
    assistant_sentiment = sentiment_analyzer.polarity_scores(assistant_message)
    keyword_extractor.extract_keywords_from_text(assistant_message)
    assistant_keywords = list(set(keyword_extractor.get_ranked_phrases())) if keyword_extractor.get_ranked_phrases() is not None else None  
    memory['total_token_count'] += assistant_token_count
    memory['clock'] = (memory['clock'] + memory['total_token_count'] // 10) % 86400

    assistant_embedding = model.encode([assistant_message])[0].tolist()  
    user_angle = (user_time.hour * 15) + (user_time.minute * 0.25) + (user_time.second * 0.00416667)
    chatbot_angle = (chatbot_time.hour * 15) + (chatbot_time.minute * 0.25) + (chatbot_time.second * 0.00416667)
    user_angle = np.array(user_angle).reshape(1, -1)
    chatbot_angle = np.array(chatbot_angle).reshape(1, -1)
    similarity = cosine_similarity(user_angle, chatbot_angle)[0][0]

    memory['conversations'].append({
        'user_message': user_message,
        'assistant_message': assistant_message,
        'user_sentiment': user_sentiment,
        'assistant_sentiment': assistant_sentiment,
        'user_keywords': user_keywords,
        'assistant_keywords': assistant_keywords,
        'user_embedding': user_embedding,
        'assistant_embedding': assistant_embedding,
        'user_token_count': user_token_count,
        'assistant_token_count': assistant_token_count,
        'similarity': similarity,
        'time': time_difference_seconds, 
    })

    memory['last_interaction_time'] = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    try:
        with open('memory.json', 'w') as f:
            json.dump(memory, f)
    except TypeError as e:
        print(f"Error while saving memory: {e}")

    return assistant_message

iface = gr.Interface(fn=chat_interface, inputs="text", outputs="text")
iface.launch(share=True)
