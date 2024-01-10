#!/usr/bin/env python3

import datetime
import token
from flask import Flask, request, jsonify
import uuid
import openai
import nltk
import pika
import configparser
import json
import threading

app = Flask(__name__)

config = configparser.ConfigParser()
config.read('/etc/secrets.ini')
secret_key = config['openai'].get('secret_key')
# Initialize OpenAI API key
openai.api_key = secret_key

# Searched in order to see if the model contains values
max_total_tokens_search_terms = [["16k", 16384], ["32k", 32768], ["gpt-4", 8191], ["gpt-3.5-turbo", 4096]]
max_total_tokens_search_terms_default = 4096

# Maximum time in minutes to look back in history
oldest_time_minutes = 20

cache = {}
cache_semaphore = threading.Semaphore()

def extract_full_sentence(overall_result):
    # Use NLTK to tokenize the text into sentences
    sentences = nltk.sent_tokenize(overall_result)
    if sentences and len(sentences) > 1:
        return sentences[0]
    return None

@app.route('/chatgpt/stream-to-audio', methods=['POST'])
def chatgpt():
    data = request.json
    text = data.get('text')
    assistant_prompt = data.get('assistant_prompt', "You are a helpful assistant.")
    model = data.get('model', "gpt-4")
    max_tokens = data.get('max_tokens', 250)
    voice_id = data.get('voice_id', '774437df-2959-4a01-8a44-a93097f8e8d5')
    __init_cache(voice_id)

    overall_result = ""
    full_result = ""
    sentence_order = 0

    # Send the result to RabbitMQ
    connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
    channel = connection.channel()
    channel.queue_declare(queue='chatgpt_stream')
    channel.queue_declare(queue='chatgpt_response')

    messages = __setup_messages(max_tokens, model, voice_id, assistant_prompt, text)

    # Connect to ChatGPT API in streaming mode
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        stream=True
    )
    __add_to_cache("user", voice_id, text)
    print("processed response")

    session_id = str(uuid.uuid4())

    for message in response:
        try:
            chunk = message['choices'][0]['delta']["content"]
        except:
            chunk = ""
            pass

        overall_result += chunk
        full_result += chunk

        sentence = extract_full_sentence(overall_result)
        while sentence:
            overall_result = overall_result[len(sentence):]
            data = {"voice_id": voice_id, "text": sentence, "sentance_index": sentence_order, "session_id":session_id}
            channel.basic_publish(exchange='', routing_key='chatgpt_stream', body=json.dumps(data))
            sentence_order += 1
            sentence = extract_full_sentence(overall_result)
    if(overall_result != None and overall_result != ""):
        data = {"voice_id": voice_id, "text": overall_result, "sentance_index": sentence_order, "session_id":session_id}
        channel.basic_publish(exchange='', routing_key='chatgpt_stream', body=json.dumps(data))
        sentence_order += 1

    channel.basic_publish(exchange='', routing_key='chatgpt_response', body=json.dumps({'request': text, 'response': full_result}))
    __add_to_cache("assistant", voice_id, full_result)

    connection.close()

    return jsonify({"message": "Processed", "result": full_result})



def __init_cache(voice_id: str)-> None:
    if(voice_id not in cache.keys()):
        cache_semaphore.acquire()
        cache[voice_id] = []
        cache_semaphore.release()

def __add_to_cache(role: str, voice_id: str, value: str)->None:
    cache_semaphore.acquire()
    message = {"role": role, "content": value, "timestamp": int(datetime.datetime.now().timestamp())}
    cache[voice_id].append(message)
    cache_semaphore.release()

def __get_token_count(search_string: str) -> int:
    for a in max_total_tokens_search_terms:
        if(a[0] in search_string):
            return a[1]
    return max_total_tokens_search_terms_default


def __prune_cache_messages():
    global cache
    check_time = int(datetime.datetime.now().timestamp()) - oldest_time_minutes*60
    cache_semaphore.acquire()
    for key in cache.keys():
        cache[key] = list(filter(lambda x: x["timestamp"] > check_time, cache[key]))
    cache_semaphore.release()

# offical count of tokens is / 4 about Calling out a bit higher because we don't want to hit an error from bringing in too many history messages.
def __count_tokens(string: str):
    return int(len(string) / 3)

def __setup_messages(max_tokens:int, model: str, voice_id: str, assistant_prompt: str, text: str):
    __prune_cache_messages()
    
    additional_prompt = f"\n keep your response to less than {int(max_tokens/1.7)} words"
    assistant_prompt = assistant_prompt + additional_prompt
    tokens_left = __get_token_count(model)
    tokens_left -= __count_tokens(text) - __count_tokens(assistant_prompt) - max_tokens
    
    messages = []
    messages.append({"role": "system", "content": assistant_prompt})
    c = cache[voice_id]
    for i in reversed(range(len(c))):
        tokens_left -= len(c[i]["content"])
        if tokens_left <= 0:
            break
        messages.append({"role": c[i]["role"], "content": c[i]["content"]})

    messages.append({"role": "user", "content": text})
    print(messages)
    return messages


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
