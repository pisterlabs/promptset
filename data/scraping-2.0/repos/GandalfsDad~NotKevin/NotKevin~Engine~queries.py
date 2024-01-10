import openai
import os
import time

DEFAULT_EMBEDDING_MODEL = 'text-embedding-ada-002'
DEFAULT_COMPLETION_MODEL = 'text-davinci-003'
DEFAULT_CHAT_COMPLETION_MODEL = 'gpt-3.5-turbo'
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_TOKENS = 1024
DEFAULT_EMBEDDING_CHUNK = 1000
DEFAULT_RETRIES = 3

openai.api_key = os.getenv('OPENAI_API_KEY')

def get_embeddings(docs, model = DEFAULT_EMBEDDING_MODEL, chunk = DEFAULT_EMBEDDING_CHUNK):
    try:
        embeddings = []
        if len(docs) > chunk:
            for i in range(0, len(docs), chunk):
                response = get_embeddings(docs[i:i+chunk])
                embeddings.extend(response)          
        else:
            response = openai.Embedding.create(model=model, input = docs) 
            embeddings = [doc['embedding'] for doc in response['data']]

        return embeddings
    except openai.error.RateLimitError as e:
        time.sleep(1)
        return get_embeddings(docs, model, chunk)

def get_completion(input, model = DEFAULT_COMPLETION_MODEL, max_tokens = DEFAULT_MAX_TOKENS, temperature = DEFAULT_TEMPERATURE,stop = None):
    try:
        response = openai.Completion.create(
            model=model,
            prompt=input,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop
            )
    except openai.error.RateLimitError as e:
        time.sleep(1)
        return get_completion(input, model, max_tokens, temperature, stop)
    
    return response['choices'][0]['text']

def get_chat_completion(input,model = DEFAULT_CHAT_COMPLETION_MODEL, max_tokens = DEFAULT_MAX_TOKENS, temperature = DEFAULT_TEMPERATURE,stop = None, attempt = 0):
    try:

        response = openai.ChatCompletion.create(
            model=model,
            messages=input,
            max_tokens=max_tokens
        )
    except openai.error.RateLimitError as e:
        time.sleep(1)
        return get_completion(input, model, max_tokens, temperature, stop)
    except Exception as e:
        if attempt < DEFAULT_RETRIES:
            time.sleep(1)
            return get_chat_completion(input, model, max_tokens, temperature, stop, attempt+1)
    
    return response['choices'][0]['message']['content']