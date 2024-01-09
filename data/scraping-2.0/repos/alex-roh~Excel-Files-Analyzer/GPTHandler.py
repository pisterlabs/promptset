import functools
import json
import os
import openai
import tiktoken
import threading
import inspect
import pickle
import hashlib
import os
import logging, sys
import SystemMessages

MAX_TOKENS_FOR_CURRENT_MODEL = 1500 # TODO: Add a feature that allows the user to select the model
openai.api_key = os.environ.get('OPENAI_API_KEY')

class GPTHandler:

    CACHE_DIR = "cache"
    cache = {}  # initialize an empty cache

    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
    
    # class variables
    lock = threading.Lock()
    encoding = tiktoken.get_encoding("cl100k_base")
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    processed_chunks = 0

    # Decorators
    def log_function_call(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # print(f"Calling function {func.__name__} with args={args} and kwargs={kwargs}")
            print(f"Calling function {func.__name__}...")
            return func(*args, **kwargs)
        return wrapper

    @staticmethod
    def _get_file_identifier(combined_string):
        return hashlib.md5(combined_string.encode()).hexdigest()

    @staticmethod
    def _create_cache_key(file_identifier, chunk_identifier):
        return str(hash(f"{file_identifier}_{chunk_identifier}"))
    
    @staticmethod
    def _save_to_cache(file_identifier, chunk_identifier, response):
        cache_file = os.path.join(GPTHandler.CACHE_DIR, f"{file_identifier}_{chunk_identifier}.pkl")
        with open(cache_file, 'wb') as f:
            pickle.dump(response, f)
    
    @staticmethod
    def _get_from_cache(file_identifier, chunk_identifier):
        cache_file = os.path.join(GPTHandler.CACHE_DIR, f"{file_identifier}_{chunk_identifier}.pkl")
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        return None

    @staticmethod
    def clear():
        GPTHandler.processed_chunks = 0

    @staticmethod 
    def get_token_count(content):
        return len(GPTHandler.encoding.encode(content))
    
    @staticmethod
    def get_chunked_tuples(data_structs, gpt_message):
        chunks = []
        current_chunk = ""
        current_token_count = 0

        system_message = gpt_message[0]
        assistant_message = gpt_message[1]

        max_tokens = MAX_TOKENS_FOR_CURRENT_MODEL - GPTHandler.get_token_count(system_message) - GPTHandler.get_token_count(assistant_message)

        # TODO: Add a feature that detects whether current chunk is too large for the model 
        for data_tuple in data_structs:

            if len(data_tuple) == 3:
                idx, target, eval = data_tuple
            elif len(data_tuple) == 2:
                idx, target = data_tuple
                eval = None

            target = str(target).replace('\n', ' ')
            data = str(idx) + ":" + target
            if eval is not None:
                data += ":" + eval + "\n"

            token_count = GPTHandler.get_token_count(data)
            
            if (current_token_count + token_count) > max_tokens:
                # If token limit exceeded, finalize the current chunk and start a new one
                current_chunk = current_chunk[:-1] # Remove the last newline
                chunks.append(current_chunk)
                current_chunk = ""
                current_token_count = 0

            current_chunk += data
            current_token_count += token_count
        
        # Add the last chunk if it has any data
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks

    @staticmethod
    def __get_response_from_chatgpt(chunk, gpt_message):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": SystemMessages.ALL_MESSAGES[gpt_message][0]},
                {"role": "assistant", "content": SystemMessages.ALL_MESSAGES[gpt_message][1]},
                {"role": "user", "content": chunk},
            ],
        ) 
        return response.choices[0].message["content"].strip()

    @staticmethod
    @log_function_call
    def __threaded_get_response(file_identifier, idx_chunk, num_chunks, chunk, response_list, gpt_message, callback=None):

        response = GPTHandler._get_from_cache(file_identifier, idx_chunk)

        try:
            # Check if the response is in the cache first
            if not response:  # If not in cache
                response = GPTHandler.__get_response_from_chatgpt(chunk, gpt_message)
                print(f"{response}")
                GPTHandler._save_to_cache(file_identifier, idx_chunk, response)  # Store the response to the cache
                print(f"({file_identifier}, {idx_chunk}) Response not in cache. Stored in cache.")
            else:
                print(f"({file_identifier}, {idx_chunk}) Response found in cache. Loaded from cache.")

            with GPTHandler.lock:
                response_list.append((idx_chunk, response))
                GPTHandler.processed_chunks += 1
                print(f"{idx_chunk + 1} received: {GPTHandler.processed_chunks}/{num_chunks} completed ({GPTHandler.get_token_count(response)} tokens)")
                if callback:
                    callback(processed_chunks=GPTHandler.processed_chunks)

        except Exception as e:
            print(f"{inspect.currentframe().f_code.co_name}: An error occurred in thread {idx_chunk}: {e}")

    @staticmethod
    def start_threaded_get_response(file_name, chunks, gpt_message, callback=None):

        if not chunks:
            print(f"{inspect.currentframe().f_code.co_name}: Please ensure that chunks are created.")
            return

        response_list = []
        threads = []
        num_chunks = len(chunks)

        for idx, chunk in enumerate(chunks):
            response_thread = threading.Thread(target=GPTHandler.__threaded_get_response, 
                                               args=(GPTHandler._get_file_identifier(file_name), idx, num_chunks, chunk, response_list, gpt_message, callback),
                                               daemon=True)
            response_thread.start()
            threads.append(response_thread)

        # Wait for all threads to finish
        for thread in threads:
            thread.join()

        response_list.sort(key=lambda x: x[0])

        return response_list
    
            