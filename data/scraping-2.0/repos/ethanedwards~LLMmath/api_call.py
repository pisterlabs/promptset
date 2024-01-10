#All API relevant functions, especially focused on proper async processing

import asyncio
import aiohttp
from aiohttp import ClientSession
import openai
import config
import time
from random import uniform
import random

# The openAI api interval, currently at one minute
OPENAI_INTERVAL = 60

# Set RETRY_AFTER_STATUS_CODES to API rate limit status code
RETRY_AFTER_STATUS_CODES = (429, 500, 502, 503, 504)


# Retry parameters
# Values set based on OpenAI 60 second interval and assuming this is the most common error
min_wait_time = 8
max_wait_time = 30  # Maximum wait time before retrying when rate limited
jitter = 0.1  # Add some randomness to avoid close polling loops between different clients

#set up openai api key pulled from the config file
openai.api_key = config.openai_api_key
openai.organization = config.openai_org_key


# Create a session to be used by aiohttp for greatest efficiency
def begin_async():
    openai.aiosession.set(ClientSession())

# Close the session when you are done, should be called at end of experiment
async def end_async():
    await openai.aiosession.get().close()


#The general api call function. Takes in the sempaphore for better handling
async def api_call(prompt: str, semaphore: asyncio.Semaphore, model: str='gpt-4', temperature: float=0.0, max_tokens: int=100, messages: list=[], max_retries: int=4):
    # Obtain a semaphore
    async with semaphore:
        result = await create_chat_completion(prompt, model, temperature, max_tokens, messages, max_retries)

    return result

#Key function to actually handle openAI completions
async def create_chat_completion(prompt: str, model: str='gpt-4', temperature: float=0.0, max_tokens: int=100, messages: list=[], max_retries: int=3):
    #Default system prompt
    if not messages:  # if messages list is empty
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    #Random condition, only for testing purposes
    #randcond = random.randint(1, 10) == 1


    #Main completion, using retries with backoff
    for i in range(max_retries):
        

        try:
            #Randomly raise an openAI api exception to test failure handling
            #if randcond:
            #    raise openai.error.APIConnectionError

            chat_completion_resp = await openai.ChatCompletion.acreate(
                model=model, 
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
                )
            
            return chat_completion_resp['choices'][0]['message']['content']
        
        #Common errors which should be retried.
        except aiohttp.ClientError as e:

            print(f"Attempt {i+1}/{max_retries} failed with error: {e}")
            if e.status in RETRY_AFTER_STATUS_CODES or 'request limit' in str(e):

                retry = await exponential_backoff(i, e, max_retries, min_wait_time, max_wait_time, jitter, prompt)
                if retry: continue

        except openai.error.RateLimitError as e:
        #Handle rate limit error (we recommend using exponential backoff)
            print(f"Attempt {i+1}/{max_retries} failed with error: {e}")
            print(f"OpenAI API request exceeded rate limit: {e}")
            retry = await exponential_backoff(i, e, max_retries, min_wait_time, max_wait_time, jitter, prompt)
            if retry: continue

        #Errors which indicate some more fundamental problem like connectivity or OpenAI Server issues
        except openai.error.APIError as e:
        #Handle API error here, e.g. retry or log
            print(f"OpenAI API returned an API Error: {e}")
            print(f'Last attempt was {prompt}')
            pass
        except openai.error.APIConnectionError as e:
            #Handle connection error here
            print(f"Failed to connect to OpenAI API: {e}")
            print(f'Last attempt was {prompt}')
            pass

#Not asynchronous, use for testing
def create_chat_completion_sync(prompt: str, model: str='gpt-4', temperature: float=0.0, max_tokens: int=100, messages: list=[], max_retries: int=3):
    if not messages:  # if messages list is empty
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]

    for i in range(max_retries):
        try:
            chat_completion_resp = openai.ChatCompletion.create(
                model=model, 
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
                )
            
            return chat_completion_resp['choices'][0]['message']['content']
            
        except openai.error.Timeout:
            if i < max_retries - 1:  # i.e. if it's not the final attempt
                time.sleep(5)
            else:
                print(f'Last attempt was {prompt}')
                raise  # re-throw the last exception if all attempts fail
        except openai.error.APIError as e:
        #Handle API error here, e.g. retry or log
            print(f"OpenAI API returned an API Error: {e}")
            print(f'Last attempt was {prompt}')
            pass
        except openai.error.APIConnectionError as e:
            #Handle connection error here
            print(f"Failed to connect to OpenAI API: {e}")
            print(f'Last attempt was {prompt}')
            pass
        except openai.error.RateLimitError as e:
        #Handle rate limit error (we recommend using exponential backoff)
            print(f"OpenAI API request exceeded rate limit: {e}")
            print(f'Last attempt was {prompt}')
            pass

#Handle exponential backoff for retries
async def exponential_backoff(retry_count, exception, max_retries, min_wait_time, max_wait_time, jitter, prompt):
    print(f"Attempt {retry_count+1}/{max_retries} failed with error: {exception}")
    if retry_count < max_retries - 1:
        wait_time = min(max_wait_time, min_wait_time * (2 ** retry_count))  # Exponential backoff
        wait_time += uniform(-jitter, jitter) * wait_time  # Random jitter
        print(f"Waiting {wait_time} seconds before retrying.")
        await asyncio.sleep(wait_time)
        return True
    else:
        print(f"Last attempt was {prompt}")
        raise exception  # Re-raise the last exception

#Special semaphore class to handle the token limit
class TokenLimiter:
    def __init__(self, tokens):
        self.sem = asyncio.Semaphore(tokens)

    async def use_tokens(self, n_tokens):
        #Makes the caller wait until tokens are available to be spent
        for _ in range(n_tokens):
            await self.sem.acquire()

    async def release_tokens(self, n_tokens, waittime: int=OPENAI_INTERVAL):
        #Waits based on api interval to release the total token count
        await asyncio.sleep(waittime)
        for _ in range(n_tokens):
            self.sem.release()