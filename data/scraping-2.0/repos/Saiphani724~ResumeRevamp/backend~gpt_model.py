from openai import OpenAI
from tenacity import retry, wait_random_exponential, stop_after_attempt
import json, os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

class GPTModel:
    def __init__(self, model):
        self.model = model
        self.client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

    @retry(wait=wait_random_exponential(min=1, max=40), stop=stop_after_attempt(2))
    async def get_completion_with_function_call(self, messages, functions, function_call, model = None):
        if model is None:
            model = self.model
        response = await openai.ChatCompletion.acreate(
            model=model,
            messages=messages,
            functions=functions,
            function_call=function_call,
            seed=42,
            temperature=0,  # this is the degree of randomness of the model's output
        )
        result = response.choices[0]['message']['function_call']['arguments']
        return result
    @retry(wait=wait_random_exponential(min=1, max=40), stop=stop_after_attempt(2))
    async def get_completion(self, messages, model = None):
        if model is None:
            model = self.model
        # response = await openai.ChatCompletion.acreate(
        #     model=model,
        #     messages=messages,
        #     seed=42,
        #     temperature=0,  # this is the degree of randomness of the model's output
        # )
        response = self.client.chat.completions.create(
            messages=messages,
            seed=42,
            temperature=0,
            model=model,
        )
        result = response.choices[0].message.content
        # result = response.choices[0]["message"]["content"]
        return result

import time
def wait_for_run_completion(client, thread_id, run_id, sleep_interval=1):
    """
    Waits for a run to complete and prints the elapsed time.:param client: The OpenAI client object.
    :param thread_id: The ID of the thread.
    :param run_id: The ID of the run.
    :param sleep_interval: Time in seconds to wait between checks.
    """
    while True:
        try:
            run = client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run_id)
            if run.completed_at:
                elapsed_time = run.completed_at - run.created_at
                formatted_elapsed_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
                break
        except Exception as e:
            break
        time.sleep(sleep_interval)


class GPTAssistantModel():

    def __init__(self):
        self.client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
    
    def get_completion(self, messages, thread_id):
            if thread_id == '':
                print('Using new thread...')
                thread = self.client.beta.threads.create()
                thread_id = thread.id
                for message in messages:
                    self.client.beta.threads.messages.create(
                        thread_id=thread_id,
                        role=message['role'],
                        content=message['content']
                )
            else: 
                print('Using retrieved thread...')
                thread = self.client.beta.threads.retrieve(thread_id)
                for message in messages:
                    self.client.beta.threads.messages.create(
                        thread_id=thread_id,
                        role=message['role'],
                        content=message['content']
                )
            # print(messages)
            # Run the assistant
            run = self.client.beta.threads.runs.create(
                thread_id=thread_id,
                assistant_id= "asst_lec4DttBXFQfyZ2xzPcUJAto"
            )
            # Wait for the run to complete
            wait_for_run_completion(self.client, thread.id, run.id)

            # Retrieve the last message in the thread
            messages = self.client.beta.threads.messages.list(
                thread_id=thread.id
            )

            last_message = messages.data[0]
            response = last_message.content[0].text.value
            return response, thread.id