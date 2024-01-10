import asyncio
import openai
import hashlib
import os

billing = {
    'gpt-3.5-turbo': 0.002,
    'gpt-3.5-turbo-0301': 0.002,
    'gpt-4': 0.03,
    'gpt-4-0314': 0.03,
    'gpt-4-32k': 0.06,
    'gpt-4-32k-0314': 0.06
}

def set_openai_key(api_key):
    openai.api_key = api_key

set_openai_key(os.environ.get('OPENAI_API_KEY'))

class GPTAPI:
    def __init__(self, model_name, budget):
        self._budget = budget
        self._cur_cost = 0
        self._total_tokens = 0
        self._cached_response = {}
        self._model_name = model_name


    def reset_usage(self):
        self._cur_cost = 0
        self._total_tokens = 0

    def print_usage(self):
        print("Current Cost: {}, Total Tokens: {}".format(
            self._cur_cost, self._total_tokens))

    def hash_string(self, input_string):
        return hashlib.sha256(input_string.encode()).hexdigest()

    async def agenerate_chat_response(self, messages, temperature=0.8, use_cache=True):
        if (self._cur_cost >= self._budget):
            print("Budget exceeded. Current cost: {}, Budget: {}".format(
                self._cur_cost, self._budget))
            return None

        event_hash = None
        if use_cache:
            event_hash = self.hash_string(" ".join([message['content'] for message in messages]))

            # wait for the response to be cached
            while event_hash in self._cached_response and self._cached_response[event_hash] == None:
                await asyncio.sleep(1)

            # if the response is cached, return it
            if event_hash in self._cached_response and self._cached_response[event_hash] != None:
                return self._cached_response[event_hash]

            # set cached response to None for inflight requests
            self._cached_response[event_hash] = None

        count = 0
        response = None
        while True:
            try:
                count += 1
                response = await openai.ChatCompletion.acreate(
                    model=self._model_name,
                    messages=messages,
                    temperature=temperature,
                )
                break
            except Exception as e:
                if count > 3:
                    raise e
                print("Error: {}".format(e))
                print("Retrying...")
                await asyncio.sleep(count * count + 1)

        # check if the response is valid
        if (not hasattr(response, 'choices') or len(response.choices) != 1):
            print("Unexpected Response: {}".format(response))
            return None

        # update the cost
        total_tokens = response.usage.total_tokens
        self._cur_cost += total_tokens * (billing[self._model_name] / 1000)
        self._total_tokens += total_tokens

        if use_cache:
            # update the cached response
            self._cached_response[event_hash] = response.choices[0].message.content

        return response.choices[0].message.content

    def generate_chat_response(self, messages, temperature=0.8, use_cache=True):
        return asyncio.run(self.agenerate_chat_response(messages, temperature, use_cache))

    def generate_text(self, prompt):
        return self.generate_chat_response([{"role": "user", "content": prompt}])