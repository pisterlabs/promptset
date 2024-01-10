from openai import AsyncOpenAI, OpenAI
import tiktoken


class Chat:
    def __init__(self, token: str, model_version: str) -> None:
        self.api_key = token
        self.model_version = model_version

    def get_response(self, message_history: dict, model_version: str = None) -> str:
        '''Fetches response from ChatGPT with entire message history'''
        fetch_model_version = model_version if model_version is not None else self.model_version

        print("Fetching response from ChatGPT")
        completion = OpenAI(api_key=self.api_key).chat.completions.create(
            model=fetch_model_version, messages=message_history)

        response = completion.choices[0].message.content

        # print(response)
        print(f"Response with {len(response)} characters")
        return response

    async def get_response_async(self, message_history: dict, model_version: str = None) -> str:
        '''Fetches response from ChatGPT with entire message history'''
        fetch_model_version = model_version if model_version is not None else self.model_version

        print("Fetching response from ChatGPT")
        completion = await AsyncOpenAI(api_key=self.api_key).chat.completions.create(
            model=fetch_model_version, messages=message_history)

        response = completion.choices[0].message.content

        # print(response)
        print(f"Response with {len(response)} characters")
        return response

    def calculate_tokens(self, messages: dict) -> int:
        '''Calculates an estimate of the tokens used by message history'''
        counter = tiktoken.encoding_for_model(self.model_version)
        raise "Not implemented yet"
        for entry in messages:
            counter.count_tokens(entry.content)
        return counter.count
