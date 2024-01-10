from openai import OpenAI, APIConnectionError, RateLimitError
import time

class ChatGPT:
    def __init__(self, api_key: str, model="gpt-3.5-turbo-16k", try_times=5) -> None:
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.try_times = try_times
        
    
    def request(self, text: str, role: str):
        for times in range(self.try_times):
            try:
                completions = self.client.chat.completions.create(
                    model = self.model,
                    messages = [
                        {"role": "system", "content": role},
                        {"role": "user", "content": text}
                    ]
                )
                return completions.choices[0].message.content
            except RateLimitError or APIConnectionError as e:
                print(f"WARNNING: connect openAI error. reason: {e}\nWe will try it again.")
                time.sleep((times+1)*5)

            if times == self.try_times - 1:
                raise OpenAIConnectError(f"We have tried to connect to chatGPT API {self.try_times} times but still no success, please check your internet connection or API Key.")
            


class OpenAIConnectError:
    pass
