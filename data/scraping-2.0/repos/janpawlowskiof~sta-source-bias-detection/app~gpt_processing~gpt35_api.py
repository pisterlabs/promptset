import time
import openai


class GPT35Paraphraser:
    def __init__(self, openai_api_key: str, system_prompt: str, temperature=0.5) -> None:
        super().__init__()
        self.system_prompt = system_prompt
        self.temperature = temperature
        openai.api_key = openai_api_key

    def process(self, text: str) -> str:
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                temperature=self.temperature,
                max_tokens=256,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": text},
                ]
            )['choices'][0]
            assert response["finish_reason"] == "stop", f"Model did not return because of 'stop', but {response['finish_reason']} for input: {text}"
            
            response_text: str = response["message"]["content"]
            response_text = response_text.replace("\n", "")
            return response_text

        except openai.error.RateLimitError as ex:
            sleep_time = 10.0
            print(f"Sleeping for {sleep_time} because openai.error.RateLimitError {ex} was thrown")
            time.sleep(sleep_time)
            return self.process(text)
        except openai.error.APIError as ex:
            sleep_time = 10.0
            print(f"Sleeping for {sleep_time} because openai.error.APIError {ex} was thrown")
            time.sleep(sleep_time)
            return self.process(text)
        except openai.error.InvalidRequestError:
            return "invalid input error"
        except AssertionError:
            return "invalid response"
