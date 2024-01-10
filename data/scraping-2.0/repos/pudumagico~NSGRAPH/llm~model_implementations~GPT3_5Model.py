from abstract_model import AbstractModel
import openai
from time import sleep


class GPT35Model(AbstractModel):
    def __init__(self, model_name: str, token: str, **kwargs):
        super().__init__(model_name, **kwargs)
        openai.api_key = token

    def _send_prompt(self, prompt) -> str:
        sleep(1)  # Rate limits by openai
        model = "gpt-3.5-turbo"
        failed = True
        while failed:
            failed = False
            try:
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                )
            except Exception as e:
                failed = True
                print(f"Exception {e}")
                print(f"Now waiting 5 min")
                sleep(60*5)
        print(response["choices"][0]["message"]["content"])
        return response["choices"][0]["message"]["content"]
