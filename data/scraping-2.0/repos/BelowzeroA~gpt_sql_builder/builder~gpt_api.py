import openai
from openai.error import RateLimitError

PRICING_GPT4_PROMPT = 0.03
PRICING_GPT4_GENERATION = 0.06
PRICING_GPT3_5_PROMPT = 0.002
PRICING_GPT3_5_GENERATION = 0.002

CHARS_PER_TOKEN = 3.4


class GPTApi:
    """
    A wrapper around OpenAI GPT-family API
    """
    def __init__(self, open_ai_api_key: str):
        openai.api_key = open_ai_api_key
        self.costs = []

    @staticmethod
    def _calculate_cost(num_chars: int, price) -> float:
        return round(num_chars / CHARS_PER_TOKEN * price / 1000, 3)

    def generate(self, system_prompt: str, prompt: str, model="gpt-3.5-turbo-16k", max_tokens=40) -> str:
        pricing_prompt = PRICING_GPT4_PROMPT if model == "gpt-4" else PRICING_GPT3_5_PROMPT
        pricing_generation = PRICING_GPT4_GENERATION if model == "gpt-4" else PRICING_GPT3_5_GENERATION
        prompt_length = len(system_prompt) + len(prompt)
        self.costs.append(self._calculate_cost(prompt_length, pricing_prompt))
        max_tries = 5
        count = 0
        success = False
        while count < max_tries and not success:
            try:
                chat = openai.ChatCompletion.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    top_p=0,
                    temperature=0.0,
                    max_tokens=max_tokens,
                )
                success = True
            except RateLimitError as e:
                continue
                # time.sleep(1)
            except Exception as e:
                count += 1
                raise e
                print(e)

        if not success:
            raise Exception("OpenAI API is not available")

        response_content = chat.choices[0].message.content
        self.costs.append(self._calculate_cost(len(response_content), pricing_generation))
        return response_content
