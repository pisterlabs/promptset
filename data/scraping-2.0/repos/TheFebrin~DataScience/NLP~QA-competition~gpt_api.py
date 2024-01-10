import os
import openai


class GPT3API:
    
    def __init__(self) -> None:
        openai.api_key = os.getenv("OPENAI_API_KEY")

    def send_prompt(
        self, 
        prompt: str, 
        max_length_tokens: int,
        temperature: float,
    ) -> str:
        assert 0 <= temperature <= 1
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            temperature=temperature,
            top_p=1,
            max_tokens=max_length_tokens,
            frequency_penalty=0,
            presence_penalty=0
        )
        return response.choices[0].text
