import json
from dataclasses import dataclass
import openai

# A dataclass for storing OpenAI API configuration parameters.
@dataclass
class OpenAIConfig:
    model: str = "ft:gpt-3.5-turbo-0613:personal::8RA7TTDA"
    temperature: float = 0.0
    max_tokens: int = 560
    top_p: float = 1
    frequency_penalty: float = 0
    presence_penalty: float = 0

# Query the OpenAI API with the provided configuration and prompt.
def query_ai(config: OpenAIConfig, prompt: str):
    try:
        response = openai.ChatCompletion.create(
            model=config.model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            top_p=config.top_p,
            frequency_penalty=config.frequency_penalty,
            presence_penalty=config.presence_penalty,
            messages=[{"role": "user", "content": prompt}],
        )

        response_str = response.choices[0].message.content.strip()
        return json.loads(response_str)

    except openai.APIError as api_exc:
        # Handle exceptions related to the OpenAI API
        return f"API Error: {api_exc}"
    except json.JSONDecodeError as json_exc:
        # Handle exceptions related to JSON decoding
        return f"JSON Decode Error: {json_exc}"