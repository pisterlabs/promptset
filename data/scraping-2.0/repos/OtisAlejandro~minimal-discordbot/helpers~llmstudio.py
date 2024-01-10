import langchain
from langchain.llms.base import LLM, Optional, List, Mapping, Any
import requests
from pydantic import Field

class LMStudio(LLM):
    endpoint: str = Field(...)

    @property
    def _llm_type(self) -> str:
        return "custom"
    

    def _call(self, prompt: str, stop: Optional[List[str]]=None) -> str:
        data = {
            'messages': [{"role": "user", "content": prompt}],
            'preset': 'None',
            'do_sample': True,
            'temperature': 0.7,
            'max_tokens': 800,
        }

        if stop is not None:
            data["stop"] = stop

        response = requests.post(f'{self.endpoint}/v1/chat/completions', json=data)
        response.raise_for_status()

        json_response = response.json()

        # Uncomment the line below to print the response when debugging.
        # print(json_response)

        if 'choices' in json_response and len(json_response['choices']) > 0 and 'message' in json_response['choices'][0] and 'content' in json_response['choices'][0]['message']:
            text = json_response['choices'][0]['message']['content']
            if stop is not None:
                for sequence in stop:
                    if text.endswith(sequence):
                        text = text[: -len(sequence)].rstrip()

            return text
        else:
            raise ValueError('Unexpected response format from API')