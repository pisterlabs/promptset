import openai as oa
import logging
import logging.config

from chatgpt_poc import cparser
from openai.error import RateLimitError

LOGGER = logging.getLogger(__name__)

oa.api_key = str(cparser["openai"]["api_key"])

def request_completion(prompt):
    
    response = oa.Completion.create(
                engine="text-davinci-003",
                prompt=prompt,
                max_tokens=1000
    )
    
    return {
            "creation_id": response.id,
            "model": response.model,
            "prompt_tokens": response.usage.prompt_tokens,
            "choices-text": [choice.text for choice in response.choices],
            "success": True
}
