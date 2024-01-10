"""
Module for monitoring Anthropic API calls.
"""

import time
from .__helpers import send_data

# pylint: disable=too-many-arguments
def init(llm, doku_url, api_key, environment, application_name, skip_resp):
    """
    Initialize Anthropic integration with Doku.

    Args:
        llm: The Anthropic function to be patched.
        doku_url (str): Doku URL.
        api_key (str): Authentication api_key.
        environment (str): Doku environment.
        application_name (str): Doku application name.
        skip_resp (bool): Skip response processing.
    """

    original_completions_create = llm.completions.create

    def patched_completions_create(*args, **kwargs):
        """
        Patched version of Anthropic's completions.create method.

        Args:
            *args: Variable positional arguments.
            **kwargs: Variable keyword arguments.

        Returns:
            AnthropicResponse: The response from Anthropic's completions.create.
        """

        start_time = time.time()
        response = original_completions_create(*args, **kwargs)
        end_time = time.time()
        duration = end_time - start_time

        model = kwargs.get('model') if 'model' in kwargs else args[0]
        prompt = kwargs.get('prompt') if 'prompt' in kwargs else args[2]

        prompt_tokens = llm.count_tokens(prompt)
        completion_tokens = llm.count_tokens(response.completion)

        data = {
                "environment": environment,
                "applicationName": application_name,
                "sourceLanguage": "python",
                "endpoint": "anthropic.completions",
                "skipResp": skip_resp,
                "completionTokens": completion_tokens,
                "promptTokens": prompt_tokens,
                "requestDuration": duration,
                "model": model,
                "prompt": prompt,
                "finishReason": response.stop_reason,
                "response": response.completion
        }

        send_data(data, doku_url, api_key)

        return response

    llm.completions.create = patched_completions_create
