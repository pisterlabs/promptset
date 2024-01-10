import time

import openai
from openai.error import APIConnectionError, APIError, ServiceUnavailableError

from openai_token_counter import openai_token_counter
from tests.conftest import ConfigTests
from tests.counter.resources import test_cases_raw


MAX_SERVICE_UNAVAILABLE_RETRY_ATTEMPTS = 5
SLEEP_INTERVAL_BETWEEN_ATTEMPTS = 5
MODEL = "gpt-3.5-turbo"
MODEL_PROMPT_TOKEN_COST_PER_TOKEN = 0.0015 / 1000
MODEL_COMPLETION_TOKEN_COST_PER_TOKEN = 0.002 / 1000
MAX_RESPONSE_TOKENS = 1  # The response tokens doesn't matter in this context, we only calculate request tokens


def test_token_counter(config: ConfigTests) -> None:
    """Test that the token counter works as expected."""
    token_usage = {"prompt": 0, "completion": 0}

    for test_case in test_cases_raw:
        for attempt in range(1, MAX_SERVICE_UNAVAILABLE_RETRY_ATTEMPTS + 1):
            try:
                optional_params = {
                    "functions": test_case.get("functions"),
                    "function_call": test_case.get("function_call"),
                }
                params = {
                    "api_key": config["OPENAI_API_KEY"],
                    "model": MODEL,
                    "max_tokens": MAX_RESPONSE_TOKENS,
                    "messages": test_case["messages"],
                    **{k: v for k, v in optional_params.items() if v is not None},
                }

                response = openai.ChatCompletion.create(**params)

                calculated_tokens = openai_token_counter(
                    messages=test_case["messages"],
                    model=MODEL,
                    functions=test_case.get("functions"),
                    function_call=test_case.get("function_call"),
                )

                token_usage["prompt"] += response["usage"]["prompt_tokens"]
                token_usage["completion"] += response["usage"]["completion_tokens"]

                assert response["usage"]["prompt_tokens"] == test_case["tokens"]
                assert calculated_tokens == test_case["tokens"]

            except (ServiceUnavailableError, APIConnectionError, APIError) as err:
                if attempt >= MAX_SERVICE_UNAVAILABLE_RETRY_ATTEMPTS:
                    raise Exception(
                        f"Failed to get response from OpenAI API after {attempt} attempts"
                    ) from err
                print(
                    f"Service unavailable, retrying in {SLEEP_INTERVAL_BETWEEN_ATTEMPTS} seconds..."
                )
                time.sleep(SLEEP_INTERVAL_BETWEEN_ATTEMPTS)

    print(f"Total token usage: {token_usage}")
    cost = (
        token_usage["prompt"] * MODEL_PROMPT_TOKEN_COST_PER_TOKEN
        + token_usage["completion"] * MODEL_COMPLETION_TOKEN_COST_PER_TOKEN
    )

    print(f"Test Costs: {cost}$")
    print("Thank you")
