# ruff: noqa: E501
import textwrap
from typing import Any
from typing import Optional

import requests
from tenacity import retry
from tenacity import stop_after_attempt
from tenacity import wait_random_exponential

import danswer.bots.zendesk_ask_compute.constants as constants
from danswer.bots.zendesk_ask_compute.logger import setup_logger

logger = setup_logger(constants.MODULE_NAME, constants.LOG_LEVEL)


@retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(3))
def chat_completion_request(
    endpoint: str,
    api_key: str,
    model: str,
    messages: list,
    temperature: Optional[float] = None,
    functions: Optional[list] = None,
    function_call: Optional[dict] = None,
) -> dict:
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "ask-compute-bot",
        "Authorization": "Bearer " + api_key,
    }
    json_data: dict[str, Any] = {"model": model, "messages": messages}
    if temperature is not None:
        json_data.update({"temperature": temperature})
    if functions is not None:
        json_data.update({"functions": functions})
    if function_call is not None:
        json_data.update({"function_call": function_call})
    try:
        response = requests.post(endpoint, headers=headers, json=json_data)
        return response.json()
    except Exception as e:
        logger.error("Unable to generate ChatCompletion response")
        logger.error(f"Exception: {e}")
        raise e


def get_thread_summary_prompt(messages: list) -> list:
    """
    Get gpt prompt for summarising the content of a thread
    """

    gpt_instruction = """You are a slack bot specialized in creating structured summary documents based on Slack thread discussions.
    Each message in the thread will be provided to you in a separate prompt, starting with the author's identification in the following format: (author): message.
    I will let you know when you should start generating the summary.
    When generating the summary, you always check if there is a question/issue being discussed in the thread.
    If the thread doesn't cover a question or an issue, skip the summary.
    Otherwise, follow below rules to generate the summary documents:
        - Always format your answer in Markdown. Ignore syntax highlighting for code blocks. Link should be in this format: [Page title](https://zendesk.com).
        - When referring to authors, always enclose their identifiers within angle brackets, like so: <@ZZZZZZZZZ>.
        - The document should have following content
            - Title: Summarize the main issue or question discussed in the thread.
            - Description: Provide a clear and concise description of the original question or issue as stated by the person who started the thread, along with any additional context or details provided initially.
            - Discussion Summary: Summarize the key points, clarifications, and additional details discussed in the thread.
            - Identified Solutions: Detail the solutions, answers, or resolutions provided in the thread, including the steps taken or recommended to resolve the issue or answer the question.
            - Takeways: List insights, best practices, or lessons learned from the discussion and any preventive measures or recommendations discussed to avoid similar issues in the future.
            - References: Provide links to any external resources, documentation, or tools mentioned in the discussion and any attachments or additional materials shared during the discussion. This value shoud start with 'The following resources were referred to during the discussion:', followed by a new line character and a markdown list of links.
        - Include any relevant snippets of code, configuration, or error messages shared during the discussion."""

    # Insert system prompt to set the tone of chat completions
    prompt = [{"role": "system", "content": textwrap.dedent(gpt_instruction).strip()}]
    # Provide thread messages
    for msg in messages:
        prompt.append(
            {"role": "user", "content": f"(<@{msg.get('user')}>): ${msg.get('text')}"}
        )
    # Append prompt to summarise the thread content into a technical document
    prompt.append(
        {
            "role": "user",
            "content": "Please follow the rules outlined in the system message to generate the summary document.",
        }
    )
    return prompt


def get_thread_summary(
    messages: list, model_version: str, model_temperature: float
) -> dict:
    # Prepare prompt for chat completions
    prompt = get_thread_summary_prompt(messages)
    # Describe functions for gpt to call
    # The model will reply in JSON, allowing us to more reliably get structured data
    functions = [
        {
            "name": "summarise_thread",
            "parameters": {
                "type": "object",
                "properties": {
                    "is_issue": {
                        "type": "boolean",
                        "description": "Whether the thread involves an issue",
                    },
                    "title": {
                        "type": "string",
                        "description": "Summarize the main issue or question discussed in the thread.",
                    },
                    "description": {
                        "type": "string",
                        "description": "Provide a clear and concise description of the original question or issue as stated by the person who started the thread, along with any additional context or details provided initially.",
                    },
                    "discussion_summary": {
                        "type": "string",
                        "description": "Summarize the key points, clarifications, and additional details discussed in the thread, including any relevant snippets of code, configuration, or error messages shared during the discussion.",
                    },
                    "identified_solutions": {
                        "type": "string",
                        "description": "Detail the solutions, answers, or resolutions provided in the thread, including the steps taken or recommended to resolve the issue or answer the question.",
                    },
                    "takeways": {
                        "type": "string",
                        "description": "Optional, List insights, best practices, or lessons learned from the discussion and any preventive measures or recommendations discussed to avoid similar issues in the future.",
                    },
                    "references": {
                        "type": "string",
                        "description": "Optional, Provide links to any external resources, documentation, or tools mentioned in the discussion and any attachments or additional materials shared during the discussion.",
                    },
                },
                "required": ["is_issue"],
            },
        }
    ]
    logger.debug("Generating thread summary via OpenAI...")
    res = chat_completion_request(
        endpoint=constants.OPENAI_ENDPOINT,
        api_key=constants.OPENAI_KEY,
        model=model_version,
        messages=prompt,
        temperature=model_temperature,
        functions=functions,
        function_call={"name": "summarise_thread"},
    )
    logger.debug("Response from OpenAI %s", res)
    return res
