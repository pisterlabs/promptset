import os
from openai import OpenAI
import json
import config
import utils

API_KEY = os.environ['OPENAI_API_KEY']
client = OpenAI(api_key=API_KEY)


def call_llm(prompt: str, system_prompt: dict[str, str | bool], temp=1):
    """Call the LLM API with the given prompt and system prompt. Expects system prompt to be a dict with the following keys:
    - prompt: the prompt to use
    - returns_json: whether the response is JSON
    - json_key: the key to use to extract the JSON
    """
    utils.print_verbose(
        f"Calling LLM with prompt: {prompt}\nSystem prompt: {json.dumps(system_prompt, indent=2)}")

    if config.dry_run:
        print("Dry run mode is enabled. Returning mock response.")
        if system_prompt["returns_json"]:
            json_response = json.loads(system_prompt["mock_response"])
            if system_prompt.get("json_key") is not None:
                return json_response[system_prompt["json_key"]]
            return json_response
        else:
            return system_prompt["mock_response"]

    response = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        response_format={
            "type": "json_object" if system_prompt["returns_json"] else "text"},
        messages=[
            {"role": "system", "content": system_prompt["prompt"]},
            {"role": "user", "content": prompt}
        ],
        temperature=temp,
        max_tokens=4096
    )

    if system_prompt["returns_json"]:
        utils.print_verbose(
            f"LLM response: {json.dumps(response.choices[0].message.content, indent=2)}")
        json_response = json.loads(response.choices[0].message.content)
        if system_prompt.get("json_key") is not None:
            return json_response[system_prompt["json_key"]]
        return json_response

    else:
        utils.print_verbose(
            f"LLM response: {response.choices[0].message.content}")
        return response.choices[0].message.content


def call_vision(context: str, url: str):
    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Describe this image for an alt text in my website. Make sure to make it concise. The image is about {context} Reply with only the alt text."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": url
                        },
                    },
                ],
            }
        ],
        max_tokens=300,
    )
    return response.choices[0].message.content
