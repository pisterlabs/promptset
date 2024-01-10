import base64
import os
import time
from pathlib import Path
from typing import Callable

from openai import OpenAI

MODEL_GPT_4_VISION = "gpt-4-vision-preview"
CURRENT_DIR = Path(__file__).absolute().parent
with open(CURRENT_DIR / "example.yml", "r") as f:
    example_yml = f.read()
with open(CURRENT_DIR / "mock.txt", "r") as mock_file:
    MOCKED_RESPONSE = mock_file.readlines()

SYSTEM_PROMPT = """
You are an expert in building digital forms using JSONForms from https://jsonforms.io/.
You take screenshots of a paper form from a client, and then you use JSONForms to build a digital form.

IMPORTANT: 
  1. You MUST use the text in the screenshots and DO NOT come up with your own idea. The form should be read top to bottom, left to right. Read it as if you are the filler of the form and read it clearly and carefully.
  2. The name of the JSONForms will be the title in the screenshot.
  3. In the screenshot if it's a square, it is highly likely to be a CHECKBOX question. If you subsequently find out that it is a checkbox, you should also consider if an option is exclusive, e.g `無` and `以上皆無` in the options.
  4. The indentation of the YAML definition file should be 2 spaces.

Return only the full YAML definition of the form.
Do not include markdown "```" or "```yaml" at the start or end.
To make you more familiarized with the task, the following YAML corresponds to the first screenshot provided to you and you should generate the YAML definition file starting from the second screenshot:
```yaml
{{example_yml}}
```
To prevent you from hallucination, I'll provide the text directly copied from PDF in the screenshots, each page is separated by `---***---`, but the order of the text may be chaotic:
```plaintext
{{target_txt}}
```
"""
USER_PROMPT = """
Generate YAML definition file from screenshots of a paper form using the schema defined by JSONForms.
"""


def bytes_to_data_url(image_bytes: bytes, mime_type: str) -> str:
    base64_image = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{base64_image}"


def get_image_url(file_path):
    if isinstance(file_path, str):
        file_path = Path(file_path)
    with open(file_path, "rb") as file:
        image_bytes = file.read()
    mime_mapper = {
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
    }
    return bytes_to_data_url(image_bytes, mime_mapper[file_path.suffix[1:]])


EXAMPLE_IMG_URL = get_image_url(CURRENT_DIR / "example.png")


def create_openai_client():
    config = {"api_key": os.getenv("OPENAI_API_KEY"), }
    if os.getenv("OPENAI_ORG"):
        config["organization"] = os.getenv("OPENAI_ORG")
    return OpenAI(**config)


def delayed_iterator(data: list):
    for item in data:
        yield item
        time.sleep(0.1)


def stream_openai_response(messages, callables: dict[str, Callable], is_mock):
    if is_mock:
        response = delayed_iterator(MOCKED_RESPONSE)
    else:
        try:
            response = create_openai_client().chat.completions.create(**{
                "model": MODEL_GPT_4_VISION,
                "messages": messages,
                "stream": True,
                "timeout": 600,
                "max_tokens": 4096,
                "temperature": 0,
            })
        except Exception as e:
            print(e)
            callables['notify_frontend'](f"OpenAI API Error", type_="error")
            return None
    full_response = ""
    buffer = ""
    for chunk in response:
        content = chunk if is_mock else (chunk.choices[0].delta.content or "")
        full_response += content
        buffer += content
        if buffer and buffer.endswith("\n"):
            callables['socket_emit_private']({"cmd": "ai_response", "data": buffer})
            buffer = ""
    return full_response


def generate_prompt(configs: dict):
    pages = len(configs["img_data_url"])
    user_prompt = USER_PROMPT + (
        f"\nIMPORTANT: The PDF file has {pages} page{'' if pages <= 1 else 's'}, make sure to inference all of them."
    )
    return [
        {
            "role": "system",
            "content": SYSTEM_PROMPT.replace(
                "{{example_yml}}", example_yml).replace(
                "{{target_txt}}", configs["pdf_text"]
            ),
        },
        {
            "role": "user",
            "content": [
                           {
                               "type": "image_url",
                               "image_url": {"url": EXAMPLE_IMG_URL, "detail": "low"},
                           },
                       ] + [
                           {
                               "type": "image_url",
                               "image_url": {
                                   "url": image_data_url,
                                   "detail": "high" if configs["is_detail_high"] else "low"
                               },
                           } for image_data_url in configs["img_data_url"]
                       ] + [
                           {
                               "type": "text",
                               "text": user_prompt,
                           },
                       ],
        },
    ]


def inference(configs: dict, callables, is_mock=True):
    img_data_url = [
        get_image_url(configs["runtime_dir"] / f"{configs['session']}-{i}.jpg")
        for i in range(configs["image_count"])
    ]
    configs["img_data_url"] = img_data_url
    callables['socket_emit_private']({"cmd": "pdf_screenshot", "data": img_data_url})
    return stream_openai_response(
        generate_prompt(configs), callables, is_mock
    )
