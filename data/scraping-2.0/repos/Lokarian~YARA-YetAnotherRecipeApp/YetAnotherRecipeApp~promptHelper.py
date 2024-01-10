import os

import openai
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_file(template_file):
    with open(template_file, 'r') as file:
        content = file.read()
    return content


def run_text_task(task_name, replace_dict):
    template_file = os.path.join('promptTemplates', task_name + '.txt')
    for key in replace_dict:
        template_file = template_file.replace("{{" + key + "}}", replace_dict[key])
    openai.api_key = os.getenv("OPENAI_API_KEY")
    completion = openai.Completion.create(
        model="text-davinci-003",
        prompt="Say this is a test",
        max_tokens=7,
        temperature=0
    )
    logger.info(completion)
    return completion.choices[0].text


def run_edit_task(task_name, input_text):
    edit_instruction = load_file(os.path.join('promptTemplates', task_name + '.txt'))
    openai.api_key = os.getenv("OPENAI_API_KEY")
    result = openai.Edit.create(
        model="text-davinci-edit-001",
        input=input_text,
        instruction=edit_instruction
    )
    logger.info(result)
    return result.choices[0].text


def run_chat_task(task_name, input_text):
    chat_instruction = load_file(os.path.join('promptTemplates', task_name + '.txt'))
    openai.api_key = os.getenv("OPENAI_API_KEY")
    result = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": chat_instruction},
            {"role": "user", "content": input_text}
        ]
    )
    return result['choices'][0]['message']['content']


def generate_images(prompt, amount, is_base64):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    result = openai.Image.create(
        prompt=prompt,
        n=amount,
        response_format="b64_json" if is_base64 else "url",
        size="1024x1024"
    )

    return result.data
