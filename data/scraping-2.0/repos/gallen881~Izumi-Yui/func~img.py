from PIL import Image
import pytesseract
import requests
import core.function as function
import os
import re

import openai
openai.api_key = function.open_json('./data/config.json')['token']['openai']


IMG_PATH = './temp/img'
OPENAI_IMG_PATH = './temp/img/openai'
function.auto_mkdir(IMG_PATH[:-4])
function.auto_mkdir(IMG_PATH)
function.auto_mkdir(OPENAI_IMG_PATH)

def ocr(img_url: str, lang: str) -> list:

    with open(f'{IMG_PATH}/temp.png', 'wb') as file:
        file.write(requests.get(img_url, headers={'user-agent': 'Mozilla/5.0'}).content)
    function.print_detail(memo='INFO', obj='Saved temp.png successfully')

    text = pytesseract.image_to_string(Image.open(f'{IMG_PATH}/temp.png'), lang=lang)
    function.print_detail(memo='INFO', obj=f'Form string: "{text}"')

    text_list = [f'```{i}```' for i in function.split_str_to_list(text, 1994)]
    if text_list != ['``````']:
        function.print_detail(memo='INFO', obj='Ocr successfully')
        return text_list
    else:
        function.print_detail(memo='INFO', obj='Character not found')
        return ['***Character not found***']

def rotate(img_url: str, angle: float) -> None:
    with open(f'{IMG_PATH}/temp.png', 'wb') as file:
        file.write(requests.get(img_url, headers={'user-agent': 'Mozilla/5.0'}).content)

    Image.open(f'{IMG_PATH}/temp.png').rotate(angle, expand=1).save(f'{IMG_PATH}/temp.png', quality=100)
    function.print_detail(memo='INFO', obj='Saved temp.png successfully')
    for i in range(95, 0, -5):
        if os.path.getsize(f'{IMG_PATH}/temp.png') >= 8388608:
            function.print_detail(memo='WARN', obj=f'Picture size is larger than 8MB, resave it with quality {i}')
            Image.open(f'{IMG_PATH}/temp.png').rotate(angle, expand=1).save(f'{IMG_PATH}/temp.png', quality=i)
            function.print_detail(memo='INFO', obj='Saved temp.png successfully')
        else:
            break


def generate(prompts: str) -> str:
    response = openai.Image.create(prompt=prompts, n=1, size='1024x1024')
    img_url = response['data'][0]['url']
    function.print_detail(memo='INFO', obj=f'Form image: "{img_url}"\nPrompt: "{prompts}"')
    arranged_prompts = function.arrange_text(prompts)
    with open(f'{OPENAI_IMG_PATH}/{arranged_prompts}.png', 'wb') as file:
        file.write(requests.get(img_url, headers={'user-agent': 'Mozilla/5.0'}).content)

    return f'{OPENAI_IMG_PATH}/{arranged_prompts}.png'
