#!/usr/local/bin/python3
import sys

from PIL import Image
import pytesseract


# import openai

# 1. 使用Tesseract OCR提取图片中的文本
def extract_text_from_image(image_path):
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image)
    return text


# 2. 使用ChatGPT处理提取的文本
# def chat_with_gpt(prompt):
#     openai.api_key = 'YOUR_OPENAI_API_KEY'
#     response = openai.Completion.create(
#         engine="text-davinci-002",
#         prompt=prompt,
#         max_tokens=150
#     )
#     return response.choices[0].text.strip()

# 3. 示例
# 从命令行读取图片路径
image_path = sys.argv[1]
image_text = extract_text_from_image(image_path)
# image_text = extract_text_from_image('./img.png')
# gpt_response = chat_with_gpt(image_text)

print("Image Text:", image_text)
# print("GPT Response:", gpt_response)
