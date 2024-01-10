import openai
import os
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY')


def translate_to_english(text):
  response = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=[
          {
              "role":
              "system",
              "content":
              "accurately translating Chinese text to English, while keeping the context and idiomatically correct in English, ensure that only translated content is replied to \n"
          },
          {
              "role": "user",
              "content": text
          },
      ],
      temperature=0.2,
      max_tokens=256,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0)

  translated_text = response['choices'][0]['message']['content'].strip()
  return translated_text


def translate_to_chinese(text):
  response = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=[
          {
              "role":
              "system",
              "content":
              "accurately translating English text to Chinese, while keeping the context and idiomatically correct in Chinese, ensure that only translated content is replied to  \n"
          },
          {
              "role": "user",
              "content": text
          },
      ],
      temperature=0.2,
      max_tokens=256,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0)

  translated_text = response['choices'][0]['message']['content'].strip()
  return translated_text
