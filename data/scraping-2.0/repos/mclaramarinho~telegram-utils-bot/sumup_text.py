import os
import requests
import openai

openai.api_key = os.getenv("OPEN_AI_API_TOKEN")


def sumup_text(text):
  response = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=[
          {
              "role":
              "system",
              "content":
              "Your role is to sum up the text sent by the user. You need to sum the text up to 1 paragraph."
          },
          {
              "role": "user",
              "content": text
          },
      ],
      temperature=0.8,
      max_tokens=100,
      top_p=0.5,
  )

  return response["choices"][0]["message"]["content"]
