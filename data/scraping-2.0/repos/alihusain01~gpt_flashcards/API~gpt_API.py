import os
import openai
import server.authentication as authentication
from pdf_parser import parse_pdf

# API_KEY: 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
# API_KEY (test key 2): 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
# openai.organization = "org-pnHGCI36uk7HqG8vEwp4HcfP"

def make_response(prompt):
  openai.api_key = authentication.API_KEY

  response = openai.ChatCompletion.create(
              model="gpt-3.5-turbo",
              messages = [{"role": "user", 
                          "content": "Make flashcards from the following text: \n" + prompt + "\n"}],
              temperature = 0
          )

  return response.choices[0].message.content

make_response(parse_pdf())

