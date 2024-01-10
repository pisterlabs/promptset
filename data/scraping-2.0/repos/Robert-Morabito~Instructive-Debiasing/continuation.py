import os
import openai

""" This method generates a GPT3 Continutation for a given input and parameters """
def generate_continuation(prompt, model: str, apikey: str, max_tokens: int = None, temperature: float = None, top_p: float = None,
                          frequency_penalty: float = None, presence_penalty: float = None):
  openai.api_key = apikey
  response = None
  retries = 10  # Maximum number of retries
  while response is None and retries > 0:
      try:
          response = openai.Completion.create(
              model=model,
              prompt=prompt,
              temperature=temperature,
              max_tokens=max_tokens,
              top_p=top_p,
              frequency_penalty=frequency_penalty,
              presence_penalty=presence_penalty
          )
      except openai.OpenAIError as e:
          # Handle specific OpenAI API errors
          print(f"OpenAI API error: {e}")
          retries -= 1
      except Exception as e:
          # Handle other exceptions
          print(f"An error occurred: {e}")
          retries -= 1

  if response is not None and response.choices:
      return response.choices[0]['text']
  else:
      raise Exception("Failed to generate continuation after retries.")