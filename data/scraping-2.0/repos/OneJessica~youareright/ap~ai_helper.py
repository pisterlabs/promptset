import openai
openai.api_key = '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'
# Note: you need to be using OpenAI Python v0.27.0 for the code below to work
import openai

created=openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Who won the world series in 2020?"},
        {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
        {"role": "user", "content": "Where was it played?"}
    ]
)
print(created)
# import openai
# import os
#
# # Set up your API key
# # openai.api_key = os.environ["OPENAI_API_KEY"]
#
# # Generate text using the GPT-3 API
# prompt = "Once upon a time"
# model = "text-davinci-002"
# response = openai.Completion.create(engine=model, prompt=prompt, max_tokens=50)
#
# # Print the generated text
# print(response.choices[0].text)
