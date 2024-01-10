import openai
import time
import os

openai.api_key = os.getenv('OPENAI_API_KEY')

# Start the timer
start_time = time.time()

question = ""
language = ""

completion = openai.ChatCompletion.create(
  model = "gpt-3.5-turbo",
  temperature = 0.8,
  max_tokens = 2000,
  messages = [
    {"role": "system", "content": "You are the best coding evaluator, you are an specialist in" + language + " programming. You are going to receive a solution made by a candidate for this question: " + question + ". Give a score out of 10 for different metrics: readibility , decisive , creativity and average score .Write and explanation of your evaluation and show what you consider a perfect answer. Give an complex score ( different parameters ) and explanation of your evaluation in a json format." },
    {"role": "user", "content": ""}
  ]
)

print(completion.choices[0].message.content)