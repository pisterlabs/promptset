import openai
import time
import sys
sys.path.append('/Users/busterblackledge/')
from keys import openai_API_key

openai.api_key = openai_API_key
start_time = time.time()
response = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
        {"role": "system", "content": "What is Arsenal Football club?"},
    ]
)
end_time = time.time()

print(response['choices'][0]['message']['content'])
print(f"Execution Time: {end_time - start_time}")

