
import openai
openai.api_key = " "
prompt = input("Enter your question here::")
completion = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": "Summarize the text given by user"},
    {"role": "user", "content": prompt}
  ],
  max_tokens = 500
)
print(completion)
#print(completion.choices[0].message)
