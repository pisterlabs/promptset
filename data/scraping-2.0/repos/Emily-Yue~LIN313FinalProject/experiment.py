from openai import OpenAI
from setup import prompt, all_messages

api_key = "api_key_here" # your api key here
client = OpenAI(api_key=api_key)

for date, message in all_messages.items():
  # print(message)
  output_filename = './chatgpt_ans/2023/03/'+date+'.json'
  
  completion = client.chat.completions.create(
    model="gpt-3.5-turbo-16k",
    messages=[
      {"role": "system", "content": prompt},
      {"role": "user", "content": message}
    ]
  )
  response = completion.choices[0].message.content
  with open(output_filename, 'w') as outfile:
    outfile.write(response)
