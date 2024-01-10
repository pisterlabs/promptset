import os
from openai import OpenAI
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY")
)

def aiResponse():
  completion = client.chat.completions.create(
      model="gpt-3.5-turbo",
      # prompt= "tell me about... lagos"
      messages=[
      {"role": "system", "content": f"You are only to answer food and grocery related question! "},
      {"role": "user", "content": "what is the best carbohydrate food in africa? if this question  don't pertains to food and grocery please don't answer"}
    ]
  )
  # print(completion['choices'][0]['message']['content'])
  return str(completion.choices[0].message.content)

# print(completion.choices)
# print(completion.choices[0].message)