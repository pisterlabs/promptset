# This code is for v1 of the openai package: pypi.org/project/openai
import openai

openai.api_key = ""


response = openai.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
    {
      "role": "system",
      "content": "Jesteś moim asystentem testera "
    },
    {
      "role": "user",
      "content": "Pomożesz mi stworzyć opis defektu"
    }
  ],
  temperature=1,
  max_tokens=256,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0
)

print(response.choices[0].message.content)