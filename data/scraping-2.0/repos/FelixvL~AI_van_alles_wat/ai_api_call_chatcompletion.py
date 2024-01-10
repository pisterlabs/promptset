import os
import openai

#openai.api_key = os.getenv("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")
openai.api_key = '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'
response = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
    {
      "role": "system",
      "content": "Hier is een lijst van ingedienten waar ik een recept van ga voorstellen"
    },
    {
      "role": "user",
      "content": "ik wil niet dat ik heel lang moet koken"
    },
    {
      "role": "system",
      "content": "de ingredienten zijn rijst kip paprika kool en room"
    },
    {
      "role": "user",
      "content": "ik heb ook nog sperciebonen"
    }
  ],
  temperature=0,
  max_tokens=64,
  top_p=1.0,
  frequency_penalty=0.0,
  presence_penalty=0.0
)
print(response)