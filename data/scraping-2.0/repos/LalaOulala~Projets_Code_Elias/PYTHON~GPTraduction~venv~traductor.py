print("Start main.py file\n")

import openai

openai.api_key = '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'

print("Configuration done")

def gptTraduction(prompt, language="anglais"):
  completion = openai.Completion.create(
    engine = "text-davinci-003",
    prompt = "traduit cette phrase ou mot : " + prompt + "dans la langue :" + language,
    max_tokens = 4000
  )

  response = completion.choices[0].text.strip()
  return  response


