import openai

def result(inputt):
  openai.api_key = "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"
  gpt_prompt = inputt


  response = openai.Completion.create(
    engine="text-davinci-002",
    prompt=gpt_prompt,
    temperature=0.5,
    max_tokens=256,
    top_p=1.0,
    frequency_penalty=0.0,
    presence_penalty=0.0
  )
  return response['choices'][0]['text']
