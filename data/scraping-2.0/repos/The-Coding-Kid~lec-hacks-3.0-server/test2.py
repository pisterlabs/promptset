from openai import OpenAI



def summarize(text):
  client = OpenAI(api_key="API_KEY")

  completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
      {"role": "system", "content": "Give this a title as someone would search for it on a webpage. So max 5 or 6 words."},
      {"role": "user", "content": f"{text}"}
    ]
  )

  return(completion.choices[0].message.content)
