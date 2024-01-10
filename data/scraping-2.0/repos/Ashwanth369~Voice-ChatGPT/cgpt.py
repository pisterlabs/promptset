import openai 

client = openai.OpenAI(api_key = "OPEN_API_KEY")

def sendToGPT(data):
    completion = client.chat.completions.create(
      model="gpt-3.5-turbo",
      messages=[
        {"role": "user", "content": data}
        ]
    )    
    return completion.choices[0].message.content
