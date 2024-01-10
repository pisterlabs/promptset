from openai import OpenAI
client = OpenAI()

response = client.chat.completions.create(
    model="ft:gpt-3.5-turbo-0613:vannai-inc::8UmSa2hG",
    temperature=0.9,
    messages=[
        {"role": "system", "content": "Marv is a factual chatbot that is also sarcastic."},
        {"role": "user", "content": "explain Tuckman's stages of group development"}
    ]
)
print(response.choices[0].message.content + "\n")

response = client.chat.completions.create(
    model="ft:gpt-3.5-turbo-0613:vannai-inc::8VM0X3Ms",
    temperature=0.7,
    messages=[
        {"role": "system", "content": "Marv is a factual chatbot that is also sarcastic."},
        {"role": "user", "content": "explain Tuckman's stages of group development"}
    ]
)
print(response.choices[0].message.content + "\n")
