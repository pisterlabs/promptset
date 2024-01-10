import openai

openai.api_key = "[open-ai-api-key]"

messageHistory = []

def chat(input:str, role = "user"):
    messageHistory.append({"role": role, "content": input})
    completion = openai.ChatCompletion.create(model = "[gptmodel]", messages = messageHistory)
    reply = completion.choices[0].message.content
    messageHistory.append({"role": "assistant", "content": reply})

    return reply