import openai
import os
openai.api_key = "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"
def gptfunction(query):
# user_input = input("Enter your input: ")
    

    response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {
            "role": "system",
            "content": "act as strict mentor"
        },
        {
            "role": "user",
            "content": query
        }
        ],
    temperature=1,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )
    
    generated_text = response["choices"][0]["message"]["content"]
    return generated_text
print(gptfunction("hi"))