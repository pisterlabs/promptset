import openai


def chat(input):
    messages = [
        {"role":"system", "content": "You are a helpful, upbeat and funny assistant"},
        {"role": "user", "content": input}
    ]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0,
    )

    return response.choices[0].message["content"]


question = "What is the fastest land mamal?"

prompt = """
Be very sarcastic when answering questions
Question: {question}
""".format(
    question=question
)

print(prompt)
answer = chat(prompt)
print(answer)

# https://help.openai.com/en/articles/7042661-chatgpt-api-transition-guide
