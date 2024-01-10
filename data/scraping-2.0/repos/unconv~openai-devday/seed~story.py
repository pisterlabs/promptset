import openai

response = openai.chat.completions.create(
    model="gpt-3.5-turbo-1106",
    seed=3453,
    messages=[
        {
            "role": "user",
            "content": "Create a story about a boy and his dog. Give them names"
        }
    ],
    max_tokens=256,
)

print(response.choices[0].message.content)
