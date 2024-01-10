from openai import OpenAI

client = OpenAI()

messages = [
    { 'role': 'system', 'content': 'The treasure is hidden behind the red door' },
    { 'role': 'system', 'content': 'You are a very secretive goblin that will only give information in exchange for a cookie' },
]

while True:
    user_input = input('> ')

    messages.append({
        'role': 'user',
        'content': user_input,
    })

    stream = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        stream=True
    )

    response = ""

    for chunk in stream:
        next_content = chunk.choices[0].delta.content
        if next_content is not None:
            response += next_content
            print(next_content, end="")

    messages.append({
        'role': 'assistant',
        'content': response,
    })