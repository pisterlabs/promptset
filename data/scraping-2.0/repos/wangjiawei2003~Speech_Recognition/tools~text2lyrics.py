# tools/text2lyrics_module.py
from openai import OpenAI

def text2lyrics(api_key):
    client = OpenAI(api_key=api_key)

    # 设置一些默认提示
    default_prompts = [
        "You are a helpful assistant who can write song lyrics.",
        "You are a knowledgeable assistant who can answer questions about science.",
        "You are a witty assistant who can engage in amusing conversations."
    ]

    # 从默认提示列表中选择一个提示
    init_prompt = default_prompts[0]  # 或者可以随机选择一个提示

    user_input = input("You: ")
    response = assistant(client, init_prompt, user_input)
    return response

def assistant(client, init_prompt, user_input):
    messages = [
        {"role": "system", "content": init_prompt},
        {"role": "user", "content": user_input}
    ]

    completion = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        temperature=0.7,
    )

    return completion.choices[0].message.content
