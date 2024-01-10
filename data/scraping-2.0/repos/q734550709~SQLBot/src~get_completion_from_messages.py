import openai

#对话函数
def get_completion_from_messages(
    messages,
    model="gpt-3.5-turbo-16k",
    temperature=0,
    max_tokens=3000):

    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message["content"]
