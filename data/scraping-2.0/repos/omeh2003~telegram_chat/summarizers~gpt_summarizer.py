import openai


def summarize_text(text):
    model = "gpt-4"
    messages = [{"role": "user", "content": text}]
    completion = openai.ChatCompletion.create(model=model, messages=messages)
    response = completion.choices[0].message.content
    return response
