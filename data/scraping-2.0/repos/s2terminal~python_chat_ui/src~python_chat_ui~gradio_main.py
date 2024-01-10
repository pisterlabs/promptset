import gradio
import openai

def generate(input_text: str) -> str:
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": input_text}],
        max_tokens=100,
    )
    return completion["choices"][0]["message"]["content"]

gradio.Interface(
    fn=generate,
    inputs="text",
    outputs="text"
).launch(server_name="0.0.0.0")
