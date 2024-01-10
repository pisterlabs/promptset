import openai
import gradio

openai.api_key = ""  # Enter your API key here

messages = [{"role": "system", "content": "You are a personal assistant"}]


def CustomChatGPT(input):
    messages.append({"role": "user", "content": input})
    response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)
    ChatGPT_reply = response["choices"][0]["message"]["content"]
    messages.append({"role": "assistant", "content": ChatGPT_reply})
    return ChatGPT_reply


demo = gradio.Interface(
    fn=CustomChatGPT, inputs="text", outputs="text", title="Tom's Chatbot"
)

demo.launch(share=True)
