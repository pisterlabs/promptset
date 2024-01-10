import openai
import gradio
 
openai.api_key = "Your API key from openai.com"

messages = [
    {"role": "system", "content": "You are a professional businessman, programmer, psychologist and just smart person"}]


def CustomChatGPT(user_input):
    messages.append({"role": "user", "content": user_input})
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    ChatGPT_reply = response["choices"][0]["message"]["content"]
    messages.append({"role": "assistant", "content": ChatGPT_reply})
    return ChatGPT_reply


demo = gradio.Interface(fn=CustomChatGPT, inputs="text", outputs="text", title="Bot's name")

demo.launch(share=True)
