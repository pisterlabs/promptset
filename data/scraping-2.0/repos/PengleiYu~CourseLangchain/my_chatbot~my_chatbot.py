import gradio as gr
from langchain.chat_models.openai import ChatOpenAI

_verbose = True
open_ai = ChatOpenAI(verbose=True, model_name='gpt-4-1106-preview', )


def answer(question, history):
    if history is None:
        history = []
    history.append(question)
    response = open_ai.predict(question)
    history.append(response)
    result = [(u, b) for u, b in zip(history[::2], history[1::2])]
    return result, history


with gr.Blocks() as demo:
    chatbot = gr.Chatbot(elem_id='chatbot')
    state = gr.State([])

    with gr.Row():
        text = gr.Textbox(show_label=False, placeholder='Enter text and press enter')
    text.submit(answer, [text, state], [chatbot, state])

# 在本地启动时会卡住主线程，在notebook中则不会
demo.launch(show_error=True, share=False)
