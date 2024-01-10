import gradio as gr
from dotenv import load_dotenv

from func_ai.utils.llm_tools import OpenAIInterface
from func_ai.utils.openapi_function_parser import OpenAPISpecOpenAIWrapper

_chat_message = []

_spec = None


def add_text(history, text):
    global _chat_message
    history = history + [(text, None)]
    _chat_message.append(_spec.api_qa(text, max_tokens=500))
    return history, ""


def add_file(history, file):
    history = history + [((file.name,), None)]
    return history


def bot(history):
    global _chat_message
    # print(temp_callback_handler.get_output())
    # response = temp_callback_handler.get_output()['output']
    history[-1][1] = _chat_message[-1]
    return history


with gr.Blocks() as demo:
    chatbot = gr.Chatbot([], elem_id="chatbot").style(height=1500)

    with gr.Row():
        with gr.Column(scale=1):
            txt = gr.Textbox(
                show_label=False,
                placeholder="Enter text and press enter",
            ).style(container=False)
    txt.submit(add_text, [chatbot, txt], [chatbot, txt]).then(
        bot, chatbot, chatbot
    )

if __name__ == "__main__":
    load_dotenv()
    _spec = OpenAPISpecOpenAIWrapper.from_url('http://petstore.swagger.io/v2/swagger.json',
                                              llm_interface=OpenAIInterface(), index=True)
    demo.launch()
