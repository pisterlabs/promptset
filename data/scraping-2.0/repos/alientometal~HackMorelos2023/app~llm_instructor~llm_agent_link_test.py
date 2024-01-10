import os
from typing import Optional, Tuple

import gradio as gr
from langchain.chains import ConversationChain
from langchain.llms import OpenAI
from threading import Lock

from dotenv import load_dotenv

load_dotenv()

print(os.environ["OPENAI_API_KEY"])

def load_chain():
    """Logic for loading the chain"""
    llm = OpenAI(model='text-davinci-003', temperature=0.2)
    chain = ConversationChain(llm=llm)
    return chain


def set_openai_api_key(api_key: str):
    """Set the api key and return chain.

    If no api_key, then None is returned.
    """
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        chain = load_chain()
        os.environ["OPENAI_API_KEY"] = ""
        return chain

class ChatWrapper:

    def __init__(self):
        self.lock = Lock()
    def __call__(
        self, api_key: str, inp: str, history: Optional[Tuple[str, str]], chain: Optional[ConversationChain]
    ):
        """Execute the chat functionality."""
        self.lock.acquire()
        try:
            history = history or []
            # If chain is None, that is because no API key was provided.
            if chain is None:
                history.append((inp, "Please paste your OpenAI key to use"))
                return history, history
            # Set OpenAI key
            import openai
            openai.api_key = api_key
            # Run chain and append input.
            output = chain.run(input=inp)
            history.append((inp, output))
        except Exception as e:
            raise e
        finally:
            self.lock.release()
        return history, history
chat = ChatWrapper()

block = gr.Blocks(css="""
    .gradio-container {
        background: linear-gradient(135deg, #caceceff, #a2ac9fff, #ceecf2ff, #04946bff, #282929ff);
        color: #a2ac9fff;
    }
    h3 {
        color: #04946bff;
    }
""")

with block:
    with gr.Row():
        gr.Markdown("<h3><center>Bonanza Chatbot</center></h3>")

        openai_api_key_textbox = gr.Textbox(
            placeholder="Paste your OpenAI API key (sk-...)",
            show_label=False,
            lines=1,
            type="password",
            css={"background-color": "#ceecf2ff", "border-color": "#a2ac9fff"}
        )

    chatbot = gr.Chatbot()

    with gr.Row():
        message = gr.Textbox(
            label="¿Tienes alguna pregunta?",
            placeholder="¿Qué es mi score crediticio?",
            lines=1,
            css={"background-color": "#ceecf2ff", "border-color": "#a2ac9fff"}
        )
        submit = gr.Button(
            value="Send",
            variant="secondary",
            css={"background-color": "#04946bff", "color": "#ceecf2ff", "border-color": "#04946bff"}
        )

    gr.Examples(
        examples=[
            "¿Qué es el score crediticio?",
            "¿Cómo puedo mejorar mi score crediticio?",
            "¿Qué es un crédito hipotecario?",
        ],
        inputs=message,
    )

    gr.HTML("Aplicación demo de Chatbot Aprende de Finanzas")

    gr.HTML(
        "<center style='color: #a2ac9fff;'>Powered by <a href='https://github.com/hwchase17/langchain' style='color: #04946bff;'>LangChain 🦜️🔗</a></center>"
    )

    state = gr.State()
    agent_state = gr.State()

    submit.click(chat, inputs=[openai_api_key_textbox, message, state, agent_state], outputs=[chatbot, state])
    message.submit(chat, inputs=[openai_api_key_textbox, message, state, agent_state], outputs=[chatbot, state])

    openai_api_key_textbox.change(
        set_openai_api_key,
        inputs=[openai_api_key_textbox],
        outputs=[agent_state],
    )

block.launch(debug=True)