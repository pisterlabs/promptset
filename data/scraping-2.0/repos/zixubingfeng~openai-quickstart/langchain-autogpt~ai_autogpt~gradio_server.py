import os
import sys
import gradio as gr
from autogpt import LangchainAutogpt
from utils import LOG


os.environ["SERPAPI_API_KEY"] = "5f4b61bb6543c72fc0a3a7f0a98c1c96b2192d7e"

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
langchainAutogpt = LangchainAutogpt()

def invoke_gpt(input_text):
    return langchainAutogpt.invoke(input_text)

def launch_gradio():

    iface = gr.Interface(
        fn=invoke_gpt,
        title="langchain autogpt",
        inputs="text",
        outputs="text",
        allow_flagging="never"
    )

    iface.launch(share=True, server_name="0.0.0.0")

if __name__ == "__main__":
    LOG.info(os.getenv("OPENAI_API_KEY"))
    LOG.info(os.getenv("SERPAPI_API_KEY"))
    dir  = os.path.dirname(os.path.abspath(__file__))
    LOG.info("启动服务" + dir)
    # 启动 Gradio 服务
    launch_gradio()