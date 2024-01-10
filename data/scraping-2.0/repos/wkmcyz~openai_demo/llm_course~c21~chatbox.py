import openai, os
import gradio as gr
from langchain import OpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chat_models import ChatOpenAI

openai.api_key = os.environ["OPENAI_API_KEY"]

if __name__ == '__main__':
    memory = ConversationSummaryBufferMemory(llm=ChatOpenAI(), max_token_limit=2048)
    conversation = ConversationChain(
        llm=OpenAI(max_tokens=2048, temperature=0.5),
        memory=memory,
    )


    def predict(input, history=[]):
        history.append(input)
        response = conversation.predict(input=input)
        history.append(response)
        responses = [(u, b) for u, b in zip(history[::2], history[1::2])]
        return responses, history


    with gr.Blocks(css="#chatbot{height:800px} .overflow-y-auto{height:800px}") as demo:
        chatbot = gr.Chatbot(elem_id="chatbot")
        state = gr.State([])

        with gr.Row():
            txt = gr.Textbox(show_label=False, placeholder="Enter text and press enter").style(container=False)

        txt.submit(predict, [txt, state], [chatbot, state])

    demo.launch()
