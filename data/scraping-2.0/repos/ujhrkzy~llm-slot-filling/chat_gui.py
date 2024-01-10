import gradio as gr
from langchain.chat_models import ChatOpenAI

from app.slot_filling_conversation import SlotFillingConversationChain
from app.slot_memory import SlotMemory

llm = ChatOpenAI(temperature=0.7)
memory = SlotMemory(llm=llm)
chat = SlotFillingConversationChain.create(llm=llm, memory=memory)


def execute_chat(message, history):
    history = history or []
    response = chat.predict_demo(input=message)
    history.append((message, response))
    return history, history


chatbot = gr.Chatbot().style(color_map=("green", "pink"))
demo = gr.Interface(
    execute_chat,
    ["text", "state"],
    [chatbot, "state"],
    allow_flagging="never",
)
if __name__ == "__main__":
    demo.launch()
