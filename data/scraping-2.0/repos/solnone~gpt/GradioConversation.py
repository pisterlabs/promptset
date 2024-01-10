# https://gradio.app/
# https://python.langchain.com/en/latest/modules/memory/types/buffer_window.html

from langchain.prompts import (
    ChatPromptTemplate, 
    MessagesPlaceholder, 
    SystemMessagePromptTemplate, 
    HumanMessagePromptTemplate
)
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from dotenv import load_dotenv
import gradio as gr
import os

load_dotenv()

def create_conversation(openai_api_key, model_name, system_message, temperature, k):
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_message),
        MessagesPlaceholder(variable_name="history"),
        HumanMessagePromptTemplate.from_template("{input}")
    ])
    llm = ChatOpenAI(model_name=model_name, temperature=temperature, openai_api_key=openai_api_key)
    # memory = ConversationBufferMemory(return_messages=True)
    memory = ConversationBufferWindowMemory(k=k, return_messages=True)
    conversation = ConversationChain(memory=memory, prompt=prompt, llm=llm, verbose=True)
    return conversation

def respond(message, chat_history, conversation, openai_api_key, model_name, system_message, temperature, k):
    if (len(message) > 0):
        if (conversation is None):
            conversation = create_conversation(openai_api_key, model_name, system_message, temperature, int(k))

        bot_message = conversation.predict(input=message)
        chat_history.append((message, bot_message))

    return "", chat_history, conversation

def clear_memory_history(chatbot, conversation):
    chatbot.clear()
    if (conversation is not None):
        conversation.memory.chat_memory.messages.clear()

    return chatbot

def remove_conversation(state, chatbot):
    chatbot.clear()
    return state, chatbot, None

with gr.Blocks() as demo:
    gr.Markdown("# Conversation Buffer Window Memory")
    conversation = gr.State()
    openai_api_key = gr.Textbox(label="OPENAI API KEY", value=os.environ["OPENAI_API_KEY"], placeholder="Paste your OpenAI API key (sk-...) and hit Enter", lines=1, type='password')
    model_name = gr.Radio(["gpt-3.5-turbo", "gpt-4"], value="gpt-3.5-turbo", label="Model")
    system_message = gr.Textbox(label="System Message", value="You are a helpful assistant.")
    temperature = gr.Slider(label="Temperature", value=0.7, minimum=0, maximum=1, step=0.1)
    k = gr.Slider(label="keep the last {k} interactions in memory", value=2, minimum=0, maximum=15, step=1)
    chatbot = gr.Chatbot(label="ChatGPT")
    msg = gr.Textbox(label="Enter your message", placeholder="Send a message")
    clear = gr.Button("Clear")

    openai_api_key.submit(remove_conversation, inputs=[openai_api_key, chatbot], outputs=[openai_api_key, chatbot, conversation])
    model_name.change(remove_conversation, inputs=[model_name, chatbot], outputs=[model_name, chatbot, conversation])
    system_message.submit(remove_conversation, inputs=[system_message, chatbot], outputs=[system_message, chatbot, conversation])
    temperature.change(remove_conversation, inputs=[temperature, chatbot], outputs=[temperature, chatbot, conversation])
    k.change(remove_conversation, inputs=[k, chatbot], outputs=[k, chatbot, conversation])
    msg.submit(respond, [msg, chatbot, conversation, openai_api_key, model_name, system_message, temperature, k], [msg, chatbot, conversation])
    clear.click(clear_memory_history, [chatbot, conversation], [chatbot])

demo.launch(show_error=True, debug=True)