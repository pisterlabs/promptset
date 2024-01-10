#from langchain.schema import AIMessage, HumanMessage
import gradio as gr
import json
import requests
# functions related to search engine
from search_engine import search_job, search_mathematicians
# functions related to conversation with ollama
from conversation import answer


with gr.Blocks() as demo:

    # THIS IS THE CHATBOT
    # with gr.Tab("Conversation"):
    #    gr.ChatInterface(answer,
    #     chatbot=gr.Chatbot(height=300),
    #     #textbox=gr.Textbox(placeholder="Ask me anything", container=False, scale=7),
    #     title="Conversation Tab",
    #     description="Ask me anything",
    # )

    # THIS IS THE SEARCH ENGINE (through embeddings)
    with gr.Tab("Jobs Search Engine"):
        gr.Interface(
            search_job, 
            inputs=[
                gr.Textbox(label='Describe your dream job', lines=1),
                gr.Number(label='best K results', value=5),
                #gr.Dataframe(label='dataset')
            ],
            outputs=[
                gr.Markdown(label='Results Table'),
                gr.Textbox(label='Results Full Description', lines=15)
            ]
        )

    with gr.Tab("Mathematicians Search Engine"):
        gr.Interface(
            search_mathematicians, 
            inputs=[
                gr.Textbox(label='Find your favourite mathematicians', lines=1),
                gr.Number(label='best K results', value=5),
                #gr.Dataframe(label='dataset')
            ],
            outputs=[
                gr.Textbox(label='The top results!', lines=15)
            ]
        )

if __name__ == "__main__":
    demo.launch(show_api=False)  