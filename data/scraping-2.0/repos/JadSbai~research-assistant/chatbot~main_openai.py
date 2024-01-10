from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage
import gradio as gr
import requests


llm = ChatOpenAI(model='gpt-4-1106-preview', openai_api_key="")

def search_documents(text, number_documents=20):
    url = "https://eed8-2a00-23c6-54e7-2c01-ddd6-167e-8696-b759.ngrok-free.app/search"
    payload = {
        "text": text,
        "number_documents": number_documents
    }
    response = requests.get(url, json=payload)
    return response.json()

def generate_prompt(message, documents):
    final_prompt='You are a scientific research bot. Below are paragraphs from papers and a question from a user. Combined the knowledge from the paragraphs to answer the question.\n'

    for document in documents:
        final_prompt += '###Paragraph:\n'
        final_prompt += document['paragraph_text'] + '\n'

    final_prompt += "###Question:\n"
    final_prompt += message
    final_prompt += "###Answer:\n"

    return final_prompt


def predict(message, history):
    history_langchain_format = []
    for human, ai in history:
        history_langchain_format.append(HumanMessage(content=human))
        history_langchain_format.append(AIMessage(content=ai))
    documents = search_documents(message)
    prompt = generate_prompt(message, documents)
    history_langchain_format.append(HumanMessage(content=prompt))
    gpt_response = llm(history_langchain_format)
    return gpt_response.content


def upload_file(files):
    endpoint_url = 'https://458b-2a00-23c6-54e7-2c01-2420-508e-6159-a35b.ngrok-free.app/upload_and_process_pdf'
    with open(files, 'rb') as pdf_file:
        files = {'file': (files, pdf_file, 'application/pdf')}
        data = {'title': 'title', 'author': 'author'}

        # Make a POST request to the endpoint
        response = requests.post(endpoint_url, files=files, data=data)

        # Check if the request was successful
        if response.status_code == 200:
            print("PDF successfully uploaded and processed.")
            return
        else:
            print("Failed to upload the PDF. Status Code:", response.status_code)
            return None

def fetch_and_process(theme):
    url = "https://458b-2a00-23c6-54e7-2c01-2420-508e-6159-a35b.ngrok-free.app/fetch_and_process_arxiv"

    # Data to be sent (form data)
    data = {'theme': theme}

    requests.post(url, data=data)

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    iface = gr.ChatInterface(predict ,title="Research Assistant", stop_btn=None, retry_btn=None, clear_btn=None, undo_btn=None)
    with gr.Row():
        with gr.Column():
            file_output = gr.File(label='Select Paper to add', render=True)
            upload_button = gr.UploadButton("Upload a Paper", file_types=["pdf"])
            upload_button.upload(upload_file, upload_button, file_output)
        with gr.Column():
            text_input = gr.Textbox(label="Type a research theme")
            submit_button = gr.Button("Retrieve papers")
            submit_button.click(fetch_and_process, inputs=text_input)

demo.launch()