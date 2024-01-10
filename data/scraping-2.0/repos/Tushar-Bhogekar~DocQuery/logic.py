import os
import gradio as gr
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
import fitz
from PIL import Image
import openai
import nltk
from nltk.corpus import stopwords
import spacy
from spacy.lang.en.stop_words import STOP_WORDS

COUNT, N = 0, 0
chat_history = []
chain = None  # Initialize 
enable_box = gr.Textbox(value=None,
                      placeholder='Upload your OpenAI API key',
                      interactive=True)
disable_box = gr.Textbox(value='OpenAI API key is Set', interactive=False)

# set the OpenAI API key
def set_apikey(api_key):
    os.environ['OPENAI_API_KEY'] = api_key
    return disable_box

# enable the API key input box
def enable_api_box():
    return enable_box

# add text to the chat history
def add_text(history, text):
    if not text:
        raise gr.Error('Enter text')
    history.append((text, ''))
    return history

def extract_text_from_pdf(pdf_file):
    text = ""
    doc = fitz.open(pdf_file)
    for page_num in range(len(doc)):
        page = doc[page_num]
        text += page.get_text()
    return text


from langchain.vectorstores import Chroma

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader

# Modify the preprocessing function in process_file
def process_file(file):
    if 'OPENAI_API_KEY' not in os.environ:
        raise gr.Error('Upload your OpenAI API key')

    loader = PyPDFLoader(file.name)
    documents = loader.load()

    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

    pdfsearch = Chroma.from_documents(documents, embeddings)

    chain = ConversationalRetrievalChain.from_llm(
        ChatOpenAI(temperature=0.5),
        retriever=pdfsearch.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )

    return chain

# Modify the generate_response function to use the advanced preprocessing
def generate_response(history, query, btn):
    global COUNT, N, chat_history, chain

    if not btn:
        raise gr.Error(message='Upload a PDF')
    if COUNT == 0:   # pdf processing
        chain = process_file(btn)
        COUNT += 1

    # Preprocess the user query using the advanced preprocessing function
    preprocessed_query = preprocess_text_advanced(query)

    try:
        result = chain({"question": preprocessed_query, 'chat_history': chat_history}, return_only_outputs=True)
    except Exception as e:
        raise gr.Error(f"Error processing the query: {str(e)}")

    # Extract information from the selected document sections for more informative responses
    selected_sections = result.get('selected_sections', [])
    for section in selected_sections:
        # Process each section and append it to the chat history
        chat_history.append((section['section_title'], section['section_text']))

    N = list(result['source_documents'][0])[1][1]['page']  # page number of doc

    # Add the generated response to the chat history
    chat_history.append((query, result["answer"]))

    # Update the history with the latest information
    history[-1] = (history[-1][0], history[-1][1] + result["answer"])

    # Return the modified history
    yield history, ''

# render a specific page of a PDF file as an image
def render_file(file):
    global N
    doc = fitz.open(file.name)
    page = doc[N]
    pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
    image = Image.frombytes('RGB', [pix.width, pix.height], pix.samples)
    return image

# generate a text summary using ChatGPT API
def generate_summary(text):
    if 'OPENAI_API_KEY' not in os.environ:
        raise gr.Error('Upload your OpenAI API key for summarization')

    openai.api_key = os.environ['OPENAI_API_KEY']
    response = openai.Completion.create(
        engine="davinci",
        prompt=text,
        max_tokens=100
    )
    return response.choices[0].text

# Add a more advanced preprocessing function
def preprocess_text_advanced(text):
    # Load the English language model
    nlp = spacy.load('en_core_web_sm')

    # Convert text to lowercase
    text = text.lower()

    # Tokenize the text into words using spacy
    doc = nlp(text)
    words = [token.text for token in doc if token.text not in STOP_WORDS]

    # Perform more advanced preprocessing steps, such as lemmatization
    lemmatized_words = [token.lemma_ for token in doc]

    # Reconstruct the preprocessed text from the list of words
    preprocessed_text = ' '.join(lemmatized_words)

    return preprocessed_text
