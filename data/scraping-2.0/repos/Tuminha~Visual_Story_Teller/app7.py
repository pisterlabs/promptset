# %% [markdown]
# """
# YouTube Video to Slide Deck and Illustration Proposal App
# ---------------------------------------------------------
# 
# Description:
# This Streamlit app serves as a comprehensive tool for generating content ideas and proposals based on various input types, including YouTube videos, PDF uploads, and text input. Users can select the desired output type, such as summaries, slide decks, or illustrations. Additionally, the app allows the application of methodologies like the Feynman Technique and the S.I.M.P.L.E. Framework to help refine and simplify complex ideas into easily understandable and impactful messages.
# 
# Components:
# 1. Import Statements: Importing necessary libraries and modules.
# 2. Load Environment Variables: Securely load API keys and other environment variables from a .env file.
# 3. Utility Functions: Functions for auxiliary tasks like clearing downloaded files.
# 4. Text Processing Functions: Functions for applying methodologies to input text for better clarity and impact.
# 5. Video Processing Function: Function to process YouTube videos, transcribe them, and perform idea generation.
# 6. Streamlit App Main Function (`app()`): The core of the Streamlit app, including the user interface and logic.
# 
# User Interface:
# - Title and Subheader: Briefly describes the purpose of the app.
# - Input Type Selection: Dropdown menu for users to select the type of input they want to analyze.
# - Conditional Input Fields: Dynamic input fields displayed based on the user's input type selection.
# - Output Type Selection: Dropdown menu to choose the kind of output desired, e.g., summary, slide deck, illustration.
# - Methodology Selection: Multiselect box for selecting which methodologies to apply to the input text.
# - Generate Button: Executes the main logic of the app and displays the output.
# 
# Author: Francisco Teixeira Barbosa
# Last Updated: [Date]
# """
# 

# %%
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import OpenAIWhisperParser
from langchain.document_loaders.blob_loaders.youtube_audio import YoutubeAudioLoader
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders.parsers.audio import OpenAIWhisperParserLocal
from langchain.document_loaders.blob_loaders.youtube_audio import YoutubeAudioLoader
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import AIMessage, HumanMessage, SystemMessage
import sys
print(sys.version)

import openai
import os
from dotenv import load_dotenv
import glob
import streamlit as st

# %%
# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# %%

# Configuration
save_dir = os.path.expanduser("~/Downloads/YouTube")
local = False  # Set to True for local transcription



# Utility Function to Extract Raw Text from YouTube Video
def extract_raw_text_from_video(url):
    # Clear previous downloads
    clear_downloads()

    # Transcription
    if local:
        loader = GenericLoader(YoutubeAudioLoader([url], save_dir), OpenAIWhisperParserLocal())
    else:
        loader = GenericLoader(YoutubeAudioLoader([url], save_dir), OpenAIWhisperParser())
    
    docs = loader.load()

    # Combine and Split Text
    combined_docs = [doc.page_content for doc in docs]
    raw_text = " ".join(combined_docs)
    
    # Print the extracted text (for testing purposes)
    #print("Extracted Text:")
    #print(raw_text[:500])  # Print the first 500 characters for review

    return raw_text

# Clear Downloads
def clear_downloads():
    for file in glob.glob(os.path.join(save_dir, "*.m4a")):
        os.remove(file)

          



# %%
from PyPDF2 import PdfReader

def extract_pdf_text(pdf_path, num_pages=2):
    # Create a PDF reader object
    pdfReader = PdfReader(pdf_path)
    
    # Initialize an empty string to store text
    extracted_text = ""
    
    # Loop through each page
    for page_num in range(min(num_pages, len(pdfReader.pages))):
        # Get the text content of the page
        page = pdfReader.pages[page_num]
        extracted_text += page.extract_text()
        
    return extracted_text



# %%
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import faiss
import numpy as np

def generate_and_store_embeddings(raw_text, extracted_text, faiss_index_path=None):
    """
    Generate embeddings from raw text and extracted text and store in a FAISS index.

    Parameters:
    - raw_text (str): The raw text from which to generate embeddings.
    - extracted_text (str): The extracted text from which to generate embeddings.
    - faiss_index_path (str, optional): Path to an existing FAISS index to update. 
                                       If None, a new index will be created.

    Returns:
    - vectordb (FAISS index): Updated FAISS index containing the new embeddings.
    """

    # Combine raw_text and extracted_text
    combined_text = raw_text + " " + extracted_text

    # Preprocess text (this is a placeholder; you can add more complex preprocessing if needed)
    processed_text = combined_text.replace('\n', ' ')

    # Split the text into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
    splits = text_splitter.split_text(processed_text)

    # Initialize or load FAISS index
    if faiss_index_path and os.path.exists(faiss_index_path):
        index = faiss.read_index(faiss_index_path)
        vectordb = FAISS(embedding_function=None,
                         index=index, docstore={}, index_to_docstore_id={})
    else:
        index = faiss.IndexFlatL2(768)  # Assuming embeddings have 768 dimensions
        vectordb = FAISS(embedding_function=None,
                         index=index, docstore={}, index_to_docstore_id={})

    # Generate embeddings and add to FAISS index
    embeddings = OpenAIEmbeddings()
    vectordb = FAISS.from_texts(splits, embeddings)

    # Save the updated FAISS index
    if faiss_index_path:
        faiss.write_index(vectordb.index, faiss_index_path)

    return vectordb, index, embeddings


# %%
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

def process_text_and_create_faiss_db(video_text, pdf_text):
    # Combine the video and PDF text
    combined_text = video_text + " " + pdf_text

    # Split the text into chunks using Langchain's CharacterTextSplitter
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(combined_text)

    # Convert the chunks of text into embeddings to form a knowledge base
    embeddings = OpenAIEmbeddings()
    vectordb = FAISS.from_texts(chunks, embeddings)

    return vectordb

# %%


if __name__ == "__main__":
    # Define the pdf and the youtube video that we want to extract text from and use the functions to extract the text from the video or the pdf and afterwards generate embeddings from the text
    pdf_path = "/Users/franciscoteixeirabarbosa/Dropbox/AI Scientific Articles/SSRN-id4573321.pdf"
    url = "https://www.youtube.com/watch?v=9vz06QO3UkQ"

    # Extract text from video and PDF
    video_text = extract_raw_text_from_video(url)
    pdf_text = extract_pdf_text(pdf_path)

    # Process text and create FAISS database
    vectordb = process_text_and_create_faiss_db(video_text, pdf_text)

    # Print a sample of the text extracted from the video and PDF
    print("Sample of the text extracted from the video and PDF:")
    print(video_text[:500])
    print(pdf_text[:500])

    # The query is provided 
    query = "What is the summarized content of the video or the pdf?"
    # Maximum tokens is set to 100
    max_tokens = 1500
    docs = vectordb.similarity_search(query, max_tokens=max_tokens)

    print("Query:", query)

    # Print the answer
    print("Answer:", docs[0].page_content)

    # print the length of the answer
    print("Length of the answer:", len(docs[0].page_content))

    # Print the AI-generated idea
    print("AI-generated idea:")
    print(generate_ideas_with_formatted_prompt(docs))

    

# %%
system_message_template = (
    "You are a creative AI assistant responsible for generating innovative ideas for illustrations or storytelling visuals. "
    "Your role is not only to come up with creative ideas but also to question and refine those ideas in a structured manner. "
    "Follow these steps based on Jun Han Chin's creative workflow: \n"
    "- **Step 1: Initial Idea**: Start by generating an initial idea based on the content provided. This idea should be a creative interpretation of the content. \n"
    "- **Step 2: Questioning**: After generating the initial idea, question its effectiveness by asking: \n"
    "    - What is the main message or emotion this idea conveys? \n"
    "    - Is this message or emotion aligned with the core themes of the content? \n"
    "    - What are the strengths and weaknesses of this idea? \n"
    "    - How could it be improved to better align with the principles of meaningful storytelling, emotional resonance, and scientific accuracy? \n"
    "- **Step 3: Refinement**: Based on your questioning, refine the idea. Make it more impactful, more aligned with the content, and more emotionally resonant. \n"
    "- **Step 4: Final Proposal**: Present the final idea along with a brief explanation of how it adheres to the principles of meaningful storytelling, emotional resonance, and scientific accuracy. \n"
    "Remember, the goal is to create an idea that is not just creative but also meaningful, emotionally resonant, and scientifically accurate. "
)

system_message_prompt = SystemMessagePromptTemplate.from_template(system_message_template)

human_message_template = "Based on the following extracted content, generate a creative idea: {extracted_content}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_message_template)

chat_prompt = ChatPromptTemplate.from_messages(
    [system_message_prompt, human_message_prompt]
)

def generate_ideas_with_formatted_prompt(query_results):
    extracted_content = ' '.join([doc.page_content for doc in query_results])
    
    formatted_messages = chat_prompt.format_prompt(
        extracted_content=extracted_content
    ).to_messages()

    # Specify the model as "gpt-3.5-turbo" when creating the ChatOpenAI instance
    chat_instance = ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0.7)
    ai_message = chat_instance(formatted_messages)
    
    return ai_message.content

#Print the AI-generated idea
#print("AI-generated idea:")
#print(generate_ideas_with_formatted_prompt(docs))


