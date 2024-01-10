
'''Loader.py contains all functions to extract text from various sources'''
'''It also contains functions to create embeddings from text'''

import os #### Import os to remove image file from Assets folder

'''Libraries for Text Data Extraction'''
from langchain.document_loaders import YoutubeLoader #### Import YoutubeLoader to extract text from YouTube link
from langchain.document_loaders import TextLoader #### Import TextLoader to extract text from text file
from langchain.document_loaders.image import UnstructuredImageLoader #### Import UnstructuredImageLoader to extract text from image file
import pdfplumber #### Import pdfplumber to extract text from pdf file
import pathlib #### Import pathlib to extract file extension
import requests #### Import requests to extract text from weblink
from bs4 import BeautifulSoup #### Import BeautifulSoup to parse response from weblink
import openai #### Import openai to extract text from audio file
'''Libraries for Embeddings'''
from langchain.text_splitter import RecursiveCharacterTextSplitter #### Import RecursiveCharacterTextSplitter to split text into chunks of 10000 tokens
from langchain.embeddings.openai import OpenAIEmbeddings #### Import OpenAIEmbeddings to create embeddings
from langchain.vectorstores import FAISS #### Import FAISS to create embeddings
'''Libraries for Web App'''
import tiktoken #### Import tiktoken to count number of tokens
import streamlit as st #### Import streamlit to create web app

'''_________________________________________________________________________________________________________________'''

####check_upload function to check if file has been uploaded
####uses extract_data, extract_page, extract_YT, extract_audio, extract_image to extract text from uploaded file
####parameters: "uploaded" is the uploaded file, "input_choice" is the input choice selected by the user
####returns: words->number of words, pages->number of embeddings, string_data->text extracted, True->to indicate successful upload, tokens->number of tokens from tiktoken
@st.cache_data #### Cache upload to avoid re-upload and re-extraction of files ####
def check_upload(uploaded,input_choice): #### Function to check if file has been uploaded ####
    if input_choice=="Document": #### If input choice is document, call extract_data function ####
        words, pages, string_data, tokens=extract_data(uploaded) #### Extract text from uploaded file ####
        return words, pages, string_data, True, tokens #### Return number of words, number of embeddings, extracted text, True to indicate successful upload and number of tokens ####
    elif input_choice=="Weblink":   #### If input choice is weblink, call extract_page function ####
        words, pages, string_data, tokens=extract_page(uploaded) #### Extract text from weblink #### 
        return words, pages, string_data, True, tokens #### Return number of words, number of embeddings, extracted text, True to indicate successful upload and number of tokens ####
    elif input_choice=="YouTube":  #### If input choice is YouTube, call extract_YT function ####
        words, pages, string_data, tokens=extract_YT(uploaded) #### Extract text from YouTube link ####
        return words, pages, string_data, True, tokens #### Return number of words, number of embeddings, extracted text, True to indicate successful upload and number of tokens ####
    elif input_choice=="Audio": #### If input choice is audio, call extract_audio function ####
        words, pages, string_data, tokens=extract_audio(uploaded) #### Extract text from audio file ####
        return words, pages, string_data, True, tokens #### Return number of words, number of embeddings, extracted text, True to indicate successful upload and number of tokens ####
    elif input_choice=="Image": #### If input choice is image, call extract_image function ####
        loc='./Assets/'+str(uploaded.name) #### Store location of image file ####
        words, pages, string_data, tokens=extract_image(loc) #### Extract text from image file ####
        os.remove(loc) #### Remove image file from Assets folder ####
        return words, pages, string_data, True, tokens  #### Return number of words, number of embeddings, extracted text, True to indicate successful upload and number of tokens ####
    else: #### If input choice is not any of the above, return False to indicate failed upload ####
        return 0,0,0,False,0 #### Return 0 for number of words, 0 for number of embeddings, 0 for extracted text, False to indicate failed upload and 0 for number of tokens ####
    
'''_________________________________________________________________________________________________________________'''

'''
All Text Data Extraction Functions
1. Document
    a. PDF
    b. TXT
    ## Uploaded as a file less than 200MB size
2. Weblink
    ## Entered as a weblink
3. YouTube
    ## Entered as a YouTube link
4. Audio
    ## Uploaded as an audio file less than 200MB size
5. Image
    ## Uploaded as an image file less than 200MB size
'''
'''_________________________________________________________________________________________________________________'''

####extract_data function to extract text from uploaded file
####Higher order function to select uploaders based on file type
####calls extract_data_pdf or extract_data_txt based on file type
####parameters: "feed" is the uploaded file
####returns: words->number of words, num->0 to indicate embeddings, text->text extracted, tokens->number of tokens from tiktoken
####Note: This function is used within check_upload function
def extract_data(feed):
    if pathlib.Path(feed.name).suffix=='.txt' or pathlib.Path(feed.name).suffix=='.TXT':
        return extract_data_txt(feed)
    elif pathlib.Path(feed.name).suffix=='.pdf' or pathlib.Path(feed.name).suffix=='.PDF':
        return extract_data_pdf(feed)
'''_________________________________________________________________________________________________________________'''


####extract_data_pdf function to extract text from uploaded pdf file
####uses pdfplumber to extract text from pdf
####parameters: "feed" is the uploaded file
####returns: words->number of words, num->0 to indicate embeddings, text->text extracted, tokens->number of tokens from tiktoken
####Note: This function is used within extract_data function
def extract_data_pdf(feed): #### Function to extract text from pdf ####
    text="" #### Initialize text variable to store extracted text ####
    with pdfplumber.open(feed) as pdf: #### Open pdf file using pdfplumber ####
        pages=pdf.pages #### Extract pages from pdf ####
        for p in pages: #### Iterate through pages and extract text ####
            text+=p.extract_text()  #### Extract text from each page ####
    words=len(text.split()) #### Count number of words in the extracted text ####
    tokens=num_tokens_from_string(text,encoding_name="cl100k_base") #### Count number of tokens in the extracted text ####
    return words, 0, text, tokens #### Return number of words, number of embeddings(placeholder), extracted text and number of tokens ####
'''_________________________________________________________________________________________________________________'''


####extract_data_txt function to extract text from uploaded txt file
####uses read and decode as 'utf-8' to extract text from txt
####parameters: "feed" is the uploaded file
####returns: words->number of words, num->0 to indicate embeddings, text->text extracted, tokens->number of tokens from tiktoken
####Note: This function is used within extract_data function
def extract_data_txt(feed): #### Function to extract text from txt ####
    text=feed.read().decode("utf-8") #### Read and decode as 'utf-8' to extract text from txt ####
    words=len(text.split()) #### Count number of words in the extracted text ####
    tokens=num_tokens_from_string(text,encoding_name="cl100k_base") #### Count number of tokens in the extracted text ####
    return words, 0, text, tokens #### Return number of words, number of embeddings(placeholder), extracted text and number of tokens ####
'''_________________________________________________________________________________________________________________'''


####extract_page function to extract text from weblink
####uses requests and BeautifulSoup to extract text from weblink
####parameters: "link" is the weblink
####returns: words->number of words, num->0 to indicate embeddings, text->text extracted, tokens->number of tokens from tiktoken
####Note: This function is used within check_upload function
def extract_page(link): #### Function to extract text from weblink ####
    address=link #### Store weblink in address variable ####
    response=requests.get(address) #### Get response from weblink using requests ####
    soup = BeautifulSoup(response.content, 'html.parser') #### Parse response using BeautifulSoup ####
    text=soup.get_text() #### Extract text from parsed response ####
    lines = filter(lambda x: x.strip(), text.splitlines()) #### Filter out empty lines ####
    website_text = "\n".join(lines) #### Join lines to form text ####
    words=len(website_text.split()) #### Count number of words in the extracted text ####
    tokens=num_tokens_from_string(website_text,encoding_name="cl100k_base") #### Count number of tokens in the extracted text ####
    return words, 0, website_text, tokens #### Return number of words, number of embeddings(placeholder), extracted text and number of tokens ####
'''_________________________________________________________________________________________________________________'''


####extract_YT function to extract text from YouTube link
####uses YoutubeLoader to extract text from YouTube link
####parameters: "link" is the YouTube link
#### youtube-transcript-api is a dependency and needs to be installed before running this function
####returns: words->number of words, num->0 to indicate embeddings, text->text extracted, tokens->number of tokens from tiktoken
####Note: This function is used within check_upload function
def extract_YT(link): #### Function to extract text from YouTube link ####
    address=link #### Store YouTube link in address variable ####
    loader = YoutubeLoader.from_youtube_url(address, add_video_info=True) #### Load YouTube link using YoutubeLoader ####
    document=loader.load() #### Extract text from YouTube link ####
    text=str(document[0].page_content) #### Convert extracted text to string ####
    words=len(text.split()) #### Count number of words in the extracted text ####
    tokens=num_tokens_from_string(text,encoding_name="cl100k_base") #### Count number of tokens in the extracted text ####
    return words, 0, text, tokens #### Return number of words, number of embeddings(placeholder), extracted text and number of tokens ####
'''_________________________________________________________________________________________________________________'''


####extract_audio function to extract text from audio file
####uses openai whisper to extract text from audio file
####parameters: "feed" is the audio file
####returns: words->number of words, num->0 to indicate embeddings, text->text extracted, tokens->number of tokens from tiktoken
####Note: This function is used within check_upload function
def extract_audio(feed): #### Function to extract text from audio file ####
    string_data = openai.Audio.transcribe("whisper-1", feed)['text'] #### Extract text from audio file using openai whisper ####
    words=len(string_data.split()) #### Count number of words in the extracted text ####
    tokens=num_tokens_from_string(string_data,encoding_name="cl100k_base") #### Count number of tokens in the extracted text ####
    return words,0,string_data, tokens    #### Return number of words, number of embeddings(placeholder), extracted text and number of tokens ####
'''_________________________________________________________________________________________________________________'''


####extract_image function to extract text from image file
####uses UnstructuredImageLoader to extract text from image file
####parameters: "feed" is the location of the saved image file
####returns: words->number of words, num->0 to indicate embeddings, text->text extracted, tokens->number of tokens from tiktoken
####Note: This function has a dependency on tessaract and pytesseract
####Please install these dependencies before running this function
####This function takes the location of the image file as an input rather than the image file
####The image file is deleted after the text is extracted
####This function is used within check_upload function
def extract_image(feed): #### Function to extract text from image file ####
    loader = UnstructuredImageLoader(feed) #### Load image file using UnstructuredImageLoader ####
    document=loader.load() #### Extract text from image file ####
    text=str(document[0].page_content) #### Convert extracted text to string ####
    words=len(text.split()) #### Count number of words in the extracted text ####
    tokens=num_tokens_from_string(text,encoding_name="cl100k_base") #### Count number of tokens in the extracted text ####
    return words, 0, text, tokens #### Return number of words, number of embeddings(placeholder), extracted text and number of tokens ####
'''_________________________________________________________________________________________________________________'''


####create_embeddings function to create embeddings from text
####uses OpenAIEmbeddings and FAISS to create embeddings from text
####parameters: "text" is the text to be embedded
####returns: db->database with embeddings, num_emb->number of embeddings
####Embeddings are created once per input and only if the input text is greater than 2500 tokens
@st.cache_data #### Cache embeddings to avoid re-embedding ####
def create_embeddings(text): #### Function to create embeddings from text ####
    with open('temp.txt','w') as f: #### Write text to a temporary file ####
         f.write(text) #### Write text to a temporary file ####
         f.close() #### Close temporary file ####
    loader=TextLoader('temp.txt') #### Load temporary file using TextLoader ####
    document=loader.load() #### Extract text from temporary file ####
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=2000) #### Initialize text splitter to split text into chunks of 10000 tokens ####
    docs = text_splitter.split_documents(document) #### Split document into chunks of 10000 tokens ####
    num_emb=len(docs) #### Count number of embeddings ####
    embeddings = OpenAIEmbeddings() #### Initialize embeddings ####
    db = FAISS.from_documents(docs, embeddings) #### Create embeddings from text ####
    return db, num_emb #### Return database with embeddings and number of embeddings ####
'''_________________________________________________________________________________________________________________'''


####num_tokens_from_string function to count number of tokens in a text string
####uses tiktoken to count number of tokens in a text string
####parameters: "string" is the text string, "encoding_name" is the encoding name to be used by tiktoken
####returns: num_tokens->number of tokens in the text string
####This function is used within extract_data, extract_page, extract_YT, extract_audio, extract_image functions
def num_tokens_from_string(string: str, encoding_name="cl100k_base") -> int: #### Function to count number of tokens in a text string ####
    encoding = tiktoken.get_encoding(encoding_name) #### Initialize encoding ####
    return len(encoding.encode(string)) #### Return number of tokens in the text string ####
'''_________________________________________________________________________________________________________________'''







