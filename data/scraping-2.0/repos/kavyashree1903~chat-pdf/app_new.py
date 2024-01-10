import streamlit as st
import openai
import requests
from bs4 import BeautifulSoup
from PIL import Image
#import clip
import chardet
import io
import json
from tabula.io import read_pdf
import tabula
import os
import pandas as pd
from PIL import Image
import requests
# from transformers import CLIPProcessor, CLIPModel
# import aspect_based_sentiment_analysis as absa
# import json
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain import OpenAI, VectorDBQA
from langchain.document_loaders import DirectoryLoader
# import magic
import os
# import nltk
from pathlib import Path

from langchain.vectorstores.faiss import FAISS
from langchain.chains.question_answering import load_qa_chain

from PyPDF2 import PdfReader
import pickle
from streamlit_chat import message


# Set up OpenAI API key
os.environ["OPENAI_API_KEY"] = "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"
openai.api_key = "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"

# Define function to generate GPT-3 comments
def generate_comment(code):
    endpoint = "https://api.openai.com/v1/completions"
    headers = {"Content-Type": "application/json",
               "Authorization": f"Bearer {openai.api_key}"}
    data = {
        "model": "text-davinci-003",
        "prompt": f"Copy the code and add comments to it explaining it step by step:{code}",
        "temperature": 0.5,
        "max_tokens": 500,
        "n": 1,
    }
    response = requests.post(endpoint, json=data, headers=headers)
    if response.ok:
        try:
            comment = response.json()["choices"][0]["text"].strip()
            return comment
        except KeyError:
            return "Failed to generate comments. Please try again later."
    else:
        return "Failed to generate comments. Please try again later."


# Define function to scrape website data
def scrape_website():
    endpoint = "https://api.openai.com/v1/completions"
    headers = {"Content-Type": "application/json",
               "Authorization": f"Bearer {openai.api_key}"}
    url = "https://ioagpl.com/cng-price/"
    page = requests.get(url)
    soup = BeautifulSoup(page.content, "html.parser")
    table = soup.find_all("table")
    table = table[:2]
    data = {
        "model": "text-davinci-003",
        "prompt": (
            f"Find out all locations and their corresponding prices in a pairwise format:\n\n"
            f"{table}\n\n"
        ),
        "temperature": 0.5,
        "max_tokens": 500,
        "n": 1,
    }
    response = requests.post(endpoint, json=data, headers=headers)
    scraped = response.json()["choices"][0]["text"].strip()
    # split the scraped data into location and price pairs
    data_pairs = scraped.split("\n")
    return data_pairs

# Define function to perform aspect-based sentiment analysis
def aspect_sentiment(text):
    endpoint = "https://api.openai.com/v1/engines/text-davinci-002/completions"
    headers = {"Content-Type": "application/json",
               "Authorization": f"Bearer {openai.api_key}"}
    data = {
        "model": "text-davinci-002",
        "prompt": (
            f"Analyze the sentiment of the following text from the perspective of the customer service aspect:\n\n"
            f"{text}\n\n"
            f"Sentiment: Positive, Negative, Neutral"
        ),
        "temperature": 0.5,
        "max_tokens": 60,
        "n": 1,
        "stop": "\n"
    }
    response = requests.post(endpoint, json=data, headers=headers)
    sentiment = response.json()["choices"][0]["text"].strip()
    return sentiment

# Define email generator
def generate_email(recipient, content):
    endpoint = "https://api.openai.com/v1/completions"
    headers = {"Content-Type": "application/json",
               "Authorization": f"Bearer {openai.api_key}"}
    prompt = f"Compose an email to {recipient} with the following context:\n\n{content}\n\n---\n"
    data = {
        "model": "text-davinci-003",
        "prompt": prompt,
        "temperature": 0.5,
        "max_tokens": 400,
        "n": 1,
    }
    response = requests.post(endpoint, json=data, headers=headers)
    content = response.json()["choices"][0]["text"].strip()
    return content


def func(filename):
    if(filename!=None):
        print(filename)
        reader = PdfReader(filename)
        
        # printing number of pages in pdf file
        pdf_len = len(reader.pages)
        
        # getting a specific page from the pdf file
        final_text=''

        final_list=list()

        for i in range(pdf_len):
                page = reader.pages[i]
                text = page.extract_text()
                final = text.replace("\n"," ")
                final_text=final_text+text

                final_list.append(final)
        

        
        # extracting text from page

        new_list = list(filter(lambda x: x != '', final_list))
        # print(new_list)
        # print(len(new_list))
        return new_list
    
    
def newList(filename):
    new_list=func(filename)
    embeddings = OpenAIEmbeddings()

    return new_list,embeddings




def chatur_gpt(filename):
    new_list,embeddings= newList(filename)
    if(new_list!=None):

        if(len(new_list)!=0):

            docsearch = FAISS.from_texts(new_list, embeddings)
            qa_chain = load_qa_chain(OpenAI(temperature=0), chain_type="refine")
            qa = VectorDBQA(combine_documents_chain=qa_chain, vectorstore=docsearch)
        
    return qa
    
    
#ABSA
# nlp = absa.load()
# # Load the aspect categories and their synonyms from CSV files
# aspects_synonyms = {}
# aspects = ["Housekeeping", "Location", "Quality", "Quantity", "Service"]
# for aspect in aspects:
#     filename = aspect + ".xlsx"
#     df = pd.read_excel(filename)
#     synonyms = df["Category"].tolist()
#     aspects_synonyms[aspect] = synonyms

# # Function to find synonyms for a word
# def find_synonyms(word):
#     prompt = f"Find synonyms for the word '{word}'.\nSynonyms:"
#     response = openai.Completion.create(
#         engine="text-davinci-003",
#         prompt=prompt,
#         max_tokens=2048,
#         temperature=0.5,
#         n=1,
#         stop=None,
#     )
#     synonyms = [s.strip() for syn in response.choices[0].text.split("\n") for s in syn.split(",")]
#     synonyms = [word.title() for word in synonyms]
#     return synonyms

# # Function to find the sentiment for an aspect in a review
# def find_aspect_sentiment(review, aspect):
#     aspect_list = aspects_synonyms[aspect]
#     var = nlp(review, aspects=aspect_list)
#     sentiment_score = var.subtasks[aspect].scores
#     sentiment = "Positive" if sentiment_score[2] > sentiment_score[1] else "Negative" if sentiment_score[1] > sentiment_score[0] else "Neutral"
#     return sentiment

# # Function to calculate the score of a review for a given aspect
# def calculate_score(review, aspect):
#     aspect_list = aspects_synonyms[aspect]
#     weight_list = [2 if syn == aspect else 1 for syn in aspect_list]
#     data = {"Category": aspect_list, "Weight": weight_list}
#     total_score = 0
#     weight_sum = 0
#     final_score = 0
#     for j in range(len(data)):
#         aspect_stem_position = data["Category"][j].lower() in review.lower().split()
#         synonyms = aspects_synonyms[aspect]
#         synonym_find = False 
#         if aspect_stem_position == False:
#             if (len(synonyms)>0):
#                 synonym_find = any(synonym in review.lower() for synonym in synonyms)
#         else:
#             synonym_find = True
#         if synonym_find:
#             sentiment_score = find_aspect_sentiment(review, data["Category"][j])
#             score = (sentiment_score[2]-sentiment_score[1]+1)*2.5*data["Weight"][j]
#             total_score = total_score + score
#             weight_sum = weight_sum + data["Weight"][j]
#     if weight_sum > 0:
#         final_score = total_score / weight_sum
#     return final_score

# # Define function to classify image using OpenAI CLIP
# def classify_image(image):
#     model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
#     processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

#     inputs = processor(text=["a photo of a clean toilet", "a photo of a dirty toilet"], images=image, return_tensors="pt", padding=True)

#     outputs = model(**inputs)
#     logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
#     probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
#     return probs
def generate_response(prompt,qa):
#     completions = openai.Completion.create(
#         engine = "text-davinci-003",
#         prompt = prompt,
#         max_tokens = 1024,
#         n = 1,
#         stop = None,
#         temperature=0.5,
#     )
#     message = completions.choices[0].text

    message = qa.run(prompt)
    return message

def get_text():
    input_text = st.text_input("You: ","Hello, how are you?", key="input")
    return input_text

def main():
    st.sidebar.title("Options")
    options = ["Document Summarizer", "Code Commenter", "Web Scraper", "Email Generator", "Custom GPT"]
    # options = ["Document Summarizer", "Code Commenter", "Web Scraper", "Email Generator", "Aspect-based Sentiment Analysis"]

    choice = st.sidebar.selectbox("Select an option", options)
    
    if choice == "Custom GPT":
        st.title("Custom GPT")
        st.write("Upload a file to train using GPT-3")
        file = st.file_uploader("Upload a file", type=["pdf"])
        
        # Storing the chat
        if 'generated' not in st.session_state:
            st.session_state['generated'] = []

        if 'past' not in st.session_state:
            st.session_state['past'] = []
        
        if file is not None:
            if os.path.isfile(file.name) == False:
                save_folder = os.getcwd()
                save_path = Path(save_folder, file.name)
                with open(save_path, mode='wb') as w:
                    w.write(file.getbuffer())
            #st.write(file.read())
            new_list,embeddings= newList(file.name)
            if(new_list!=None):

                if(len(new_list)!=0):
                    docsearch = FAISS.from_texts(new_list, embeddings)
                    qa_chain = load_qa_chain(OpenAI(temperature=0), chain_type="refine")
                    qa = VectorDBQA(combine_documents_chain=qa_chain, vectorstore=docsearch)
#             st.write(file.name)
#             res = chatur_gpt
                    user_input = get_text()

                    if user_input:
                        output = generate_response(user_input,qa)
                        # store the output 
                        st.session_state.past.append(user_input)
                        st.session_state.generated.append(output)

                    if st.session_state['generated']:
                        for i in range(len(st.session_state['generated'])-1, -1, -1):
                            message(st.session_state["generated"][i], key=str(i))
                            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')

        
    
    if choice == "Document Summarizer":
        st.title("Document Summarizer")
        st.write("Upload a file to summarize using GPT-3")
        file = st.file_uploader("Upload a file", type=["txt"])
        if file is not None:
            # Read the file content
            file_content = file.read()
            st.write(file_content)
            result = chardet.detect(file_content)
            encoding = result['encoding'] if result['encoding'] is not None else 'utf-8'
            text = io.StringIO(file_content.decode(encoding)).read()
            # Split the text into sections of 2048 tokens or less
            max_tokens_per_section = 2048
            sections = []
            section_start = 0
            while section_start < len(text):
                section_end = section_start + max_tokens_per_section
                section = text[section_start:section_end]
                sections.append(section)
                section_start = section_end

            # Summarize each section separately
            summaries = []
            with st.spinner("Summarizing..."):
                for section in sections:
                    response = openai.Completion.create(
                        engine="text-davinci-002",
                        prompt=(
                            "Summarize the following document section in detail and also explain the highlights available from the tables and output the response in bullets:\n\n"
                            + section
                            + "\n\nSummary:"
                        ),
                        max_tokens=2048,
                        n=1,
                        stop=None,
                        temperature=0.5,
                    )
                    summary = response.choices[0].text.strip()
                    summaries.append(summary)

            # Combine the summaries into a single summary
            summary = "\n".join(summaries)

            st.write("Summary:")
            st.write(summary)

    # elif choice == "Report Summarizer":
    #     st.title("Report Summarizer")
    #     st.write("Preview the report and generate its summary using GPT-3")
    #     filename = st.file_uploader("Upload a file", type=["txt", "pdf"])
    #     df = tabula.read_pdf(filename, pages="all")
    #     df.to_excel("example.xlsx", index=False)
    #     # filename = "example.pdf"
    #     if os.path.exists(filename):
    #         # File exists, so read or convert it
    #         df = tabula.read_pdf(filename, pages="all")
    #         # or
    #         tabula.convert_into(filename, "example.xlsx", output_format="xlsx", pages="all")
    #     else:
    #         print(f"File {filename} does not exist")
    

    elif choice == "Web Scraper":
        st.title("Web Scraper")
        st.write("Get data of CNG prices from IOAGPL website")
        st.write("Here is the [Link to IOAGPL website](https://ioagpl.com/cng-price/), click on the link to compare it with your scrape results" )

        # url = st.text_input("URL")
        # field = st.text_input("HTML field")
        if st.button("Scrape"):
            with st.spinner("Scraping website..."):
                data = scrape_website()
                # Extracting the information from scraped data
            st.write("Data: ")
            st.write(data)

    elif choice == "Email Generator":
        st.title("Email Generator")
        st.write("Fill the fields to generate a customized email")
        r = st.text_input("Who are you sending this mail to?")
        field = st.text_input("What is the context of this mail?")
        if st.button("Generate Email"):
            with st.spinner("Generating email..."):
                email_content = generate_email(r, field)
            st.write("Email Content:")
            st.write(email_content)
            scraped_data = scrape_website()

    elif choice == "Aspect-based Sentiment Analysis":
        st.title("Aspect-Based Sentiment Analysis")
        review = st.text_area("Enter your review here:")
        aspect = st.selectbox("Select an aspect:", aspects)
        if review and aspect:
                score = calculate_score(review, aspect)
                st.write(f"The score of your review for {aspect} is {score:.2f}")
        else:
                st.write("Please enter a review and select an aspect.")

    elif choice == "Code Commenter":
        st.title("Code Commenter")
        col1, col2 = st.columns(2)
        with col1:
            st.write("Enter a piece of code to comment using GPT-3")
            code = st.text_area("Paste your code here")
        with col2:
            if code:
                comments = generate_comment(code)
                st.write("Comments:")
                st.code(comments, language="python")

    elif choice == "Image Classifier":
        st.title("Image Classifier")
        st.write("Upload an image to classify using OpenAI CLIP")
        file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if file is not None:
            image = Image.open(file)
            with st.spinner("Classifying image..."):
                category = classify_image(image)
            st.write("Image corresponds to:")
            st.write(category)

main()