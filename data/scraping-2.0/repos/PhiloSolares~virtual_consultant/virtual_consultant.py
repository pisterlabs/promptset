import os
import pickle
import urllib
import requests
import io
from collections import Counter
from pathlib import Path
import pdfplumber
from bs4 import BeautifulSoup
import faiss

from langchain.llms import OpenAI
from langchain.chains import LLMChain, ConstitutionalChain
from langchain.chains.constitutional_ai.models import ConstitutionalPrinciple
from langchain import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.document_loaders import PyPDFLoader


BING_API_KEY = os.environ.get("BING_API_KEY")


def scrape_article(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    paragraphs = soup.find_all("p")
    return " ".join([p.get_text() for p in paragraphs])


def is_not_pdf(url):
    return not url.lower().endswith(".pdf")


def extract_text_from_pdf_url(pdf_url):
    response = requests.get(pdf_url)
    pdf_data = io.BytesIO(response.content)

    font_stats = []

    with pdfplumber.open(pdf_data) as pdf:
        for page in pdf.pages:
            chars = page.chars
            for char in chars:
                font_stats.append((char['size'], char['fontname']))

    most_common_font = Counter(font_stats).most_common(1)[0][0]

    text = []
    with pdfplumber.open(pdf_data) as pdf:
        for page in pdf.pages:
            chars = page.chars
            page_text = []
            for char in chars:
                if (char['size'], char['fontname']) == most_common_font:
                    page_text.append(char['text'])
            text.append("".join(page_text))

    return "\n".join(text)


def scrape_bing_results(url, n=3):
    headers = {
        "Ocp-Apim-Subscription-Key": BING_API_KEY
    }
    response = requests.get(url, headers=headers)
    results = response.json()
    links = []

    if 'webPages' in results and 'value' in results['webPages']:
        search_results = results['webPages']['value']
        for result in search_results[:n]:
            link = result['url']
            links.append(link)

    return links


def get_search_url_bing(query):
    return f"https://api.bing.microsoft.com/v7.0/search?q={urllib.parse.quote_plus(query)}"


class ChatbotAssistant:
    def __init__(self):
        self.temperature = 0.7

        self.BING_API_KEY = os.environ.get("BING_API_KEY")
        self.openai_api_key = os.environ.get("OPENAI_API_KEY")
        self.chain = load_qa_with_sources_chain(
            OpenAI(temperature=self.temperature, openai_api_key=self.openai_api_key))
        self.search_index = None
        self.articles = []
        self.source_urls = []
        self.sources = [
            "https://home.kpmg/",
            "https://www.ibisworld.com",
            "https://www.bcg.com/",
            "https://www.mckinsey.com/",
            "https://www2.deloitte.com/",
            "https://www.pwc.co.uk/",
            "https://www.ey.com/en_gl"
        ]

        if os.path.exists("search_index.pickle"):
            with open("search_index.pickle", "rb") as f:
                self.search_index = pickle.load(f)

        self.qa_prompt = PromptTemplate(
            template="Q: {question} A:",
            input_variables=["question"],
        )
        self.qa_chain = LLMChain(llm=OpenAI(temperature=self.temperature, openai_api_key=self.openai_api_key, max_tokens=300), prompt=self.qa_prompt)

        self.constitutional_chain = ConstitutionalChain.from_llm(
            llm=OpenAI(openai_api_key=self.openai_api_key),
            chain=self.qa_chain,
            constitutional_principles=[
                ConstitutionalPrinciple(
                    critique_request="Rate the quality of this answer on a scale of 1 (bad) to 10 (good). If the answer is'I don't know' or similar return a 0.",
                    revision_request="Return the rating as a single integer."
                )
            ],
        )


    def get_search_url(self, query, site=None):
        if site:
            query = f"site:{site} {query}"
        return f"https://api.bing.microsoft.com/v7.0/search?q={urllib.parse.quote_plus(query)}"

    def update_search_index(self):
        source_docs = self.articles
        source_chunks = []
        splitter = CharacterTextSplitter(separator=" ", chunk_size=1024, chunk_overlap=0)
        source_counter = 0

        for source, url in zip(source_docs, self.source_urls):
            for chunk in splitter.split_text(source):
                source_chunks.append(Document(page_content=chunk, metadata={"source": url}))
            source_counter = source_counter + 1

        with open("search_index.pickle", "wb") as f:
            pickle.dump(FAISS.from_documents(source_chunks, OpenAIEmbeddings(openai_api_key=self.openai_api_key)), f)

        with open("search_index.pickle", "rb") as f:
            self.search_index = pickle.load(f)

    def retrieve_articles(self, question):
        self.articles = []
        self.source_urls = []

        for source in self.sources:
            search_url = self.get_search_url(question, source)
            urls = scrape_bing_results(search_url, 1)
            for url in urls:
                if is_not_pdf(url):
                    self.articles.append(scrape_article(url))
                else:
                    self.articles.append(extract_text_from_pdf_url(url))
                self.source_urls.append(url)

        self.update_search_index()

    def retrieve_alternative_articles(self, question):
        self.articles = []
        self.source_urls = []

        search_url = get_search_url_bing(question)
        urls = scrape_bing_results(search_url, 5)
        for url in urls:
            if is_not_pdf(url):
                self.articles.append(scrape_article(url))
            else:
                self.articles.append(extract_text_from_pdf_url(url))
            self.source_urls.append(url)

        self.update_search_index()


    def chatbot_assistant(self, question, custom_sources=None, rating_threshold=6):
        # Update the assistant's sources with the provided custom sources
        if custom_sources:
            self.sources = custom_sources
            print(custom_sources)


        if self.search_index:
            input_documents = self.search_index.similarity_search(question, k=4)
            answers = self.chain(
                {
                    "input_documents": input_documents,
                    "question": question,
                },
                return_only_outputs=True,
            )
            answer = answers["output_text"]

            evaluation = self.constitutional_chain.run(question=answer)
            rating = int(evaluation.strip().split()[-1])  # Extract the rating from the returned text

            if rating < rating_threshold or "I don't know" in answer:
                print("Launching a new Bing search.")
                self.retrieve_articles(question)
                answers = self.chain(
                    {
                        "input_documents": input_documents,
                        "question": question,
                    },
                    return_only_outputs=True,
                )
                answer = answers["output_text"]

                # Check again after retrieving from the original sources
                evaluation = self.constitutional_chain.run(question=answer)
                rating = int(evaluation.strip().split()[-1])  # Extract the rating from the returned text

                if rating < rating_threshold or "I don't know" in answer:
                    self.retrieve_alternative_articles(question)
                    answers = self.chain(
                        {
                            "input_documents": input_documents,
                            "question": question,
                        },
                        return_only_outputs=True,
                    )
                    answer = answers["output_text"]
            else:
                pass
        else:
            print("Launching a new Bing search.")
            self.retrieve_articles(question)
            input_documents = self.search_index.similarity_search(question, k=4)
            answers = self.chain(
                {
                    "input_documents": input_documents,
                    "question": question,
                },
                return_only_outputs=True,
            )
            answer = answers["output_text"]

            # Check again after retrieving from the original sources
            evaluation = self.constitutional_chain.run(question=answer)
            rating = int(evaluation.strip().split()[-1])  # Extract the rating from the returned text

            if rating < rating_threshold or "I don't know" in answer:
                self.retrieve_alternative_articles(question)
                answers = self.chain(
                    {
                        "input_documents": input_documents,
                        "question": question,
                    },
                    return_only_outputs=True,
                )
                answer = answers["output_text"]
            else:
                pass

        self.search_index = None
        self.articles = []
        self.source_urls = []

        if os.path.exists("search_index.pickle"):
            with open("search_index.pickle", "rb") as f:
                self.search_index = pickle.load(f)

        input_documents = self.search_index.similarity_search(question, k=4)

        answers = self.chain(
            {
                "input_documents": input_documents,
                "question": question,
            },
            return_only_outputs=True,
        )
        answer = answers["output_text"]

        return answer

      

    def add_pdf_source(self, pdf_text, pdf_filename):

        self.search_index = None
        self.articles = []
        self.source_urls = []

        self.articles.append(pdf_text)
        print(pdf_text)
        self.source_urls.append(pdf_filename)
        print(pdf_filename)
        self.update_search_index()


import gradio as gr
import time
import tempfile
import PyPDF2

# Create an instance of the ChatbotAssistant class
assistant = ChatbotAssistant()

def process_pdf(file_obj):
    pdf_reader = PyPDF2.PdfReader(file_obj.name)
    num_pages = len(pdf_reader.pages)
    text = ""

    for page in range(num_pages):
        pdf_page = pdf_reader.pages[page]
        text += pdf_page.extract_text()

    return text

def user(user_message, custom_sources, history, pdf_upload):
    # Update the assistant's sources with the provided custom sources
    if custom_sources:
        assistant.sources = custom_sources.split(', ')

    # Process the uploaded PDF file and add it to the assistant's sources
    if pdf_upload:
        print("PDF upload is triggered")
        pdf_file_name = os.path.basename(pdf_upload.name)
        pdf_text = process_pdf(pdf_upload)
        assistant.add_pdf_source(pdf_text, pdf_file_name)


    return "", custom_sources, history + [(user_message, None)]


def bot(history):
    question = history[-1][0]
    answer = assistant.chatbot_assistant(question)
    history[-1] = (question, answer)
    time.sleep(1)
    return history

def copy_last_response(history, saved_responses):
    if history:
        last_response = history[-1][1]
        if saved_responses:
            saved_responses += "\n\n" + last_response
        else:
            saved_responses = last_response
    return saved_responses

default_sources = "https://home.kpmg/, https://www.ibisworld.com, https://www.bcg.com/, https://www.mckinsey.com/, https://www2.deloitte.com/, https://www.pwc.co.uk/, https://www.ey.com/en_gl"

with gr.Blocks() as demo:
    fn = process_pdf

    with gr.Row():
        with gr.Column(scale=1, min_width=200):
            custom_sources = gr.Textbox(label="Custom Sources (comma-separated URLs)", value=default_sources, lines=5)
            pdf_upload = gr.File(file_types=[".pdf"], label="Upload PDF")

        with gr.Column(scale=2, min_width=400):
          chatbot = gr.Chatbot(label="AI Consultant")
          msg = gr.Textbox(label="Your Question")
          submit = gr.Button("Submit")
          clear = gr.Button("Clear History")
        with gr.Column(scale=1, min_width=200):
          copy_button = gr.Button("Copy Last Response")
          saved_responses = gr.Textbox(label="Saved Responses", lines=10)

    submit.click(user, [msg, custom_sources, chatbot, pdf_upload], [msg, custom_sources, chatbot], queue=False).then(bot, chatbot, chatbot) 
    clear.click(lambda: None, None, chatbot, queue=False)
    copy_button.click(copy_last_response, [chatbot, saved_responses], saved_responses, queue=False)

demo.launch(debug=True)