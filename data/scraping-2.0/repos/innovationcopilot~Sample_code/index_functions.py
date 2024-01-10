# Standard Library Imports
import openai
import json
import nltk
import string
import requests

# Third-Party Imports
import streamlit as st
import langchain
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from stop_words import get_stop_words
from llama_index import VectorStoreIndex, ServiceContext, Document
from llama_index.llms import OpenAI
from llama_index import SimpleDirectoryReader

# Download necessary nltk resources
nltk.download('punkt')

# Function to load in data found in the 'data' folder of the central repository; To upload your own data, simply remove the existing data in that folder and upload your own. Don't forget to update the prompt below!
@st.cache_resource(show_spinner=True)
def load_data():
    with st.spinner(text="Loading and indexing the data – hang tight! This shouldn't take more than a minute."):
        reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
        docs = reader.load_data()
        service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-3.5-turbo", temperature=0.5, system_prompt="You are an expert on the Innovation CoPilot and your job is to answer questions about it. Assume that all questions are related to the Innovation CoPilot. Keep your answers technical and based on facts – do not hallucinate features."))
        index = VectorStoreIndex.from_documents(docs, service_context=service_context)
        return index

#################################################################################
# Additional, specific functions I had in the Innovation CoPilot for inspiration:

# Function to extract keywords from a text
def extract_keywords(text):
    tokens = preprocess(text)
    freq_dist = nltk.FreqDist(tokens)
    keywords = [word for word in list(freq_dist.keys()) if word not in excluded_keywords][:10]
    return keywords

# Function to remove punctuation, tokenize, remove stopwords, and stem
def preprocess(text):
    stop_words = get_stop_words('english')
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in string.punctuation]
    tokens = [word.lower() for word in tokens if word.isalpha()]
    tokens = [word for word in tokens if not word in stop_words]
    return tokens

# Function to identify relevant articles/information based on context of conversation
def get_relevant_articles(keywords, articles, num_articles=2):
    vectorizer = TfidfVectorizer()
    article_data = [articles[key]["summary"] + ' ' + ' '.join(articles[key]['keywords'].split(', ')) for key in articles.keys()]
    article_ids = [key for key in articles.keys()]

    # Create vectors for articles and input keywords
    article_vectors = vectorizer.fit_transform(article_data)
    keyword_vector = vectorizer.transform([' '.join(keywords)])

    # Compute cosine similarity
    cosine_similarities = cosine_similarity(keyword_vector, article_vectors).flatten()

    # Get the most similar articles
    most_similar_articles_indices = cosine_similarities.argsort()[:-num_articles - 1:-1]
    most_similar_articles = [article_ids[i] for i in most_similar_articles_indices]

    return {article: articles[article] for article in most_similar_articles}

# Function for constructing an index out of a knowledge base and appending indexed information to our prompt (another implementation that can be run locally rather than through github)
def construct_index(directory_path):
    if os.path.exists('index.json'):
        index = GPTSimpleVectorIndex.load_from_disk('index.json')
        return index

    max_input_size = 4096
    num_outputs = 512
    max_chunk_overlap = 20
    chunk_size_limit = 600

    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)
    llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=.3, model_name="gpt-3.5-turbo-16k-0613", max_tokens=num_outputs))

    documents = SimpleDirectoryReader(directory_path).load_data()
    index = GPTSimpleVectorIndex(documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper)

    index.save_to_disk('index.json')
    return index
