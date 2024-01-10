import os
import requests
from bs4 import BeautifulSoup
import pandas as pd
import streamlit as st 

from langchain.document_loaders import (
    YoutubeLoader,
    DataFrameLoader
)
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from googleapiclient.discovery import build

YOUTUBE_API_KEY = os.environ.get('YOUTUBE_API_KEY')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

def splitter_strategy():
    # Define text chunk strategy
    splitter = CharacterTextSplitter(
    chunk_size=2000, 
    chunk_overlap=50,
    separator=" "
    )
    return splitter

def load_youtube_data(video_id):
  yt_loader = YoutubeLoader(video_id)
  yt_data = yt_loader.load()
  return yt_data

def get_youtube_data():
    # # Set up the API client
    youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)

    # Specify the playlist ID
    playlist_id = "PLezjrIbG3mtrIfVQHiTugr8YNS1BGVSsh"

    # Retrieve all videos from the playlist
    video_ids = []
    next_page_token = None

    while True:
        # Request playlist items
        playlist_response = youtube.playlistItems().list(
            part="snippet,contentDetails",
            playlistId=playlist_id,
            maxResults=50,  # Adjust the value as per your requirement
            pageToken=next_page_token
        ).execute()

        # Extract video IDs
        for item in playlist_response["items"]:
            video_ids.append(item["contentDetails"]["videoId"])

        # Check if there are more pages
        next_page_token = playlist_response.get("nextPageToken")
        if not next_page_token:
            break

    yt_data_list = []
    splitter = splitter_strategy()
    for video in video_ids:
        yt_data = load_youtube_data(video)
        yt_data_list += yt_data
        yt_data_split = splitter.split_documents(yt_data)
    
    return yt_data_split

def get_blog_data():
    base_url = "https://churnzero.com/blog"
    page_number = 1
    all_links = []

    repeat = False
    while True:
        if page_number == 49:
            break
        # Construct the URL for the current page
        url = f"{base_url}/page/{page_number}/"

        # Send a GET request to fetch the HTML content
        response = requests.get(url)

        # Check if the page exists
        if response.status_code == 200:
            # Parse the HTML content
            soup = BeautifulSoup(response.content, "html.parser")

            # Find all the link elements and extract their URLs
            links = soup.find_all("a")
            for link in links:
                href = link.get("href")
                if href and href.startswith("https://churnzero.com/blog"):
                    all_links.append(href)
            # Move to the next page
            page_number += 1
        else:
            # Break the loop if the page doesn't exist
            break

    filtered_links = [link for link in all_links if "/page/" not in link and not link.endswith("/blog")]
    filtered_links = list(set(filtered_links))

    # Empty lists to store titles and text
    titles = []
    texts = []

    # Iterate over the blog links
    for link in filtered_links:
        response = requests.get(link)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract the title from the <title> tag
        title = soup.title.string.strip() if soup.title else None
        
        # Extract the text from <p> tags within the specified element
        text_element = soup.find('div', class_="et_pb_module et_pb_post_content et_pb_post_content_0_tb_body")
        paragraphs = text_element.find_all('p') if text_element else []
        text = ' '.join([p.get_text().strip() for p in paragraphs])
        
        # Append the title and text to the respective lists
        titles.append(title)
        texts.append(text)

    # Create a DataFrame from the titles and texts
    df = pd.DataFrame({'Title': titles, 'Text': texts})

    # Display the DataFrame
    # print(df)

    blog_data = DataFrameLoader(
        df[["Title", "Text"]], 
        page_content_column="Text")
    blog_data = blog_data.load()
    splitter = splitter_strategy()
    blog_data_split = splitter.split_documents(blog_data)

    return blog_data_split

def get_model(blog_data_split, yt_data_split):
    # Define embedding model
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    research_data = blog_data_split + yt_data_split
    research_store = Chroma.from_documents(
        research_data, embeddings, collection_name="research"
    )

    #Define the model 
    llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0,
    openai_api_key=OPENAI_API_KEY,
    max_tokens=512,
    )

    research_template = """As a ChurnZero marketing bot, your goal is to provide accurate and helpful information about ChurnZero,
    a powerful customer success software that helps subscription businesses fight customer churn.
    You should answer user inquiries based on the context provided and avoid making up answers.
    If you don't know the answer, simply state that you don't know.
    Remember to provide relevant information about ChurnZero's features, benefits,
    and use cases to assist the user in understanding its value for customer success.

    {context}

    Question: {question}"""

    RESEARCH_PROMPT = PromptTemplate(
        template=research_template, input_variables=["context", "question"]
    )
    model = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=research_store.as_retriever(),
        chain_type_kwargs={"prompt": RESEARCH_PROMPT},
    )
    return model

#Pass data to the model 
youtube_data = get_youtube_data()
blog_data = get_blog_data()
model = get_model(youtube_data,blog_data )

#Display input and output using streamlit 
st.title("Competitor Research")

user_question = st.text_input(
    "Enter question : ",
    placeholder = "What are the benefits of using ChurnZero?",
)

if st.button("Tell me about it", type="primary"):
    try:
        answer = model.run(user_question)
        print(answer)
        st.success(answer)
    except:
        st.error("Error")
