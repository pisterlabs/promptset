from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from selenium import webdriver
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import pinecone
import re
import os

load_dotenv()

city_dict = {
    'chicago': 'il--chicago',
    # 'philadelphia': 'pa--philadelphia',
}

def clean_string(input_string):
    cleaned_string = re.sub(r'[ /]', '_', input_string).lower()
    return cleaned_string

def make_event_txt(driver, event):
    event_name = event.find('h2', class_='Typography_root__4bejd #3a3247 Typography_body-lg__4bejd event-card__clamp-line--two Typography_align-match-parent__4bejd').text
    cleaned_event_name = clean_string(event_name)
    event_link = event.find('a', class_='event-card-link')['href']
    event_id = event.find('a', class_='event-card-link')['data-event-id']
    driver.get(event_link)
    driver.implicitly_wait(10)
    event_soup = BeautifulSoup(driver.page_source, 'html.parser')
    event_description = str(event_soup.find('div', class_='eds-text--left').text)
    with open(f'./events/{str(event_id)}.txt', 'w') as f:
        # f.write(f'Event Name: {event_name}\n')
        # f.write(f'Event Link: {event_link}\n')
        # f.write(f'Event ID: {event_id}\n')
        f.write(f'{event_description}')

def scrape_events():
    driver = webdriver.Chrome()
    for _, city_code in city_dict.items():
        for i in range(1, 2):
            try:
                driver.get(f"https://www.eventbrite.com/d/{city_code}/all-events/?page={i}")
                driver.implicitly_wait(10)
                soup = BeautifulSoup(driver.page_source, 'html.parser')
                events_list = soup.find_all('li', class_='search-main-content__events-list-item')
                for event in events_list:
                    make_event_txt(driver, event)
            except Exception as e:
                print(e)
                break
    driver.quit()

def start_pinecone():
    OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_ENV = os.getenv("PINECONE_ENV")
    pinecone.init(
        api_key=PINECONE_API_KEY,
        environment=PINECONE_ENV
    )

def clear_vectorstore():
    start_pinecone()
    index = pinecone.Index("events-index")
    index.delete(delete_all=True)

def load_vectors():
    start_pinecone()
    for file in os.listdir('events'):
        with open(f'events/{file}') as f:
            try:
                event = f.read()
                text_splitter = CharacterTextSplitter(   
                    separator = "\n\n",
                    length_function = len,
                )
                texts = text_splitter.create_documents([event], metadatas=[{'event-id': file.split('.')[0]}])
                index = pinecone.Index("events-index")
                embeddings = OpenAIEmbeddings()
                vectorstore = Pinecone(index, embeddings.embed_query, "text")
                vectorstore.add_documents(texts)
            except Exception as e:
                print(e)

def get_qa():
    start_pinecone()
    docsearch = Pinecone.from_existing_index("events-index", embedding=OpenAIEmbeddings())
    matched_docs = docsearch.similarity_search("i love taylor swift", k=3)
    for i in matched_docs:
        print(i.metadata)
        print(i.page_content)
        print('\n')
        print('\n')