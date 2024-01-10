from utils import fetch_url, user_agents, llm
from lxml import html, etree
from langchain.chat_models import ChatOpenAI
from langchain.chains import create_extraction_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
import asyncio
import time
from langchain_core.documents import Document
from schemas import schema_city, schema_restaurant, schema_menu
from dotenv import load_dotenv, find_dotenv
import pprint
import random

load_dotenv(find_dotenv())


def extract(content: str, schema: dict, llm: ChatOpenAI):
    return create_extraction_chain(schema=schema, llm=llm).run(content)


async def scrape_cities(url):
    try:
        user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:90.0) Gecko/20100101 Firefox/90.0"
        headers = {'User-Agent': user_agent}
        html_content = await fetch_url(url, headers)
        tree = html.fromstring(html_content)
        a_elements = tree.xpath('//body//a')

        elements = ' '.join([etree.tostring(el, encoding='unicode') for el in a_elements])

        documents = [Document(page_content=elements, metadata={"source": url})]

        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=10000, chunk_overlap=0
        )
        splits = splitter.split_documents(documents)

        start_extraction = time.time()
        extracted_data = await asyncio.gather(*[asyncio.to_thread(extract, split.page_content, schema=schema_city, llm=llm) for split in splits])
        end_extraction = time.time()
        print(f"Time taken for LLM extraction: {end_extraction - start_extraction} seconds")
        extracted_cities = [city for sublist in extracted_data for city in sublist if '/oras' in city['city_link']]
        for city in extracted_cities:
            city['city_link'] = city['city_link'].replace('oras', 'restaurante')
        return extracted_cities
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return None


async def scrape_restaurants_base_function(url, user_agents):
    try:
        headers = {'User-Agent': user_agents}
        html_content = await fetch_url(url, headers)
        tree = html.fromstring(html_content)
        a_elements = tree.xpath('//body//a')

        elements = ' '.join([etree.tostring(el, encoding='unicode') for el in a_elements])

        documents = [Document(page_content=elements, metadata={"source": url})]

        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=10000, chunk_overlap=0
        )
        splits = splitter.split_documents(documents)

        start_extraction = time.time()
        extracted_data = await asyncio.gather(
            *[asyncio.to_thread(extract, split.page_content, schema=schema_restaurant, llm=llm) for split in
              splits])
        end_extraction = time.time()
        print(f"Time taken for LLM extraction: {end_extraction - start_extraction} seconds")
        extracted_restaurants = [restaurant for sublist in extracted_data for restaurant in sublist if
                                 '/restaurant' in restaurant['restaurant_link']]
        return extracted_restaurants
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return None


async def scrape_restaurants(urls):
    tasks = [
        scrape_restaurants_base_function(url, random.choice(user_agents))
        for url in urls
    ]

    results = await asyncio.gather(*tasks)

    for url, result in zip(urls, results):
        print(f"Results for {url}:")
        pprint.pprint(result)


async def scrape_restaurant_menus_base_function(restaurant_link, user_agent):
    try:
        headers = {'User-Agent': user_agent}
        html_content = await fetch_url(restaurant_link, headers)
        tree = html.fromstring(html_content)
        h5_elements = tree.xpath('//h5[@class="title-container"]/text()')
        span_elements = tree.xpath('//span[@class="price-container zprice"]/text()')

        elements = ' '.join(h5_elements + span_elements)

        documents = [Document(page_content=elements, metadata={"source": restaurant_link})]

        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=10000, chunk_overlap=0
        )
        splits = splitter.split_documents(documents)

        start_extraction = time.time()
        extracted_menus = await asyncio.gather(*[asyncio.to_thread(extract, split.page_content, schema=schema_menu, llm=llm) for split in splits])
        end_extraction = time.time()
        print(f"Time taken for LLM extraction: {end_extraction - start_extraction} seconds")

        return extracted_menus
    except Exception as e:
        print(f"Error scraping {restaurant_link}: {e}")
        return None


async def scrape_restaurant_menus(urls):
    tasks = [
        scrape_restaurant_menus_base_function(url, random.choice(user_agents))
        for url in urls
    ]

    results = await asyncio.gather(*tasks)

    for url, result in zip(urls, results):
        print(f"Menus for {url}:")
        pprint.pprint(result)







