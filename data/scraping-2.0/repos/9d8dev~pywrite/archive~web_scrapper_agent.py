import dotenv
dotenv.load_dotenv()

from langchain.document_loaders import AsyncChromiumLoader
from langchain.document_transformers import BeautifulSoupTransformer
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")

from langchain.chains import create_extraction_chain

schema = {
    "properties": {
        "hotel_name": {"type": "string"},
        "slug_of_hotel_page": {"type": "string"},
        "description_of_the_hotel": {"type": "string"},
        "location_of_hotel": {"type": "string"},
        "main_hotel_image_url": {"type": "string"},
    },
    "required": ["hotel_name", "slug_of_hotel_page", "description_of_the_hotel", "location_of_hotel", "main_hotel_image_url"],
}


def extract(content: str, schema: dict):
    return create_extraction_chain(schema=schema, llm=llm).run(content)


import pprint

from langchain.text_splitter import RecursiveCharacterTextSplitter


def scrape_with_playwright(urls, schema):
    loader = AsyncChromiumLoader(urls)
    docs = loader.load()
    bs_transformer = BeautifulSoupTransformer()
    docs_transformed = bs_transformer.transform_documents(
        docs, tags_to_extract=["div"]
    )
    print("Extracting content with LLM")

    # Grab the first 1000 tokens of the site
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000, chunk_overlap=0
    )
    splits = splitter.split_documents(docs_transformed)

    # Process the first split
    extracted_content = extract(schema=schema, content=splits[0].page_content)
    pprint.pprint(extracted_content)
    return extracted_content


urls = ["https://www.comohotels.com/destinations"]
extracted_content = scrape_with_playwright(urls, schema=schema)

import json

with open('extracted_content.json', 'w') as file:
    json.dump(extracted_content, file)
    print("JSON Created")
