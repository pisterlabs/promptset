import dotenv
import pprint
import json
from langchain.document_loaders import AsyncChromiumLoader
from langchain.document_transformers import BeautifulSoupTransformer
from langchain.chat_models import ChatOpenAI
from langchain.chains import create_extraction_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter

dotenv.load_dotenv()

llm = ChatOpenAI(temperature=0, model="gpt-4-1106-preview")

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

def scrape_with_playwright(urls, schema):
    loader = AsyncChromiumLoader(urls)
    docs = loader.load()
    bs_transformer = BeautifulSoupTransformer()
    docs_transformed = bs_transformer.transform_documents(
        docs, tags_to_extract=["body"]
    )
    print("Extracting content with AI")

    extracted_content = extract(schema=schema, content=docs_transformed[0].page_content)
    pprint.pprint(extracted_content)

    print(extracted_content)

    return extracted_content

url = input("Please enter the link: ")
urls = [url]
extracted_content = scrape_with_playwright(urls, schema=schema)

with open('extracted_content.json', 'w') as file:
    json.dump(extracted_content, file)
    print("JSON Created")
