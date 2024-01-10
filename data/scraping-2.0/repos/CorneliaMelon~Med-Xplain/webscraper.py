from urllib.parse import urljoin

import dotenv
from langchain.chains import create_extraction_chain

dotenv.load_dotenv()
import os
import openai

import urllib.parse

openai.api_key = os.getenv("OPENAI_API_KEY")


def extract(content: str, schema: dict):
    from langchain.chat_models import ChatOpenAI
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")
    return create_extraction_chain(schema=schema, llm=llm).run(content)


def call_nhs_search(query):
    from langchain.document_loaders import AsyncChromiumLoader
    # from langchain.document_transformers import BeautifulSoupTransformer

    from bs4 import BeautifulSoup
    # Load HTML
    loader = AsyncChromiumLoader(
        [f"https://www.nhs.uk/search/results?q={urllib.parse.quote(query)})"])
    html = loader.load()

    soup = BeautifulSoup(html[0].page_content, "html.parser")
    link_tags = soup.find_all('a')

    # Create a list of dictionaries with title and href as key-value pairs
    links_with_titles = [
        {"page_title": tag.get_text().strip("\n "), "page_url": "https://www.nhs.uk/" + tag.get('href'), } for
        tag in link_tags[11:-10]]

    return links_with_titles

    # bs_transformer.
    # docs_transformed = bs_transformer.transform_documents(html,
    #                                                       unwanted_tags=["script", "style", "head", "title", "meta",
    #                                                                      "[document]", "header"], remove_lines=True,
    #                                                       tags_to_extract=["a"], attributes_to_extract=["href"])

    # schema = {
    #     "properties": {
    #         "page_title": {"type": "string"},
    #         "page_url": {"type": "string"},
    #     },
    #     "required": ["page_title", "page_url"],
    # }
    # extracted_content = extract(docs_transformed[0].page_content, schema=schema)

    # if len(extracted_content) > 0 and extracted_content[0]["page_title"] == "Cookies":
    #     extracted_content.pop(0)
    # # TODO: If extracted content is empty, return a message saying that no results were found
    # return extracted_content


if __name__ == "__main__":
    print(call_nhs_search("iron deficiency"))
