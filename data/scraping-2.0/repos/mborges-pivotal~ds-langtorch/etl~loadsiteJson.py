import os
from langchain.vectorstores import AstraDB
from langchain.schema import Document
from langchain.embeddings import OpenAIEmbeddings

from datasets import load_dataset
from dotenv import load_dotenv

from langchain.document_loaders import AsyncHtmlLoader
from langchain.document_transformers import BeautifulSoupTransformer

load_dotenv()

ASTRA_DB_APPLICATION_TOKEN = os.environ.get("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_API_ENDPOINT = os.environ.get("ASTRA_DB_API_ENDPOINT")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

embedding = OpenAIEmbeddings()
vstore = AstraDB(
    embedding=embedding,
    collection_name="FourSeasonsSite",
    token=os.environ["ASTRA_DB_APPLICATION_TOKEN"],
    api_endpoint=os.environ["ASTRA_DB_API_ENDPOINT"]
)

# List of URLs
urls = [
        "https://www.fourseasons.com/austin/",
        "https://www.fourseasons.com/austin/accommodations/",
        "https://www.fourseasons.com/austin/landing-pages/property/ice-rodeo/",
        "https://www.fourseasons.com/magazine/best-of/austin-photo-tour/",
        "https://www.fourseasons.com/austin/landing-pages/property/staycation/",
        "https://www.fourseasons.com/austin/services-and-amenities/pet-friendly-stays/",
        "https://www.fourseasons.com/austin/services-and-amenities/fitness/",
        "https://www.fourseasons.com/austin/offers/winter-escape/",
        "https://www.fourseasons.com/austin/offers/stay-longer-fourth-night-free/",
        "https://www.fourseasons.com/austin/offers/room_rate/",
        "https://www.fourseasons.com/austin/offers/austin-getaway-20-off/",
        "https://www.fourseasons.com/austin/offers/bed_and_breakfast/"
]

# Load HTML
# loader = AsyncChromiumLoader(["https://www.fourseasons.com/austin/"])
loader = AsyncHtmlLoader(urls)
html = loader.load()

# // div class="normal
bs_transformer = BeautifulSoupTransformer()
docs_transformed = bs_transformer.transform_documents(html,tags_to_extract=["div"])

# Further transformation
# [{'message': "Failed to insert document with _id '0a5bab1d7eca4dbc9dd9c8d40e24414c': 
# INVALID_ARGUMENT: Term of column query_text_values exceeds the byte limit for index. 
# Term size 11.239KiB. Max allowed size 5.000KiB."}]
docs = []
for entry in docs_transformed:
    # metadata = {"author": entry["author"]}
    # if entry["tags"]:
    #     # Add metadata tags to the metadata dictionary
    #     for tag in entry["tags"].split(";"):
    #         metadata[tag] = "y"

    # Add a LangChain document with the quote and metadata tags
    # doc = Document(page_content=entry["quote"], metadata=metadata)
    doc = Document(page_content=entry.page_content[0:5000])
    docs.append(doc)

inserted_ids = vstore.add_documents(docs)
print(f"\nInserted {len(inserted_ids)} documents.")