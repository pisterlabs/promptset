# In this example, we'll use the Website Content Crawler Actor,
# which can deeply crawl websites such as documentation, knowledge
# bases, help centers, or blogs and extract text content from the web
# pages. Then we feed the documents into a vector index and answer
# questions from it.

from langchain.document_loaders.base import Document
from langchain.indexes import VectorstoreIndexCreator
from langchain.utilities import ApifyWrapper
import os
import openai
from dotenv import load_dotenv
from langchain import OpenAI

# _ = load_dotenv(find_dotenv())
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


apify = ApifyWrapper()

loader = apify.call_actor(
    actor_id="apify/website-content-crawler",
    run_input={"startUrls": [{"url": "https://python.langchain.com/en/latest/"}]},
    dataset_mapping_function=lambda item: Document(
        page_content=item["text"] or "", metadata={"source": item["url"]}
    ),
)


index = VectorstoreIndexCreator().from_loaders([loader])


query = "What is Google Ads API?"
result = index.query_with_sources(query)

print(result["answer"])
print(result["sources"])
