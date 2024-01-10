# %% [markdown]
# [![Open In Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/langchain-ai/langchain/blob/master/docs/docs/use_cases/web_scraping.ipynb)
# 
# ## Use case
# 
# [Web research](https://blog.langchain.dev/automating-web-research/) is one of the killer LLM applications:
# 
# * Users have [highlighted it](https://twitter.com/GregKamradt/status/1679913813297225729?s=20) as one of his top desired AI tools. 
# * OSS repos like [gpt-researcher](https://github.com/assafelovic/gpt-researcher) are growing in popularity. 
#  
# ![Image description](/img/web_scraping.png)
#  
# ## Overview
# 
# Gathering content from the web has a few components:
# 
# * `Search`: Query to url (e.g., using `GoogleSearchAPIWrapper`).
# * `Loading`: Url to HTML  (e.g., using `AsyncHtmlLoader`, `AsyncChromiumLoader`, etc).
# * `Transforming`: HTML to formatted text (e.g., using `HTML2Text` or `Beautiful Soup`).
# 
# ## Quickstart

# %%

# !pip install -q openai langchain playwright beautifulsoup4 playwright install python-dotenv dotenv-python

print("Checking dependencies...")
# !pip install --upgrade pip --quiet
# !pip install langchain --quiet
# !pip install python-dotenv --quiet
# !pip install openai --quiet
# !pip install playwright --quiet
# !pip install beautifulsoup4 --quiet 
# !pip install chromadb --quiet
# !pip install google-api-python-client --quiet
# !pip install unstructured[rst]
# !pip install libmagic --quiet
print("Done!")


from dotenv import load_dotenv
load_dotenv()


# %% [markdown]
# Scraping HTML content using a headless instance of Chromium.
# 
# * The async nature of the scraping process is handled using Python's asyncio library.
# * The actual interaction with the web pages is handled by Playwright.

# %%
# from langchain.document_loaders import AsyncChromiumLoader

# Load HTML

# loader = AsyncChromiumLoader(["https://www.wsj.com"])

from langchain.document_transformers import BeautifulSoupTransformer
# from langchain.document_loaders import UnstructuredURLLoader
from langchain.document_loaders import UnstructuredRTFLoader, DirectoryLoader
from pathlib import  Path

diataxis_folder = Path("//Users/jon/github_repos/jonmatthis/pauljon/diataxis-documentation-framework-repo")
diataxis_documents = []
loader = DirectoryLoader(diataxis_folder, glob="**/*.rst")
docs = loader.load()
f=9

# html = loader.load()

# %% [markdown]
# Scrape text content tags such as `<p>, <li>, <div>, and <a>` tags from the HTML content:
# 
# * `<p>`: The paragraph tag. It defines a paragraph in HTML and is used to group together related sentences and/or phrases.
#  
# * `<li>`: The list item tag. It is used within ordered (`<ol>`) and unordered (`<ul>`) lists to define individual items within the list.
#  
# * `<div>`: The division tag. It is a block-level element used to group other inline or block-level elements.
#  
# * `<a>`: The anchor tag. It is used to define hyperlinks.
# 
# * `<span>`:  an inline container used to mark up a part of a text, or a part of a document. 
# 
# For many news websites (e.g., WSJ, CNN), headlines and summaries are all in `<span>` tags.

# %%
# Transform
bs_transformer = BeautifulSoupTransformer()
docs_transformed = bs_transformer.transform_documents(docs,tags_to_extract=["span"])

# %%
# Result
docs_transformed[0].page_content[0:500]

# %% [markdown]
# These `Documents` now are staged for downstream usage in various LLM apps, as discussed below.
# 
# ## Loader
# 
# ### AsyncHtmlLoader
# 
# The [AsyncHtmlLoader](docs/integrations/document_loaders/async_html) uses the `aiohttp` library to make asynchronous HTTP requests, suitable for simpler and lightweight scraping.
# 
# ### AsyncChromiumLoader
# 
# The [AsyncChromiumLoader](docs/integrations/document_loaders/async_chromium) uses Playwright to launch a Chromium instance, which can handle JavaScript rendering and more complex web interactions.
# 
# Chromium is one of the browsers supported by Playwright, a library used to control browser automation. 
# 
# Headless mode means that the browser is running without a graphical user interface, which is commonly used for web scraping.

# %%
from langchain.document_loaders import AsyncHtmlLoader
urls = ["https://www.espn.com","https://lilianweng.github.io/posts/2023-06-23-agent/"]
loader = AsyncHtmlLoader(urls)
docs = loader.load()

# %% [markdown]
# ## Transformer
# 
# ### HTML2Text
# 
# [HTML2Text](docs/integrations/document_transformers/html2text) provides a straightforward conversion of HTML content into plain text (with markdown-like formatting) without any specific tag manipulation. 
# 
# It's best suited for scenarios where the goal is to extract human-readable text without needing to manipulate specific HTML elements.
# 
# ### Beautiful Soup
#  
# Beautiful Soup offers more fine-grained control over HTML content, enabling specific tag extraction, removal, and content cleaning. 
# 
# It's suited for cases where you want to extract specific information and clean up the HTML content according to your needs.

# %%
from langchain.document_loaders import AsyncHtmlLoader
urls = ["https://www.espn.com", "https://lilianweng.github.io/posts/2023-06-23-agent/"]
loader = AsyncHtmlLoader(urls)
docs = loader.load()

# %%
from langchain.document_transformers import Html2TextTransformer
html2text = Html2TextTransformer()
docs_transformed = html2text.transform_documents(docs)
docs_transformed[0].page_content[0:500]

# %% [markdown]
# ## Scraping with extraction
# 
# ### LLM with function calling
# 
# Web scraping is challenging for many reasons. 
# 
# One of them is the changing nature of modern websites' layouts and content, which requires modifying scraping scripts to accommodate the changes.
# 
# Using Function (e.g., OpenAI) with an extraction chain, we avoid having to change your code constantly when websites change. 
# 
# We're using `gpt-3.5-turbo-0613` to guarantee access to OpenAI Functions feature (although this might be available to everyone by time of writing). 
# 
# We're also keeping `temperature` at `0` to keep randomness of the LLM down.

# %%
from langchain.chat_models import ChatOpenAI
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")

# %% [markdown]
# ### Define a schema
# 
# Next, you define a schema to specify what kind of data you want to extract. 
# 
# Here, the key names matter as they tell the LLM what kind of information they want. 
# 
# So, be as detailed as possible. 
# 
# In this example, we want to scrape only news article's name and summary from The Wall Street Journal website.

# %%
from langchain.chains import create_extraction_chain

schema = {
    "properties": {
        "news_article_title": {"type": "string"},
        "news_article_summary": {"type": "string"},
    },
    "required": ["news_article_title", "news_article_summary"],
}

def extract(content: str, schema: dict):
    return create_extraction_chain(schema=schema, llm=llm).run(content)

# %% [markdown]
# ### Run the web scraper w/ BeautifulSoup
# 
# As shown above, we'll be using `BeautifulSoupTransformer`.

# %%
import pprint
from langchain.text_splitter import RecursiveCharacterTextSplitter

def scrape_with_playwright(urls, schema):
    
    loader = AsyncChromiumLoader(urls)
    docs = loader.load()
    bs_transformer = BeautifulSoupTransformer()
    docs_transformed = bs_transformer.transform_documents(docs,tags_to_extract=["span"])
    print("Extracting content with LLM")
    
    # Grab the first 1000 tokens of the site
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=1000, 
                                                                    chunk_overlap=0)
    splits = splitter.split_documents(docs_transformed)
    
    # Process the first split 
    extracted_content = extract(
        schema=schema, content=splits[0].page_content
    )
    pprint.pprint(extracted_content)
    return extracted_content

urls = ["https://www.wsj.com"]
extracted_content = scrape_with_playwright(urls, schema=schema)

# %% [markdown]
# We can compare the headlines scraped to the page:
# 
# ![Image description](/img/wsj_page.png)
# 
# Looking at the [LangSmith trace](https://smith.langchain.com/public/c3070198-5b13-419b-87bf-3821cdf34fa6/r), we can see what is going on under the hood:
# 
# * It's following what is explained in the [extraction](docs/use_cases/extraction).
# * We call the `information_extraction` function on the input text.
# * It will attempt to populate the provided schema from the url content.

# %% [markdown]
# ## Research automation
# 
# Related to scraping, we may want to answer specific questions using searched content.
# 
# We can automate the process of [web research](https://blog.langchain.dev/automating-web-research/) using a retriever, such as the `WebResearchRetriever` ([docs](https://python.langchain.com/docs/modules/data_connection/retrievers/web_research)).
# 
# ![Image description](/img/web_research.png)
# 
# Copy requirements [from here](https://github.com/langchain-ai/web-explorer/blob/main/requirements.txt):
# 
# `pip install -r requirements.txt`
#  
# Set `GOOGLE_CSE_ID` and `GOOGLE_API_KEY`.

# %%
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models.openai import ChatOpenAI
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.retrievers.web_research import WebResearchRetriever

# %%
# Vectorstore
vectorstore = Chroma(embedding_function=OpenAIEmbeddings(),persist_directory="./chroma_db_oai")

# LLM
llm = ChatOpenAI(temperature=0)

# Search 
search = GoogleSearchAPIWrapper()

# %% [markdown]
# Initialize retriever with the above tools to:
#     
# * Use an LLM to generate multiple relevant search queries (one LLM call)
# * Execute a search for each query
# * Choose the top K links per query  (multiple search calls in parallel)
# * Load the information from all chosen links (scrape pages in parallel)
# * Index those documents into a vectorstore
# * Find the most relevant documents for each original generated search query

# %%
# Initialize
web_research_retriever = WebResearchRetriever.from_llm(
    vectorstore=vectorstore,
    llm=llm, 
    search=search)

# %%
# Run
import logging
logging.basicConfig()
logging.getLogger("langchain.retrievers.web_research").setLevel(logging.INFO)
from langchain.chains import RetrievalQAWithSourcesChain
user_input = "How do LLM Powered Autonomous Agents work?"
qa_chain = RetrievalQAWithSourcesChain.from_chain_type(llm,retriever=web_research_retriever)
result = qa_chain({"question": user_input})
result

# %% [markdown]
# ### Going deeper 
# 
# * Here's a [app](https://github.com/langchain-ai/web-explorer/tree/main) that wraps this retriever with a lighweight UI.

# %% [markdown]
# ## Question answering over a website
# 
# To answer questions over a specific website, you can use Apify's [Website Content Crawler](https://apify.com/apify/website-content-crawler) Actor, which can deeply crawl websites such as documentation, knowledge bases, help centers, or blogs,
# and extract text content from the web pages.
# 
# In the example below, we will deeply crawl the Python documentation of LangChain's Chat LLM models and answer a question over it.
# 
# First, install the requirements
# `pip install apify-client openai langchain chromadb tiktoken`
#  
# Next, set `OPENAI_API_KEY` and `APIFY_API_TOKEN` in your environment variables.
# 
# The full code follows:

# %%
from langchain.docstore.document import Document
from langchain.indexes import VectorstoreIndexCreator
from langchain.utilities import ApifyWrapper

apify = ApifyWrapper()
# Call the Actor to obtain text from the crawled webpages
loader = apify.call_actor(
    actor_id="apify/website-content-crawler",
    run_input={"startUrls": [{"url": "https://python.langchain.com/docs/integrations/chat/"}]},
    dataset_mapping_function=lambda item: Document(
        page_content=item["text"] or "", metadata={"source": item["url"]}
    ),
)

# Create a vector store based on the crawled data
index = VectorstoreIndexCreator().from_loaders([loader])

# Query the vector store
query = "Are any OpenAI chat models integrated in LangChain?"
result = index.query(query)
print(result)


