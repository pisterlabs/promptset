import json, base64, openai, asyncio, os, pysqlite3, sys, shutil
from enum import Enum
from timeit import default_timer
from typing import Callable, Any
from json_parser import JsonOutputParser
from playwright.async_api import async_playwright, Browser
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.vectorstores import VectorStore
from langchain_core.pydantic_v1 import Field
from langchain.llms.base import BaseLLM
from langchain.utilities.google_serper import GoogleSerperAPIWrapper
from langchain.prompts import PromptTemplate
from langchain.document_loaders import WebBaseLoader, AsyncChromiumLoader
from langchain.document_transformers  import BeautifulSoupTransformer
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

def timer(func: Callable) -> Any:
  def inner(*args, **kwargs):
    start = default_timer()  
    result = func(*args, **kwargs)
    end = default_timer()
    print(f"{func.__name__}:", end - start)
    return result
  return inner

class WebScrapeMethod(str, Enum):
  simple = "simple"
  http = "http"
  browser = "browser"
  screenshot = "screenshot"

class WebResearchRetriever(BaseRetriever):  
  """Will use Google search to find the top 3 relevant web pages and return the most relevant document results."""
  
  llm: BaseLLM = Field(..., description="LLM used to ask NLP questions")
  vector_store: VectorStore = Field(..., description="Vector store used to find relevant documents")
  web_scrape_method: WebScrapeMethod = Field(WebScrapeMethod.http, description="Method used to scrape web pages")
  smart_top_links: bool = Field(True, description="Whether to use the LLM to find the top links")
  links_n: int = Field(3, description="Number of links to return")
  vector_n: int = Field(5, description="Number of vectors to return")
  
  @classmethod
  def from_llm(
    self, 
    llm: BaseLLM, 
    vector_store: VectorStore,
    web_scrape_method: WebScrapeMethod,
    smart_top_links: bool,
    links_n: int,
    vector_n: int
  ) -> "WebResearchRetriever":
    return self(
      llm=llm, 
      vector_store=vector_store, 
      web_scrape_method=web_scrape_method,
      smart_top_links=smart_top_links,
      links_n=links_n,
      vector_n=vector_n
    )
  
  def _parse_search_results(self, results: list[dict]) -> list[dict]:
      """Parse search results and return title, snippet, and link."""
      parsed_results = []
      for result in results:
          parsed_result = {
              "title": result["title"],
              "snippet": result["snippet"],
              "link": result["link"],
          }
          parsed_results.append(parsed_result)
      return parsed_results

  @timer
  def _web_search(self, query: str) -> list[dict]:
    """Search for query on Google and return results."""
    search_result = GoogleSerperAPIWrapper().results(query)
    if self.web_scrape_method == WebScrapeMethod.simple:
      if "answerBox" in search_result:
        answer = search_result["answerBox"]
      else:
        answer = search_result["organic"][0]
      return self._parse_search_results([answer])
    else:
      return self._parse_search_results(search_result["organic"])

  @timer
  def _fine_best_web_sources(self, llm: BaseLLM, query: str, search_result: list[dict]) -> list[str]:
    """Find the best web sources from the search results."""
    search_dump = json.dumps(search_result, indent=2)
    best_search_prompt = PromptTemplate.from_template(
      """You are an expert at looking at search results and picking the best ones. You will be given the <query> and the top 10 <results>. Each result will include <title> and <description> and <link>. Using the search result details, you will pick the top results based on how likely the result will have information related to the <query>. Remember, only return the top {links_n} results, and respond with a JSON list of <link> values.
      QUERY:
      {query}

      RESULTS:
      '''json
      {results}
      '''

      TOP {links_n} RESULTS:                                                  
      """)

    chain = best_search_prompt | llm | JsonOutputParser()
    best_links = chain.invoke({"query": query, "results": search_dump, "links_n": self.links_n}) 
    return best_links

  def _find_best_links(self, query: str, search_results: list[dict]):
    if self.links_n > len(search_results):
      best_links = [x["link"] for x in search_results]
    elif self.smart_top_links:
      best_links = self._fine_best_web_sources(self.llm, query, search_results)
    else:
      best_links = [x["link"] for x in search_results[:self.links_n]]
    return best_links

  @timer
  def _fast_web_scrape(self, best_links: list[str]) -> list[Document]:
    """Fast web scraping using just the URL."""
    web_loader = WebBaseLoader(best_links)
    search_docs = web_loader.aload()
    return search_docs

  async def _browserless_scrape_url(self, browser: Browser, url: str) -> Document:
    """Web scraping URL with Browswerless.io"""
    print(url)
    context = await browser.new_context()
    page = await context.new_page()
    await page.goto(url, wait_until="load")
    page_content = await page.content()
    await context.close()
    return Document(page_content=page_content, metadata={"source": url})

  async def _browserless_scrape_urls(self, best_links: list[str], token: str) -> list[Document]:
    """Web scraping URLs with Browswerless.io"""
    cdp_url = f"wss://chrome.browserless.io?token={token}&blockAds"
    async with async_playwright() as p:
      browser = await p.chromium.connect_over_cdp(cdp_url)
      calls = [self._browserless_scrape_url(browser, url) for url in best_links]
      search_docs = await asyncio.gather(*calls)
      await browser.close()
      return search_docs

  @timer
  def _browser_scrape(self, best_links: list[str]) -> list[Document]:
    """Web scraping using a headless browser."""
    token = os.getenv('BROWSERLESS_TOKEN')
    if token:
      print("using browserless.io")
      search_docs = asyncio.run(self._browserless_scrape_urls(best_links[:4], token)) # max of 4 urls
    else:
      web_loader = AsyncChromiumLoader(best_links)
      search_docs = web_loader.load()
      
    bs_transformer = BeautifulSoupTransformer()
    docs_transformed = bs_transformer.transform_documents(
      search_docs, 
      tags_to_extract=["div", "span"]
    )
    return docs_transformed

  async def _take_screenshot(self, browser: Browser, url: str, i: int) -> str:
    page = await browser.new_page()
    try:
      await page.goto(url, wait_until="load")
    except:
      return None
    filename = f"screenshots/image{i}.png"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    await page.screenshot(path=filename, full_page=True)
    return filename

  async def _take_screenshots(self, urls: str) -> list[str]:
    token = os.getenv('BROWSERLESS_TOKEN')
    if os.path.exists("screenshots"):
      shutil.rmtree("screenshots")
    async with async_playwright() as p:
      if token:
        print("using browserless.io")
        browser = await p.chromium.connect_over_cdp(f"wss://chrome.browserless.io?token={token}&blockAds")
        urls = urls[:4] # max of 4 urls
      else:
        browser = await p.chromium.launch()
      calls = [self._take_screenshot(browser, url, i) for i, url in enumerate(urls)]
      filenames = await asyncio.gather(*calls)
      filenames = [f for f in filenames if f is not None]
      await browser.close()
      return filenames
      
  def _extract_description_from_images(self, filenames: list[str]) -> str:
    image_messages = []
    for f in filenames:
      with open(f, "rb") as f:
        b64_image = base64.b64encode(f.read()).decode() 
      image_messages.append({
        "type": "image",
        "image_url": {
          "url": f"data:image/png;base64,{b64_image}"
        }
      })
    content = [{
      "type": "text",
      "text": f"Find information about the main product:"
    }]
    content.extend(image_messages)
    
    messages = [
      {
        "role": "system",
        "content": """You are a web scraper, whose job is to extract information about a product from website screenshots. The intention is to extract the product details that can be useful to describe the product. DO NOT include details about return policies, warranty information, or price."""
      },
      {
        "role": "user",
        "content": content
      }
    ]
    
    llm = openai.Client()
    response = llm.chat.completions.create(
      model="gpt-4-vision-preview",
      messages = messages,
      max_tokens=4000
    )
    message = response.choices[0].message
    return message.content
    
  @timer
  def _screenshot_scrape(self, best_links: list[str]) -> list[Document]:
    filenames = asyncio.run(self._take_screenshots(best_links))
    description = self._extract_description_from_images(filenames)
    return [Document(page_content=description, metadata={"source": ";".join(best_links)})]

  @timer
  def _find_relevant_search_docs(self, search_docs: list[Document], query: str) -> list[Document]:
    """Find the most relevant text within the search docs."""
    split_docs = RecursiveCharacterTextSplitter().split_documents(search_docs)  
    self.vector_store.add_documents(split_docs)
    related_docs = self.vector_store.similarity_search(query, k=self.vector_n)
    return related_docs
  
  def _web_scrape(self, best_links: list[str], search_results: list[Document]):
    if self.web_scrape_method == WebScrapeMethod.http:
      return self._fast_web_scrape(best_links)
    elif self.web_scrape_method == WebScrapeMethod.browser:
      return self._browser_scrape(best_links)
    elif self.web_scrape_method == WebScrapeMethod.screenshot:
      return self._screenshot_scrape(best_links)
    elif self.web_scrape_method == WebScrapeMethod.simple:
      return [Document(page_content=search_results[0]["snippet"], metadata={"source": best_links[0]})]
    else:
      raise ValueError(f"Unknown web scrape method: {self.web_scrape_method}")
  
  def get_relevant_documents(
    self, 
    query: str,
    url: str = None,     
    *, 
    run_manager: CallbackManagerForRetrieverRun
  ) -> list[Document]:
    """Search Google for documents related to the query input."""
    if url:
      search_results = [{"link": url}]
      self.web_scrape_method = WebScrapeMethod.http
    else:
      search_results = self._web_search(query)
    best_links = self._find_best_links(query, search_results)
    search_docs = self._web_scrape(best_links, search_results)
    search_docs = self._find_relevant_search_docs(search_docs, query)    
    return search_docs
