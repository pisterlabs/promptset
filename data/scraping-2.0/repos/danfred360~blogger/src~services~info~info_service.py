import asyncio
from asyncio import gather
from typing import Dict, List
from urllib.parse import quote_plus

import requests
import selenium
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException
from langchain.chains import create_extraction_chain_pydantic
from langchain.chat_models import ChatOpenAI
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from webdriver_manager.chrome import ChromeDriverManager

from ...config import Config
from .models import Metadata, Result


class InfoService:
    def __init__(self, config: Config):
        self.llm = ChatOpenAI(
            temperature=0, model="gpt-3.5-turbo", openai_api_key=config.openai_api_key
        )

    async def fetch_content(self, link: str) -> Dict:
        chrome_service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=chrome_service)
        driver.get(link)
        await asyncio.sleep(5)
        page_source = driver.page_source
        driver.quit()

        page_soup = BeautifulSoup(page_source, "html.parser")
        content = {
            "summary": page_soup.find("meta", {"name": "description"})["content"]
            if page_soup.find("meta", {"name": "description"})
            else "",
            "links": [
                a["href"]
                for a in page_soup.find_all("a", href=True)
                if a["href"].startswith("http")
            ],
            "favicon": f"{link}favicon.ico",
            "text": page_soup.get_text(),
        }
        return content

    async def process_result(self, result) -> Result:
        title = snippet = ""
        link = None
        try:
            span_element = result.find_element(
                By.CSS_SELECTOR, "span[jscontroller='msmzHf']"
            )
            link_element = (
                span_element.find_element(By.TAG_NAME, "a") if span_element else None
            )
            link = link_element.get_attribute("href") if link_element else ""
        except selenium.common.exceptions.NoSuchElementException:
            pass  # handle the exception or log it as needed

        # If link is found, proceed to fetch content and process result
        if link:
            content = await self.fetch_content(link)
            metadata = create_extraction_chain_pydantic(
                pydantic_schema=Metadata, llm=self.llm
            ).run(content["text"])
            title = (
                span_element.text if span_element else ""
            )  # Assuming title is the text within span_element
            snippet = ""
            return Result(
                title=title, url=link, snippet=snippet, extracted_metadata=metadata
            )
        else:
            return None

    async def search_google(self, query: str, n: int):
        chrome_service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=chrome_service)
        driver.maximize_window()
        driver.get("https://google.com")
        search_box = driver.find_element(By.NAME, "q")
        search_box.send_keys(query + Keys.RETURN)
        await asyncio.sleep(5)  # Replace implicit wait with asyncio.sleep
        results = driver.find_elements(By.CSS_SELECTOR, "div.g")

        if n > len(results):
            n = len(results)  # Ensure n is within bounds

        tasks = [self.process_result(results[i]) for i in range(n)]
        top_results = await gather(*tasks)
        driver.quit()  # Don't forget to close the browser

        return top_results
