import os
import requests
import serpapi
from bs4 import BeautifulSoup
import streamlit as st
from dotenv import load_dotenv
load_dotenv()
# from langchain.text_splitter import CharacterTextSplitter


class SerpApiClient:
    """
    Client for SerpApi.
    """

    def __init__(self, api_key: str):
        """
        Args:
        api_key (str): SerpApi API key.
        """
        self.api_key = api_key


    def webpages_from_serpapi(self, query: str, num_results: int = 3) -> list[tuple([str, str, str])]:
        """Get webpages from SerpApi.

        Args:
            query (str): Query string.
            num_results (int): Number of results to return.

        Returns:
            list[tuple([str, str, str])]: List of webpages. Each webpage is a tuple of (title, link, webpage). webpage is a string of all the paragraphs (<p></p>) in the webpage with 100+ characters.
        """

        params = {
            "q": query,
            "location": "Mumbai, Maharashtra, India",
            "api_key": self.api_key,
        }
        search = serpapi.GoogleSearch(params)
        results = search.get_dict()

        if "error" in results:
            return f"Error: {results['error']}"
        else:
            print(f"Number of organic results: {len(results['organic_results'])}")

        results_condensed = [(result['title'], result['link'])
                            for result in results['organic_results']]

        content_p = ""
        webpages = []

        for title, link in results_condensed[:num_results]:
            print(f"Title: {title}")

            try:
                print(f"Requesting {link}")
                response = requests.get(link)
            except requests.exceptions.ConnectionError:
                print("Connection timed out... Moving to next link")
                continue
            except Exception as e:
                print(f"Error: {e}")
                continue

            if response.status_code != 200:
                print(f"{link} is not accessible. Response code: {response.status_code}")
                continue

            soup = BeautifulSoup(response.text, 'html.parser')
            content_p += f'## {title}' + "\n"
            webpage = f'## {title}' + "\n"

            for p in soup.find_all('p'):
                paragraph = p.get_text(separator=' ')
                if len(paragraph) > 100:
                    webpage += paragraph + "\n"
                    content_p += paragraph
                    content_p += "\n"

            content_p += "\n---\n"
            webpages.append((title, link, webpage))

        return webpages
