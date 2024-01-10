"""
arxiv_fetcher.py - Module for fetching papers from the arXiv API and providing methods for analysis.

This module contains the ArxivPaperFetcher class, which is responsible for fetching papers from the arXiv API
and providing methods for analyzing and processing the retrieved papers.
"""

import urllib.request
from urllib.error import URLError, HTTPError
import xml.etree.ElementTree as ET
import time
from openai import OpenAI
from scipy.spatial.distance import cosine


class ArxivPaperFetcher:
    """
    ArxivPaperFetcher - Fetches papers from the arXiv API and provides methods for analysis.

    Attributes:
        ARXIV_API_URL (str): The arXiv API URL for querying papers.

    Methods:
        __init__(self, api_key: str) -> None:
            Initializes the ArxivPaperFetcher with the OpenAI API key.

        fetch_papers(self) -> list:
            Fetches papers from the arXiv API and returns them as a list of strings.

        get_embedding(self, text: str) -> list:
            Gets the OpenAI embedding for the given text using text-embedding-ada-002.

        calculate_semantic_similarity(self, embedding1: list, embedding2: list) -> float:
            Calculates the cosine similarity between two embeddings.

        find_most_similar_papers(self, user_query: str, papers: list, k: int = 5) -> list:
            Finds the top k most semantically similar papers to the user's query.
    """

    ARXIV_API_URL = 'http://export.arxiv.org/api/query?search_query=ti:llama&start=0&max_results=70'

    def __init__(self, api_key: str) -> None:
        """
        Initializes the ArxivPaperFetcher with the OpenAI API key.

        Parameters:
            api_key (str): The OpenAI API key.
        """
        self.client = OpenAI(api_key=api_key)

    def fetch_papers(self) -> list:
        """
        Fetches papers from the arXiv API and returns them as a list of strings.

        Returns:
            list: A list of papers.
        """
        try:
            with urllib.request.urlopen(self.ARXIV_API_URL) as response:
                data = response.read().decode('utf-8')
                root = ET.fromstring(data)

            papers_list = []
            for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
                title = entry.find('{http://www.w3.org/2005/Atom}title').text
                summary = entry.find(
                    '{http://www.w3.org/2005/Atom}summary').text
                paper_info = f"Title: {title}\nSummary: {summary}\n"
                papers_list.append(paper_info)

            time.sleep(3)

            return papers_list
        except (URLError, HTTPError) as exception:
            print(f"Error fetching papers from arXiv: {exception}")
            return []

    def get_embedding(self, text: str) -> list:
        """
        Gets the OpenAI embedding for the given text using text-embedding-ada-002.

        Parameters:
            text (str): The input text.

        Returns:
            list: The OpenAI embedding for the text.
        """
        response = self.client.embeddings.create(
            input=text,
            model="text-embedding-ada-002"
        )
        return response.data[0].embedding

    def calculate_semantic_similarity(self, embedding1: list, embedding2: list) -> float:
        """
        Calculates the cosine similarity between two embeddings.

        Parameters:
            embedding1 (list): The first embedding.
            embedding2 (list): The second embedding.

        Returns:
            float: The cosine similarity between the embeddings.
        """
        similarity = 1 - cosine(embedding1, embedding2)
        return similarity

    def find_most_similar_papers(self, user_query: str, papers: list, k: int = 5) -> list:
        """
        Finds the top k most semantically similar papers to the user's query.

        Parameters:
            user_query (str): The user's query.
            papers (list): The list of papers.
            k (int, optional): The number of top papers to retrieve. Defaults to 5.

        Returns:
            list: A list of tuples containing paper information and similarity scores.
        """
        user_query_embedding = self.get_embedding(user_query.lower())
        similarity_scores = []

        for paper in papers:
            paper_embedding = self.get_embedding(paper.lower())
            similarity = self.calculate_semantic_similarity(
                user_query_embedding, paper_embedding)

            similarity_scores.append((paper, similarity))

        sorted_similarity_scores = sorted(
            similarity_scores, key=lambda x: x[1], reverse=True)

        top_k_papers = sorted_similarity_scores[:k]

        return top_k_papers
