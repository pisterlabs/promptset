import requests
import os
import re
import json
import pickle
import concurrent.futures
from typing import Callable, Iterable
from langchain.docstore.document import Document
from pathlib import Path
from bs4 import BeautifulSoup
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Iterable

from langchain.vectorstores import FAISS
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


class Assistant:

    def __init__(self):
        """
        Initializes the class instance.

        This function sets up the necessary attributes for the class. It initializes the following attributes:
        - `OPENAI_API_KEY`: The OpenAI API key obtained from the environment variables.
        - `HF_TOKEN`: The Hugging Face token obtained from the environment variables.
        - `client`: An instance of the OpenAI class initialized with the `OPENAI_API_KEY`.
        - `vector_store_path`: The path to the pickle file where the vector store data is stored.
        - `vector_store`: The loaded vector store data from the `vector_store_path` file.

        Parameters:
        None

        Returns:
        None
        """
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        self.HF_TOKEN = os.getenv("HF_TOKEN")
        self.client = OpenAI(api_key=self.OPENAI_API_KEY)
        self.vector_store_path = Path("url_texts.pkl")
        self.vector_store = self.load_data()

    def run_parallel_exec(self, exec_func: Callable, iterable: Iterable, *func_args, **kwargs):
        """
        Executes the given function in parallel for each element in the iterable.

        Args:
            exec_func (Callable): The function to be executed.
            iterable (Iterable): The iterable of elements.
            *func_args: Additional positional arguments to pass to the function.
            **kwargs: Additional keyword arguments to pass to the ThreadPoolExecutor.

        Keyword Args:
            max_workers (int): The maximum number of workers to use in the thread pool. Defaults to 100.

        Returns:
            dict or list: A dictionary containing the results of the function execution for each element in the iterable,
                or a list of tuples containing the element and any exception raised during execution.

        Raises:
            Exception: If an exception occurs while executing the function.

        """
        max_workers = kwargs.pop("max_workers", 100)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_element_map = {
                executor.submit(exec_func, element, *func_args): element
                for element in iterable
            }
            result = []
            for future in as_completed(future_element_map):
                element = future_element_map[future]
                try:
                    data = future.result()
                except Exception as exc:
                    print(
                        f"Got error while running parallel_exec: {element}: \n{exc}")
                    result.append((element, exc))
                else:
                    result.append((element, data))
            try:
                return dict(result)
            except Exception as e:
                return result

    def get_site_text(self, url):
        """
        Get the text content of a website.

        Args:
            url (str): The URL of the website to retrieve the text from.

        Returns:
            str: The text content of the website.
        """
        with requests.get(url) as response:
            return response.text

    def parse_site_text(self, site_text):
        """
        Parse the site text using BeautifulSoup.

        Parameters:
            site_text (str): The text of the website.

        Returns:
            str: The parsed text of the website.
        """
        soup = BeautifulSoup(site_text, 'html.parser')
        all_text = soup.get_text(separator='\n', strip=True)
        return all_text

    def parse_url_text(self, url):
        """
        Parse the text of a given URL.

        Parameters:
            url (str): The URL to parse.

        Returns:
            The parsed site text.
        """
        site_text = self.get_site_text(url)
        print("Parsing {}".format(url))
        return self.parse_site_text(site_text)

    def load_data(self):
        """
        Load data from the vector store if it exists, otherwise load data from sitemap.xml file and process it.

        Returns:
            vectorStore_openAI: The loaded data from the vector store or processed data from sitemap.xml.
        """
        if self.vector_store_path.exists():
            with open(self.vector_store_path, "rb") as file:
                return pickle.load(file)

        with open("sitemap.xml", "r") as file:
            xml_data = file.read()

        urls = re.findall(r'<loc>(.*?)</loc>', xml_data)

        if Path("url_texts.json").exists():
            with open("url_texts.json", "r") as file:
                url_texts = json.load(file)
        else:
            url_texts = ""

        if not url_texts:
            url_texts = self.run_parallel_exec(self.parse_url_text, urls)

        with open("url_texts.json", "w") as file:
            json.dump(url_texts, file)

        documents = [Document(page_content=text, metadata={
            "source": url}) for url, text in url_texts.items()]

        text_splitter = CharacterTextSplitter(separator='\n',
                                              chunk_size=1024,
                                              chunk_overlap=50)

        docs = text_splitter.split_documents(documents)

        embeddings = HuggingFaceInferenceAPIEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            # repo_id="sentence-transformers/all-mpnet-base-v2",
            api_key=self.HF_TOKEN,
        )

        vectorStore_openAI = FAISS.from_documents(docs, embeddings)

        with open(self.vector_store_path, "wb") as file:
            pickle.dump(vectorStore_openAI, file)

        return vectorStore_openAI

    def main(self, user_question: str):
        """
        Generates a function comment for the given function body.

        Args:
            user_question (str): The user's question.

        Returns:
            str: The generated function comment.

        Raises:
            None
        """
        query = user_question.strip().lower()

        if query == 'exit':
            return "Exiting chat. Goodbye!"

        prompt = (
            "You are a helpful assistant for the users of FiftyFive Technologies Ltd. You will be given a context and a question, "
            "and you will answer the question based on the context by formulating a brief and relevant answer. If the user sends 'Hello,' "
            "don't go through the context; only respond with 'Hi there.'\n\n"
            "Here is the context:\n{context}\n\n"
            "Question: {question}\n"
            "Response Length: Please ensure your response is within 100-120 words.\n"
        )

        docs = self.vector_store.similarity_search(query, k=3)
        context = "\n".join(doc.page_content for doc in docs)

        output = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            temperature=0,
            messages=[
                {"role": "user", "content": prompt.format(
                    context=context, question=query)}
            ]
        )

        sources = ', '.join(x.metadata.get("source") for x in docs)
        output_statement = output.choices[0].message.content + "\n"

        return output_statement


if __name__ == "__main__":
    assistant = Assistant()
    assistant.main()
