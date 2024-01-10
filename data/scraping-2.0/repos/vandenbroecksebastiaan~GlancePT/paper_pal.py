from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromiumService
from webdriver_manager.chrome import ChromeDriverManager
from webdriver_manager.core.utils import ChromeType
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup

from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from util import paper_pal_template

import argparse
import os
import requests
from requests.packages.urllib3.exceptions import InsecureRequestWarning
from typing import List
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import numpy as np
import PyPDF2
from sklearn.metrics.pairwise import cosine_similarity
import re
import pickle
import nltk
import annoy
import openai
openai.api_key = os.environ["OPENAI_API_KEY"]

from util import wos_category_names

class WebOfSciencePaper:
    """
    Represent an academic article with title, abstract, doi, authors and
    publication year. Provices methods to download and scrape the pdf.
    """
    def __init__(self, title, abstract, doi, authors, publication_year):
        self.title = title
        self.abstract = abstract
        self.process_abstract()
        self.doi = doi
        self.authors = authors
        self.year = publication_year
        self.source = self._get_source()
        self.abstract_embedding = None
        self.text_embedding = None

    def get_abstract_embedding(self, model):
        """Get the embedding of the abstract."""
        self.abstract_embedding = model.encode(self.abstract).tolist()

    def get_text_embedding(self, model):
        """Gets one embedding for every sentence in the text."""
        self.text_embedding = []
        for sentence in self.text:
            self.text_embedding.append(model.encode(sentence).tolist())

    # def create_or_load_annoy(self, n_trees_per_sentence):
    #     annoy_files = [i.replace(".ann", "") for i in os.listdir("data/annoy_files")]
    #     self.annoy = annoy.AnnoyIndex(768, "angular")
    #     if not self.unavailable:
    #         n_trees = n_trees_per_sentence * len(self.text)
    #         if self.title not in annoy_files:
    #             for idx, vector in enumerate(self.text_embedding):
    #                 self.annoy.add_item(idx, vector)
    #             self.annoy.build(n_trees)
    #             self.annoy.save(f"data/annoy_files/{self.title}.ann")
    #         else:
    #             self.annoy.load(f"data/annoy_files/{self.title}.ann")

    def _init_driver(self):
        # I have to reconstruct the driver for every paper
        options = webdriver.ChromeOptions()
        profile = {"plugins.plugins_list": [{"enabled": False,
                                             "name": "Chrome PDF Viewer"}],
                   "download.default_directory": "/home/sebastiaan/fun/GlossPT",
                   "download.extensions_to_open": "",
                   "plugins.always_open_pdf_externally": True}

        options.add_experimental_option("prefs", profile)
        options.add_argument('--headless')
        options.add_argument("--remote-debugging-port=9222") # do not delete
        options.add_argument("--window-size=800,800")
        options.add_argument('--disable-blink-features=AutomationControlled')
        driver = webdriver.Chrome(
            options=options,
            service=ChromiumService(
                ChromeDriverManager(chrome_type=ChromeType.CHROMIUM).install()
            )
        )
        return driver

    def _get_soup(self, driver):
        """Get html from sci-hub. Enter the search term and click the search
           button. Return the html."""
        base_url = "https://sci-hub.ru/"
        driver.get(base_url)
        # Wait until button becomes visible
        # from selenium.webdriver.support.ui import WebDriverWait
        # from selenium.webdriver.support import expected_conditions as EC
        # btn = WebDriverWait(driver, 10)\
        #         .until(EC.presence_of_element_located((By.ID, "request")))
        # Search for the paper
        search_target = self.doi if self.doi!="" else self.title
        search_inputs = driver.find_elements(By.ID, "request")

        # I have since found out that this is for a DDOS protection
        # In this case, the pdf might be available, but we cannot access it
        # So the title should not be added to unavailable_papers.txt

        if len(search_inputs) == 0:
            return None

        search_inputs[0].send_keys(search_target)
        # Find and click the search button
        buttons = driver.find_elements(By.TAG_NAME, "button")
        buttons[0].click()
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        return soup

    def _scrape_pdf(self):
        """Scrape the pdf from sci-hub using the DOI or title if it is not
           available."""
        # Construct driver, enter search term, go to next page, return html
        base_url = "https://sci-hub.ru/"
        driver = self._init_driver()
        soup = self._get_soup(driver)

        # There is no search bar on the page
        # This callback ensures that the driver can be closed
        # if soup == 1:
        #     driver.quit()
        #     self.pdf_available = False
        #     raise PaperNotFoundException

        # If there is no button, sci-hub does not have this paper and did not
        # return a page with a pdf
        result = soup.find_all('button')
        if len(result) == 0:
            driver.quit()
            raise PaperNotFoundException

        # If there is a button, sci-hub has the paper and we download the pdf
        if "onclick" in result[0].attrs.keys():
            link = result[0]["onclick"]
            link = link.replace("location.href='", "").replace("'", "")
            if "sci-hub" not in link:
                base_url = base_url[:-1]
                link = base_url + link
            else:
                link = link.replace("//", "")
                link = "https://" + link
            requests.packages.urllib3.disable_warnings(InsecureRequestWarning)
            response = requests.get(link, verify=False) # Yes, I know
            driver.quit()
            if response.status_code == 200:
                with open(f"data/pdf_files/{self.title}.pdf", "wb") as file:
                    file.write(response.content)
                    driver.quit()
                    return
            # This is in the case of a 404 error, even if we found the link
            else:
                driver.quit()
                raise PaperNotFoundException

        driver.quit()
        raise PaperNotFoundException

    def get_pdf(self):
        """Try to find the pdf in the pdf_files folder. If it is not there,
           scrape sci-hub and raise an exception if it is not available."""
        # If the pdf has already been downloaded, get that
        pdf_files = os.listdir("data/pdf_files")
        pdf_files = [i.replace(".pdf", "") for i in pdf_files]

        if self.title in pdf_files:
            pass
        if self.title not in pdf_files:
            self._scrape_pdf()

    def process_pdf(self):
        """Read the pdf and extract the text."""
        # reader = PyPDF2.PdfReader(BytesIO(self.pdf))
        reader = PyPDF2.PdfReader(open(f"data/pdf_files/{self.title}.pdf", "rb"))
        pdf_text = [reader.pages[i].extract_text() for i in range(len(reader.pages))]
        pdf_text = "".join(pdf_text)
        pdf_text = re.sub("\s+", " ", pdf_text)
        pdf_text = pdf_text.replace("\n", "") \
                           .replace("\t", "") \
                           .replace("- ", "") \
                           .replace("  ", " ") \
                           .encode("ascii", "ignore") \
                           .decode("ascii")

        # If the text is empty, raise a PaperNotFoundException
        if len(pdf_text) < 100: raise PaperNotFoundException

        # Delete the references section
        if "references" in pdf_text.lower():
            references_index = pdf_text.lower().rfind("references")
            pdf_text = pdf_text[:references_index]

        sentences = nltk.tokenize.sent_tokenize(pdf_text)
        sentences = [i for i in sentences if len(i) > 30]
        sentences = [i for i in sentences if i != None]
        sentences = [i for i in sentences if "keywords" not in i.lower()]
        sentences = [i for i in sentences if "arxiv" not in i.lower()]
        # Remove all sentences with a link in them
        sentences = [i for i in sentences if not bool(re.search(r'https?://\S+|www\.\S+', i))]
        sentences = [i.strip() for i in sentences]

        self.text = sentences

    def _get_source(self):
        if len(self.authors) == 1:
            author = self.authors[0].split(",")[0]
            source = f"({author}, {self.year})"
        else:
            author = self.authors[0].split(",")[0]
            source = f"({author} et al., {self.year})"
        return  source

    def process_abstract(self):
        self.abstract = nltk.tokenize.sent_tokenize(self.abstract)
        # Make self.abstract into an nested list with overlapping strings
        self.abstract = [self.abstract[i:i+3] for i in range(len(self.abstract)-2)]


class PaperNotFoundException(Exception):
    def __init__(self):
        pass


class WebOfScienceScraper:
    """
    Scrapes a list of tab delimited files that has been exported from Web of
    Science.
    """
    def __init__(self):
        self.data = None # [[title, abstract, doi]]

    def read_tdf(self, file_paths: List[str], top_n: int):
        # TODO: implement path as last of multiple files
        data = []
        for path in file_paths:
            with open(path, "r") as file:
                data.extend(file.readlines())

        data = data[:top_n]
        data = [i.replace("\ufeff", "").replace("\n", "") for i in data]
        data = [i.split("\t") for i in data]

        # Remove all columns except title, abstract, and doi
        idx_to_keep = []
        for k in wos_category_names.keys():
            try:
                idx = data[0].index(k)
                idx_to_keep.append(idx)
            except ValueError:
                continue
        data = [[i[j] for j in idx_to_keep] for i in data]

        # Delete obs with no abstract
        data = [i for i in data if len(i[1]) > 0]

        # Titles can't be longer than 255 characters
        for i in range(len(data)):
            if len(data[i][1]) > 200:
                data[i][1] = data[i][0][:200]

        data = [i for i in data if i[0] != "A comparison between Fuzzy Linguistic RFM Model and traditional RFM model applied to Campaign Management. Case study of retail business."]
        data = [i for i in data if i[0] != "Predicting Customer Profitability Dynamically over Time: An Experimental Comparative Study"]

        # Make a dict out of each row of data
        data[0] = [wos_category_names[i] for i in data[0]]
        data = [dict(zip(data[0], i)) for i in data[1:]]

        for i in data: i["Authors"] = i["Authors"].split(";")
        for i in data: i["title"] = i["title"].replace("'", "").replace("/", "")

        self.data = data[1:]

    def _delete_duplicates(self, papers: List[WebOfSciencePaper]):
        titles = [i.title for i in papers]
        titles, idx = np.unique(titles, return_index=True)
        return [papers[i] for i in idx]

    def generate_papers(self) -> List[WebOfSciencePaper]:
        # Construct the papers
        papers = [
            WebOfSciencePaper(
                title=i["title"], abstract=i["abstract"], doi=i["DOI"],
                publication_year=i["Year Published"], authors=i["Authors"]
            ) for i in self.data
        ]
        # Delete duplicates
        papers = self._delete_duplicates(papers)
        return papers


class PaperCollection:
    def __init__(self, papers: List[WebOfSciencePaper]):
        self.papers = papers
        self.model = SentenceTransformer("all-mpnet-base-v2")
        self._process_papers()

    def _process_papers(self):
        processed_papers = []
        saved_papers = [i.replace(".pt", "") for i in os.listdir("data/papers/")]

        for paper in tqdm(self.papers, desc="Processing papers" ,leave=False):
            if paper.title in saved_papers:
                paper = pickle.load(open(f"data/papers/{paper.title}.pt", "rb"))
            else:
                try:
                    paper.get_pdf()
                    paper.process_pdf()
                    paper.get_text_embedding(self.model)
                    paper.unavailable = False
                except PaperNotFoundException:
                    with open("data/unavailable_titles.txt", "a") as file:
                        file.write(paper.title + "\n")
                    paper.unavailable = True 

                with open(f"data/papers/{paper.title}.pt", "wb") as file:
                    pickle.dump(paper, file)

            processed_papers.append(paper)

        self.papers = processed_papers

    def query(self, query: str, n_results=10):
        """Returns the most similar sentences in the collection of the query."""
        query_embedding = self.model.encode(query).tolist()

        # Directly query the embedding instead of annoy
        all_sentences = []
        all_similarities = []
        all_sources = []
        for paper in self.papers:
            if not paper.unavailable:
                similarities = cosine_similarity(
                    np.array(paper.text_embedding),
                    np.array(query_embedding).reshape(1, -1)
                ).tolist()
                all_sentences.extend(paper.text)
                all_similarities.extend(similarities)
                all_sources.extend([paper.source]*len(paper.text))

        # Return the top results
        # TODO: return a few sentences after the most similar sentence
        assert len(all_sentences) == len(all_similarities) == len(all_sources)
        data = sorted(zip(all_similarities, all_sentences, all_sources),
                      key=lambda x: x[0], reverse=True)
        # Grab the results and slap the source on that bad boy
        # top_sentences = [i[1] +  " SOURCES " + str(i[2]) for i in data][:n_results]
        top_sentences = [i[1] for i in data][:n_results]
        return top_sentences

class PaperPal():
    """
    This the chatbot class.
    """
    def __init__(self, topics: str, file_paths: List[str], n_papers: int):
        self.topics = topics
        self.n_papers = n_papers
        self.model = SentenceTransformer("all-mpnet-base-v2")
        # Scrape the data file
        top_papers = self.load_papers(file_paths, n=5000)
        self.paper_collection = PaperCollection(top_papers)
        print("Number of available papers:",
              len([i for i in self.paper_collection.papers if not i.unavailable]))

    def load_papers(self, file_paths: List[str], n: int):
        scraper = WebOfScienceScraper()
        scraper.read_tdf(file_paths=file_paths, top_n=n)
        self.papers = scraper.generate_papers()
        print("Number of scraped papers from csv files:", len(self.papers))

        # For each paper, create its abstract embedding
        # Check if the paper has been pickled and if so load it
        pickled_papers = [i.replace(".pt", "") for i in os.listdir("data/papers")]
        print("Number of pickled papers:", len(pickled_papers))
        pbar = tqdm(enumerate(self.papers), desc="Loading papers",
                    total=len(self.papers))
        for idx, paper in pbar:
            if paper.title in pickled_papers:
                paper = pickle.load(open(f"data/papers/{paper.title}.pt", "rb"))
                self.papers[idx] = paper
            if paper.abstract_embedding is None:
                paper.get_abstract_embedding(self.model)
                self.papers[idx] = paper

        self.papers = [i for i in self.papers if len(i.abstract_embedding) > 0]
        print("Number of papers with abstract embeddings:", len(self.papers))

        # Create the query embedding
        query_embedding = self.model.encode(self.topics).tolist()
        # Calculate the angle between them
        similarities = []
        for paper in self.papers:
            similarity = cosine_similarity(paper.abstract_embedding,
                                           [query_embedding]*len(paper.abstract_embedding))
            similarities.append(similarity.mean())
        # Return the most similar papers to the topics
        data = sorted(zip(similarities, [i.title for i in self.papers]),
                      key=lambda x: x[0], reverse=True)
        top_titles = [i[1] for i in data[:self.n_papers]]
        top_papers = [i for i in self.papers if i.title in top_titles]
        return top_papers

    def chat(self):
        prompt = PromptTemplate(input_variables=["input", "text", "history"],
                                template=paper_pal_template)
        memory = ConversationBufferWindowMemory(k=5, input_key="input",
                                                memory_key="history")
        chain = LLMChain(llm=OpenAI(temperature=0), prompt=prompt, memory=memory,
                         verbose=True)

        print("Hello! I am PaperPal. I can help you explore the literature on a "
              "topic of your choice. Type your question below or 'quit' to exit.")

        while True:
            user_input = input(">>> ")
            if user_input == "quit":
                break
            else:
                top_sentences = self.paper_collection.query(user_input, n_results=30)
                top_sentences = " \n".join(top_sentences)
                output = chain.predict(text=top_sentences, input=user_input)
                print(output)


def main():
    # Add command line arguments
    parser = argparse.ArgumentParser(description="PaperPal")
    parser.add_argument("--reset_papers", "--reset_papers", action="store_true", default=False)
    parser.add_argument("--reset_annoy", "--reset_annoy", action="store_true", default=False)
    args = parser.parse_args()

    # What are the new contributions
    # Summary of the paper
    # What are the limitations
    # What are the future directions
    # What are the most similar papers

    if args.reset_papers:
        for i in os.listdir("data/papers"): os.remove("data/papers/" + i)
    if args.reset_annoy:
        for i in os.listdir("data/annoy_files"): os.remove("data/annoy_files/" + i)

    paper_pal = PaperPal(topics="Recency, frequency, monetary, RFM features",
                         file_paths=["data/savedrecs_1_RFM.txt", "data/savedrecs_2_RFM.txt"],
                         n_papers=1000)
    paper_pal.chat()


if __name__ == "__main__":
    main()
 