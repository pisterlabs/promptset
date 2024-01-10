from googlesearch import search
from bs4 import BeautifulSoup
import requests
from textwrap import dedent
from typing import Sequence, Any, Type
# from ctransformers import AutoModelForCausalLM as LLM
from llm_client import LocalLLMClient as LLM
from llm_client import LocalEmbeddingsClient as Encoder
from qa import answer_query
from page_processor import process_soup
from pydantic import BaseModel
from langchain.text_splitter import CharacterTextSplitter
from VectorDb import VectorDB, FaissDB
from typing import List

select_link_template = dedent('''
You are a helpful agent and very skilled with doing online searches. Your task is to find the link in which it will be the most likely to find the answer to an user's question.
{possible_criteria}

The user asked the following question {query}

A Google search yielded the following results
{links}

Please select the link that will most likely contain the answer to the question. Please respond only with the selected link.

''')

def select_link_to_click(llm : LLM, query : str , links : Sequence[str], criteria = []):

    formatted_links = "\n".join(links)
    
    if len(criteria) == 0:
        possible_criteria = ""
    else:
        possible_criteria = (
            "You should take into consideration the following criteria:"
            + 
            "\n".join(criteria)
        )

    prompt = select_link_template.format(query = query, links = formatted_links, possible_criteria = possible_criteria)

    result = llm(prompt, temperature = 0)

    return result

def load_page(page_url : str) -> BeautifulSoup:

    server_response = requests.get(page_url)

    return BeautifulSoup(server_response.content, features="lxml")

def index_page(processed_page : str, encoder : Encoder, vector_db_type : Type[VectorDB] = FaissDB) -> VectorDB:

    splitted = CharacterTextSplitter(chunk_size = 200).split_text(processed_page)
    print("Begin encoding document chunks...", end=" ")
    feature_matrix = encoder.encode(splitted)
    print("Finished")
    return vector_db_type(splitted, feature_matrix, encoder)


class SearchTool(BaseModel):

    name : str = "Search"
    description : str = "This is a tool that allows us to search something in the internet. It takes a query and returns a sentence that answers the question."
    search_criteria : List[str] = [
        "Prioritize results from official sources, like government websites",
        "Avoid wikipedia"   
    ]
    llm : Any
    encoder : Any

    def run(self,query : str) -> str:

        search_results = list(search(query, stop = 3))
        selected_link = select_link_to_click(self.llm, query,search_results, self.search_criteria)
        soup = load_page(selected_link)

        page = process_soup(soup)

        index = index_page(page, self.encoder, FaissDB)

        answer = answer_query(query, self.llm, index)

        return answer

    async def run_async(self, topic : str) -> str:
        raise NotImplementedError("This tool does not support async")


if __name__ == "__main__":
    '''
    We can test the seach tool directly here by running this script!
    '''

    encoder = Encoder("http://127.0.0.1:8000")
    
    # we are using mistral instruct 7b prompt template
    llm = LLM("http://127.0.0.1:8000", prompt_template = "<s>[INST] {prompt} [/INST]")

    query = "What is Johnny Sins's penis size?"

    # criteria that will affect which internet links will be clicked (those links are provided by Google)
    search_criteria = [
        "Prioritize results from official sources, like government websites",
        "Avoid wikipedia"   
    ]

    search_results = list(search(query, stop = 3))
    print(search_results)

    selected_link = select_link_to_click(llm, query,search_results, search_criteria)
    print(selected_link)
    soup = load_page(selected_link)

    page = process_soup(soup)

    index = index_page(page, encoder, FaissDB)

    answer = answer_query(query, llm, index)

    print("answer :", answer)
