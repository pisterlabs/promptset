import difflib
import random
from utils import logger, json2string
from typing import List
import os
import datetime

PAPER_NOT_FOUND_INFO = "Sorry, we cannot find the paper you are looking for."
COLLECTION_NOT_FOUND_INFO = "Sorry, we cannot find the paper collection you are looking for."
RETRIEVE_NOTHING_INFO = "Sorry, we retrieve no relevant paper for your query."
ERRORS = [PAPER_NOT_FOUND_INFO, COLLECTION_NOT_FOUND_INFO]

uid = 'test_user' 

# load paper_corpus.json
paper_pickle_path = '/data/survey_agent/paper_corpus.pkl' #os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data' , 'paper_corpus.pkl') 

import json
import time
import pickle 
from langchain.retrievers import BM25Retriever
from langchain.schema import Document

from langchain.docstore.document import Document
import os



print("="*10 + f"准备开始 - 时间3.1: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}" + "="*10 )

if not os.path.exists(paper_pickle_path):
    
    paper_corpus_path='/data/survey_agent/processed_data'
    paper_corpus_json = []
    for filename in os.listdir(paper_corpus_path):
        file_path = os.path.join(paper_corpus_path, filename)
        if filename.endswith('.json') and os.path.isfile(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                try:
                    paper_corpus_json += json.load(file)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON in file {filename}: {e}")
    standard_keys = ['authors','title','url','abstract','arxiv_id','published_date','year','source','institution','introduction','conclusion','full_text'] 
    paper_corpus = { p['title']:{key: p[key] if key in p and p[key] is not None else "" for key in standard_keys} for p in paper_corpus_json }

    paper_docs = [] #[ Document(page_content=p['full_text'], metadata={k:p[k] for k in ['title']})]
    for title, p in paper_corpus.items():
        page_content = p['full_text']
        # 将p['full_text']划分为多个段落，每个段落1000个字符，naive
        page_content_pieces = [page_content[i:i+1000] for i in range(0, len(page_content), 1000)]

        paper_docs.extend([Document(page_content=page_content_piece, metadata={**{k:p[k] for k in ['title']}, **{'ith_piece': i}}) for i, page_content_piece in enumerate(page_content_pieces)])

    retriever = BM25Retriever.from_documents(paper_docs)


    # save paper_corpus, paper_docs and retriever into one pkl file
    with open(paper_pickle_path, 'wb') as f:
        pickle.dump([paper_corpus, paper_docs, retriever], f)

else:
    # and load it, would it be quicker? 72s -> 27s, 2.7x faster
    with open(paper_pickle_path, 'rb') as f:
        paper_corpus, paper_docs, retriever = pickle.load(f)

print("="*10 + f"准备开始 - 时间3.2: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}" + "="*10 )

from utils import _sync_paper_collections

# load paper collections data
paper_collections = _sync_paper_collections()


def get_papers_and_define_collections(paper_titles: List[str], paper_collection_name: str) -> str:
    """
    This function processes a list of paper titles, matches them with corresponding entries in the database, and defines a collection of papers under a specified name. 
    Note that:
        1. If certain papers are not found, do not attempt to use the search_papers function again to look for those papers. 
        2. Only use this function when the user inputs a list of papar titles. Do not use it without explicit intention from the user.

    Args:
        paper_titles (List[str]): A list of paper titles to be searched in the database.
        paper_collection_name (str): The name to be assigned to the newly defined collection of papers.

    Returns:
        str: A string of JSON format representing information of the found papers, including their title, authors, year and url.
    """

    found_papers = _get_papers_by_name(paper_titles)
    _define_paper_collection(found_papers, paper_collection_name, uid)
    return _display_papers(found_papers, paper_collection_name, user_inputs=paper_titles)

def _get_paper_content(paper_name, mode):
    """Get text content of a paper based on its exact name."""
    if paper_name in paper_corpus.keys():
        if mode == 'full':
            return paper_corpus[paper_name]['full_text'] 
        else: # == 'abstract':
            return paper_corpus[paper_name]['abstract'] 
    else:
        return PAPER_NOT_FOUND_INFO

def get_paper_content(paper_name: str, mode: str) -> str:
    """
    Retrieve the content of a paper. Set 'mode' as 'full' for the full paper, or 'abstract' for the abstract.  

    Args:
        paper_name (str): The name of the paper.
        mode (str): The mode of retrieval - 'full' for the complete text of the paper, 
                    or 'abstract' for the paper's abstract.

    Returns:
        str: If the paper is found, returns its full text or abstract based on the mode. 
             If not found, returns information indicating the paper was not found.
    """
    paper_name = _get_papers_by_name([paper_name])[0]
    if paper_name:
        return _get_paper_content(paper_name, mode)
    else:
        return PAPER_NOT_FOUND_INFO

def _get_paper_metadata(paper_name):
    """Get metadata of a paper based on its exact name."""
    if paper_name in paper_corpus.keys():
        return { key: paper_corpus[paper_name][key] for key in ['title', 'authors', 'year', 'url'] }
    else:
        return PAPER_NOT_FOUND_INFO

def get_paper_metadata(paper_name: str) -> str:
    """
    Retrieve the metadata of a paper, including its title, authors, year and url.

    Args:
        paper_name (str): The name of the paper.

    Returns:
        str: If the paper is found, returns its metadata. 
             If not found, returns information indicating the paper was not found.
    """

    paper_name = _get_papers_by_name([paper_name])[0]
    if paper_name:
        return _get_paper_metadata(paper_name)
    else:
        return PAPER_NOT_FOUND_INFO 
    
    
def _get_papers_by_name(paper_titles):
    """Find corresponding papers based on a list of fuzzy paper names."""
    found_papers = []
    for fuzzy_name in paper_titles:
        matches = difflib.get_close_matches(fuzzy_name, paper_corpus.keys(), n=1, cutoff=0.8)
        # 这个matches还挺慢的，@鑫凤 可不可以优化一下？
       
        if matches:
            found_papers.append(matches[0])
        else:
            found_papers.append(None)  # Append None if no matching paper is found
    # log relevant information
    #found_papers = [p for p in found_papers if p]  # Remove None from the list
    logger.info(f"Found {len([p for p in found_papers if p])} papers out of {len(paper_titles)}")
    
    return found_papers

def _display_papers(paper_titles, paper_collection_name, user_inputs=None, paper_content=None):

    """Display paper information based on a list of exact paper names."""
    paper_info = []
    if user_inputs:
        # called by get_papers_and_define_collections, where there might be None in paper titles
        for paper_name, user_input_name in zip(paper_titles, user_inputs):
            if paper_name:
                paper_info.append(_get_paper_metadata(paper_name))
                paper_info[-1]['authors'] = ', '.join(paper_info[-1]['authors'])
            else:
                paper_info.append({'title': user_input_name, 'status': PAPER_NOT_FOUND_INFO})
    else:
        # called by other functions, where paper titles are all valid
        for i, paper_name in enumerate(paper_titles):
            paper_info.append(_get_paper_metadata(paper_name))
            paper_info[-1]['authors'] = ', '.join(paper_info[-1]['authors'])
            if paper_content:
                paper_info[-1]['relevant content'] = paper_content[i].page_content

    return json2string({'Collection': paper_collection_name, 'Papers': paper_info})

def _define_paper_collection(found_papers, paper_collection_name, uid):
    """Define a paper list based on a list of exact paper names."""
    found_papers = list(set([p for p in found_papers if p]))  # Remove None from the list
    
    # if found_papers are empty
    if not found_papers:
        logger.info(f"The target paper collection {paper_collection_name} is empty.")
        return False

    paper_collections.setdefault(uid, {}) 
    if paper_collection_name in paper_collections[uid]:
        # paper_collection_name already exists, use a random name instead
        import string 
        import random 
        new_paper_collection_name = ''.join(random.choice(string.ascii_letters + string.digits) for i in range(5)) + paper_collection_name
        logger.info(f"Paper collection name {paper_collection_name} already exists, use a random name {new_paper_collection_name} instead.")
        paper_collection_name = new_paper_collection_name

    paper_collections[uid][paper_collection_name] = found_papers
    # log relevant info
    logger.info(f"Paper collection {paper_collection_name} created for user {uid} with {len(found_papers)} papers.")
    _sync_paper_collections(paper_collections)
    return True 

    
def _get_papercollection_by_name(collection_name, uid):
    """Find the name of the paper collection that best matches a fuzzy collection name."""
    # Find the closest match for the collection name
    
    paper_collection_name_simplified = {p.replace('search results of ', '').replace('recommend results of ', ''): p for p in paper_collections[uid].keys()}
    match = difflib.get_close_matches(collection_name, paper_collection_name_simplified.keys(), n=1, cutoff=0.7)

    match = [paper_collection_name_simplified[m] for m in match]
    
    if match:
        return match[0]
    else:    
        return COLLECTION_NOT_FOUND_INFO


def _get_collection_papers(collection_name, uid):
    paper_titles = paper_collections[uid][collection_name]
    return paper_titles

def get_papercollection_by_name(collection_name: str) -> str:
    """
    Retrieve a specified paper collection by its name, display the paper collection's name and information of its papers.
    Only use this function when the user explicitly asks for information about the collection. Avoid using this when the user poses a request about the collection, in which case the agent should use 'query_based_on_paper_collection' instead.

    ===
    When the user's request pertains to inquiring about a research field, this action has a higher priority than search_papers. That is, the agent should first check if there is an existing paper collection related to the research area. If not, then the agent should proceed to find papers using the search_papers function.

    Args:
        collection_name (str): The name of the paper collection.

    Returns:
        str: If the collection is found, returns a string containing the collection's name 
             and information of some papers from the collection for display. 
             If the collection is not found, returns a string with information 
             indicating the collection was not found.
    """

    paper_collection_name = _get_papercollection_by_name(collection_name, uid)
    if paper_collection_name == COLLECTION_NOT_FOUND_INFO:
        return COLLECTION_NOT_FOUND_INFO
    else:
        collection_papers = _get_collection_papers(paper_collection_name, uid)[:3]
        return _display_papers(collection_papers, paper_collection_name)
    

def update_paper_collection(target_collection_name: str, source_collection_name: str, paper_indexes: str, action: str) -> bool:
    """
    Updates the target paper collection based on a specified action ('add' or 'del') and paper indices (Indices start from 0. The format should be comma-separated, with ranges indicated by a dash, e.g., "0, 2-4") from the source collection.

    Args:
        target_collection_name (str): The name of the target collection to be updated.
        source_collection_name (str): The name of the source collection where papers will be taken from.
        paper_indexes (str): A string representing the indices of papers in the source collection to be used in the action. Indices start from 0. The format should be comma-separated, with ranges indicated by a dash (e.g., "0, 2-4").
        action (str): The action to perform - either "add" to add papers to the target collection or "del" to delete them.

    Returns:
        bool: True if the update operation was successful, False otherwise.
    """

    _target_collection_name = _get_papercollection_by_name(target_collection_name, uid)
    if _target_collection_name:
        target_collection_name = _target_collection_name 

    
    source_collection_name = _get_papercollection_by_name(source_collection_name, uid)
    if source_collection_name == COLLECTION_NOT_FOUND_INFO:
        logger.error(f"Source collection {source_collection_name} does not exist.")
        return False

    """Update target paper collection based on provided action and paper numbers from the source collection. """
    # Convert string paper_indexes to actual list of indices
    indices = []
    for part in paper_indexes.split(','):
        if '-' in part:
            start, end = map(int, part.split('-'))
            indices.extend(range(start, end + 1))
        else:
            indices.append(int(part))
    
    # sort and remove duplicates
    indices = sorted(list(set(indices)))

    # Get the current list of papers for the user
    paper_collections.setdefault(uid, {})
    target_collection = paper_collections.get(uid, {}).get(target_collection_name, [])
    try:
        source_collection = paper_collections[uid][source_collection_name]
    except:
        raise ValueError(f"Source collection {source_collection_name} does not exist.")
    
    update_papers = [source_collection[i] for i in indices]
    target_collection_set = set(target_collection)
    
    # Perform the add or delete action
    if action == "add":
        target_collection += [p for p in update_papers if p not in target_collection_set]
    elif action == "del":
        target_collection = [p for p in target_collection if p not in update_papers]

    paper_collections[uid][target_collection_name] = target_collection
    _sync_paper_collections(paper_collections)
    return True

num_retrival = 3

print("="*10 + f"准备开始 - 时间3.4: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}" + "="*10 )

def _retrieve_papers(query):
    result = retriever.get_relevant_documents(query)
    if len(result) > 0:
        return result[:num_retrival]
    else:
        return None
    
def retrieve_papers(query: str) -> str:
    """
    Retrieve the most relevant content in papers based on a given query, using the BM25 retrieval algorithm. Output the relevant paper and content. This function should be used when the query is about a specific statement, rather than being composed of keywords.

    Args:
        query (str): The search query used to find the most relevant paper.

    Returns:
        str: The relevant paper and content. 
    """
    result = _retrieve_papers(query)

    if result:
        found_papers = [p.metadata['title'] for p in result]
        paper_collection_name = f'<BM25 results of query "{query}">'
        _define_paper_collection(found_papers, paper_collection_name, uid)

        return _display_papers(found_papers, paper_collection_name, paper_content=result) 
    else:
        return RETRIEVE_NOTHING_INFO


if __name__ == '__main__':

    print('retrieve_papers: ', retrieve_papers('''what is Numerical Question Answering?'''))

    
    print('get_papers_and_define_collections: ', get_papers_and_define_collections(paper_titles=["Semantic Relation Classification via Bidirectional LSTM Networks with Entity-aware Attention using", 'Robust Numerical Question Answering: Diagnosing Numerical Capabilities of NLP', 'Does Role-Playing Chatbots Capture the Character Personalities? Assessing Personality Traits for Role-Playing'], paper_collection_name='Paper Collection 123'))
    

    print('get_papercollection_by_name: ', get_papercollection_by_name("Paper Collection "))
    
    
    print('_get_paper_content: ', _get_paper_content('Towards Robust Numerical Question Answering: Diagnosing Numerical Capabilities of NLP Systems', mode='abstract'))

    print('update_paper_collection ', update_paper_collection('123 asd Papers', 'Paper Collection 123', '1-2', 'del'))
    


    