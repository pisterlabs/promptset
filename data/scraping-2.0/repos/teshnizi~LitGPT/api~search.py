import openai
import os
import numpy as np
from langchain.embeddings import OpenAIEmbeddings
import sys
import arxiv
from typing import List, Tuple
import json
from scholarly import scholarly

# ======================================================================================================
# Uncomment this section to load the fetched embeddings from the files and save them in a single file
# ======================================================================================================

# folder = 'api/database/'
# files = os.listdir(folder)
# files = [f for f in files if f.endswith('.npy')]
# files = sorted(files)


# global_idx = 0
# emb_matrix = []
# paper_ids = {}


# for f in files:
#     if f.startswith('papers_all'):
#         continue
#     print(f"Loading file {f}...")
    
#     embs = []
#     papers = np.load(folder + f, allow_pickle=True).item()
#     for idx in papers:
#         embs.append(papers[idx]['embedding'])
#         paper_ids[global_idx + idx] = papers[idx]['paper_id'].split('/')[-1]
#     embs = np.array(embs)
#     emb_matrix.append(embs)

#     global_idx += len(papers)
    
# emb_matrix = np.concatenate(emb_matrix, axis=0)
# emb_matrix = np.array(emb_matrix, dtype=np.float16)

# # save to file
# np.save('api/data/emb_matrix.npy', emb_matrix)
# np.save('api/data/paper_ids.npy', paper_ids)

# ======================================================================================================
# ======================================================================================================

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def get_paper_details(paper_ids: List[str]) -> List[Tuple[str, str]]:
    # paper_ids = [link.split('/')[-1] for link in paper_links]

    search = arxiv.Search(
        query="",
        id_list=[str(paper_id) for paper_id in paper_ids],
        max_results=len(paper_ids),
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )

    paper_details = []
    for paper in search.results():
        paper_details.append((paper.title, paper.summary, [author.name for author in paper.authors], paper.published, paper.entry_id))

    return paper_details
    

def get_relevant_papers(query_title, query_abstract, key, emb_matrix, num_papers=15):
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=key)
    query = f"Title: {query_title}\n ===== Abstract: {query_abstract}\n\n"
    emb = np.array(embeddings.embed_query(query)).reshape(1, -1) # (1, 1536_)
    emb = np.array(emb, dtype=np.float16)
    id_list = np.argsort(-np.dot(emb, emb_matrix.T), axis=1)[0][:num_papers]
    # create a dict with titles, authors, year, and link:
    return id_list

def search_google_scholar(query, num_papers=15):
    search_results = scholarly.search_pubs(query)
    papers = []
    authors = {}
    
    for i, paper in enumerate(search_results):
        if i >= 15:
            break
        try:
            gist = {
                'title': paper['bib'].get('title', ''),
                'author': paper['bib'].get('author', ''),
                'year': paper.get('pub_year', paper['bib'].get('pub_year', '')),
                'url': paper.get('pub_url', paper['bib'].get('url', '')),
                'citations': paper['num_citations'],
            }
            author_ids = paper.get('author_id', paper['bib'].get('author_id', []))
                        
            for id in author_ids:
                if id == '':
                    continue
                if id not in authors:
                    authors[id] = (0, 0)
                authors[id] = (authors[id][0] + 1, authors[id][1] + gist['citations'])
                   
            # print(gist)
            papers.append(gist)
        except Exception as e:
            print(e)

    print(authors)
    ret_authors = []
    # get info for the top 5 authors (by number of papers, and number of citations to break ties)
    for i, author in enumerate(sorted(authors, key=lambda x: (-authors[x][0], -authors[x][1]))):
        if i >= 4:
            break
        ret_authors.append(scholarly.search_author_id(author))
        
    return papers, ret_authors

# Example usage

# print('=-=-=-=-=-=-=-=-=-=')
# print(papers)
# print(authors)

    
def get_search_terms(query_title, query_abstract, key, model="gpt-3.5-turbo", num_papers=20):
    prompt = "I want to do a literature review to understand the research space around this paper:\n\n"
    prompt += f"=====\nTitle: {query_title}\nAbstract: {query_abstract}\n\n===== \n\n"
    prompt += "I want you to generate 3 to 5 research sub-areas that I should explore. For each sub-area generate a google scholar search terms that will find the papers related to the main one."
    prompt += "Use a json format like this:\n\n"
    prompt += "=====\n"
    prompt += "{\n"
    prompt += "\"sub-area 1\": \"search term 1\",\n"
    prompt += "\"sub-area 2\": \"search term 1\",\n"
    prompt += "...,\n"
    prompt += "\"sub-area 5\": \"search term 1\",\n"
    prompt += "}\n" 
    prompt += "=====\n\n"  
    prompt += "Just generate the json file and nothing else."
    
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
                {"role": "user", "content": prompt},
            ]
    )
    
    res = response["choices"][0]["message"]["content"]
    
    search_terms = json.loads(res)
    return search_terms
    
    
def get_paper_lists(search_terms):
    lists = {}
    for sub_area in search_terms:
        papers, authors = search_google_scholar(search_terms[sub_area])
        lists[sub_area]['papers'] = papers
        lists[sub_area]['authors'] = authors
    return lists    
    

def generate_literature(query_title, query_abstract, id_list, key, paper_ids, model="gpt-3.5-turbo"):
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=key)
    citations = []
    
    context = ""
    # print(paper_ids['270379'])
    
    id_list = [paper_ids[id] for id in id_list]
    related_papers = get_paper_details(id_list)
    print(related_papers)
    
    for id, paper in enumerate(related_papers):
        title, abs, authors, date, link = paper
        abs = abs.replace('\n', ' ')
        
        if model != 'gpt-4':
            if len(abs) > 1500:
                abs = abs[:1500] + " ..."
        
        context += "====="
        context += f"ID: {id+1}\n"
        context += f"-Title: {title}\n-Abstract: {abs}\n\n"
        
        citations.append({'id': str(id+1),
                          'title': title, 
                          'authors': [' ' + author for author in authors], 
                          'date': str(date.date()),
                          'link': link,})
        
    context += "=====\n"

    paper = f"-Title: {query_title}\n-Abstract: {query_abstract}\n\n"
    
    
    prompt = "I want you to write the related work section of a paper, using a list of related papers. You can only use these papers listed. Use this format \"apples are red [1,7]\" where numbers are paper IDs."
    prompt += "Here is the main paper:\n\n"
    prompt += "=====\n" + paper + "\n=====\n"
    prompt += " Here are the papers:\n\n"
    prompt += context
    prompt += "Just generate the related work section, not the references themselves."


    print("Generating related work section...")
    print("Prompt: ", prompt)
    print('*****************************************')

    response = openai.ChatCompletion.create(
        model=model,
        messages=[
                {"role": "user", "content": prompt},
            ]
    )
    
    res = response["choices"][0]["message"]["content"]
    
    return res, citations
