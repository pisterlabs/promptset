from .llm_generation import *
from langchain.embeddings import OpenAIEmbeddings
import chromadb
import ujson as json
from multiprocessing import Pool
from prompts import *
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff
import tiktoken
import random


@retry(wait=wait_fixed(0.1))
def get_embedding(text, model="text-embedding-ada-002"):
   return [openai.Embedding.create(input=[enc.decode(enc.encode(t)[:8000]) for t in text], model=model, api_key=LLM_API_KEY, organization=ORG)['data'][0]['embedding']]

enc = tiktoken.encoding_for_model("gpt-4")

class MathlibRetriever:
    def __init__(self, logger=None):
        self.logger = logger
        chroma_client = chromadb.PersistentClient(path=f"mathlib_library/skill/vectordb/")
        self.vectordb = chroma_client.get_collection(
            name="skill_library",
            embedding_function=get_embedding
        )
    
    def get_related_declaration_from_mathlib(self, query, n_neighborhood=100):
        neighbor = self.vectordb.query(query_texts=[query], n_results=n_neighborhood) # Get the n_results nearest neighbor embeddings for provided query_texts by similarity.
        return {ids : {
            'source code' : metadatas['source'], 
            'description' : doc, 
            'distances' : distances, 
            'filename' : metadatas['filename'],
            'type' : metadatas['type']
            } for ids, doc, distances, metadatas in zip(neighbor['ids'][0], neighbor['documents'][0], neighbor['distances'][0], neighbor['metadatas'][0])} # Get the name, Lean 3 source code and the natural language description of the results
    
    def premise_selection(self, query : str, n_neighborhood : int=20):
        related_decl_infos = self.get_related_declaration_from_mathlib(query, n_neighborhood)
        related_decl_statements = '- Retrieved declarations from the mathlib library:\n\n' + \
            '\n\n'.join(set(i['filename'] for i in related_decl_infos.values())) + '\n\n' + \
            '\n\n'.join(i['type'] for i in related_decl_infos.values())
        return related_decl_statements
    
    def premise_selection_content(self, query : str, n_neighborhood : int=20, max_token=1000, shuffle=True):
        related_decl_infos = self.get_related_declaration_from_mathlib(query, n_neighborhood)
        choose_list = list(related_decl_infos.keys())
        if shuffle:
            random.shuffle(choose_list)
        result_list = []
        for item in choose_list:
            if len(enc.encode(related_decl_infos[item]['source code']+related_decl_infos[item]['description'])) > max_token:
                continue
            result_list.append(
                f"- Description:\n\n{related_decl_infos[item]['description']}" + \
                f"\n\n- Implementation:\n\n```lean\nimport {related_decl_infos[item]['filename']}\n{related_decl_infos[item]['source code']}\n```"
            )
            if len(enc.encode('\n\n---\n\n'.join(result_list))) > max_token:
                break
        return '\n\n---\n\n'.join(result_list)

    def eval_decls(self, input_info : tuple):
        """
        The Python function `eval_decls(input_info)` takes a tuple of a query and related declaration contents as input. The function returns a list of tuples, each containing a declaration name and its corresponding score.
        :param input_info: a tuple containing a query and related declaration contents
        """
        query, related_decl_contents = input_info
        decl_regex = re.compile(r"#\d+ declaration `([^`]+)`")
        score_regex = re.compile(r"Score: (\d+)")
        eval_result = []
        messages = [
            {"role": "system", 
            "content": premise_selection_prompt},
            {"role": "user", 
            "content": '\n\n---\n\n'.join([f"Mathematical problem:\n{query}"] + related_decl_contents)}
        ]
        response = llm_generate(self.logger, messages)
        for res in response['choices'][0]['message']['content'].split('---'):
            decl_name = decl_regex.search(res).group(1)
            score = score_regex.search(res).group(1)
            eval_result.append((decl_name, int(score)))
        return eval_result


    def premise_selection_with_llm(self, query : str, n_neighborhood : int=100, chunk_size : int=10, max_choice : int=20, problem=''):
        """
        The Python function `get_useful_declaration_from_mathlib(query, n_neighborhood=100, chunk_size=10, max_choice=10)` retrieves related declarations from the mathlib library for a given query. The query may contain a mathematical problem to sovle, or some Lean error to resolve, or other tasks related with mathlib. It processes these declarations in batches and evaluates their usefulness on assisting in resolving problems stated in the query. The function then sorts the evaluated results by score in descending order and returns the top results up to a maximum specified by `max_choice`.
        :param query: a query containing tasks and informations to retrieve related declarations from the mathlib library
        :param n_neighborhood: the number of related declarations to retrieve (default is 100)
        :param chunk_size: the size of each batch for processing declarations (default is 10)
        :param max_choice: the maximum number of top results to return (default is 10)
        """
        if problem:
            query = problem + '\n\n' + query
        related_decl_infos = self.get_related_declaration_from_mathlib(query, n_neighborhood)
        # for k in related_decl_infos:
        #     related_decl_infos[k]['file name'] = decls[k]['filename']
        related_decl_statements = [
            f"#{i} statement of declaration `{decl}`:\n" + 
            # f"/-- {related_decl_infos[decl]['description'].strip()} -/\n" + 
            " ".join([f'{self.decls[decl]["kind"]} {decl}'] + [i['arg'] for i in self.decls[decl]['args']] + [f": {self.decls[decl]['type']}"])
            for i, decl in enumerate(list(related_decl_infos.keys()))
            ]
        retrieve_input = [(
            query,
            related_decl_statements[i * chunk_size : (i + 1) * chunk_size]
            ) for i in range(len(related_decl_statements) // chunk_size)] # Prcess by batches
        with Pool(len(retrieve_input)) as p:
            eval_result = p.map(
                self.eval_decls,
                retrieve_input
            )
        self.logger.info(f"Retrieve result:\n{eval_result}")
        eval_result = sorted([j for i in eval_result for j in i], key=lambda x : x[1], reverse=True)[ : max_choice] # Sort by score
        return ('Retrieved declarations from the mathlib library:\n\n' + 
                '\n\n'.join(f"import {self.decls[decl]['filename']}\n\n" + 
            # (f"/-- {decls[decl]['description'].strip()} -/\n" if decls[decl]['description'] else "") + 
            " ".join([f'{self.decls[decl]["kind"]} {decl}'] + [i['arg'] for i in self.decls[decl]['args']] + [f": {self.decls[decl]['type']}"])
            for decl, i in eval_result if i > 0))
