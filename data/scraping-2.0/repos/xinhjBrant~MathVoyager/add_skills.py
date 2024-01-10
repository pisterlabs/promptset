from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
import ujson as json
from key import *
from tqdm import tqdm, trange
from multiprocessing import Pool, freeze_support, RLock, Manager, Process
import os
import pickle
import chromadb
from pathlib import Path

num_processes = 20

def main(pid, total_len, decl_queue, decls):
    results = []
    embedding_function = OpenAIEmbeddings(openai_api_key=EMBEDDING_API_KEY,
                                          request_timeout=6000)
    with trange(total_len, position=pid, desc=f"#{pid}") as tbar:
        while not decl_queue.empty():
            decl_name = decl_queue.get()
            texts = [decls[decl_name]["informal_statement"] + '\n\n' + decls[decl_name]["informal_proof"]]
            embeddings = embedding_function.embed_documents(texts)
            ids=[decl_name]
            if decl_name.startswith('aime') or decl_name.startswith('imo') or decl_name.startswith('induction'):
                metadatas=[{"type": "hard"}]
            else:
                metadatas=[{"type": "easy"}]
            results.append([metadatas, embeddings, texts, ids])
            tbar.n = total_len - decl_queue.qsize()
            tbar.refresh()
    return results

if __name__ == "__main__":
    with open('extended_prompt_examples/dataset.json', 'r', encoding='utf8') as f:
        decls = json.load(f)
    # with open('skill_library/skill/skills.json', 'r', encoding='utf8') as f:
    #     old_decls = json.load(f)
    
    manager = Manager()
    decl_queue = manager.Queue()
    decl_list = [i.stem for i in Path('extended_prompt_examples').glob('*.lean')]
    total_len = len(decl_list)
    for p in decl_list:
        decl_queue.put(p)
    pool = Pool(processes=num_processes, initargs=(RLock(),), initializer=tqdm.set_lock)
    checking_results = [pool.apply_async(main, args=(pid, total_len, decl_queue, decls)) for pid in range(num_processes)]
    pool.close()
    checking_results = [j for i in checking_results for j in i.get()]
    pool.join()
    # with open('temp.pickle', 'wb') as f:
    #     pickle.dump(checking_results, f, pickle.HIGHEST_PROTOCOL)
    # with open('temp.pickle', 'rb') as f:
    #     checking_results = pickle.load(f)
    dataset = list(map(lambda x : [i[0] for i in x], zip(*checking_results)))
    chroma_client = chromadb.PersistentClient(path=f"extended_prompt_examples/skill/vectordb")
    vectordb = chroma_client.create_collection(name="skill_library")
    vectordb.add(metadatas=dataset[0], embeddings=dataset[1], documents=dataset[2], ids=dataset[3])
    # for metadatas, embeddings, texts, ids in tqdm(checking_results):
    #     try:
    #         vectordb._collection.add(metadatas=metadatas, embeddings=embeddings, documents=texts, ids=ids)
    #     except:
    #         pass
    # chroma_client.persist()
    # for k, v in decls.items():
    #     old_decls[k] = v
    # with open('skill_library/skill/skills.json', 'w', encoding='utf8') as f:
    #     json.dump(old_decls, f, ensure_ascii=False, indent=4)