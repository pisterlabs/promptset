import ujson as json
import jsonlines
from key import *
from tqdm import tqdm, trange
from multiprocessing import Pool, freeze_support, RLock, Manager, Process
import os
import pickle
import chromadb
from pathlib import Path
import re
import openai
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff
import utils as U
import random

# U.f_remove('./multi_level_library')

lean_code_regex = re.compile(r'\n#\d+ (.*?)(?=\n)', re.DOTALL)

theorem_name = re.compile(r'(?:theorem|lemma)\s+(.*?)\s+')

num_processes = 30

@retry(wait=wait_random_exponential(min=0.3, max=10))
def get_embedding(text, model="text-embedding-ada-002"):
   return [openai.Embedding.create(input=text, model=model, api_key=EMBEDDING_API_KEY)['data'][0]['embedding']]

def get_embedding_list(pid, dataset, result_queue):
    for count, item in tqdm(dataset, position=pid+1, desc=f"#{pid}"):
        error_message = '\n\n'.join(item['error_message'])
        text = f'''- A tentative Lean 3 file:

{item['invalid_code']}

- Error messages:

{error_message}

{item['revise']}'''
        if not (m := theorem_name.findall(item['revised_code'])):
            continue
        else:
            name = m[-1]
        text = re.sub(r'\n\n\n+', '\n\n', text)
        embeddings = get_embedding([error_message])
        ids=[str(count)]
        metadatas=[{"name": name}]
        result_queue.put([metadatas, embeddings, [text], ids])

def add_item(result_queue, total):
    # chroma_client = chromadb.PersistentClient(path=f"multi_level_library/skill/vectordb")
    # vectordb = chroma_client.create_collection(name="skill_library")
    dataset = []
    with trange(total, position=0) as tbar:
        while True:
            item = result_queue.get()
            if item is None:
                with open('small_temp.pickle', 'wb') as f:
                    pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
                break
            dataset.append(item)
            # metadatas, embeddings, texts, ids = item
            # vectordb.add(metadatas=metadatas, embeddings=embeddings, documents=texts, ids=ids)
            tbar.update()

def main():
    manager = Manager()
    result_queue = manager.Queue()
    with open('successful_revision.json') as f:
        input_dataset = json.load(f)
    input_dataset = [[i, item] for i, item in enumerate(input_dataset)]
    total_len = len(input_dataset)
    per_size = total_len // num_processes + 1
    p = Process(target=add_item, args=(result_queue, total_len))
    p.start()
    pool = Pool(processes=num_processes, initargs=(RLock(),), initializer=tqdm.set_lock)
    checking_results = [pool.apply_async(get_embedding_list, args=(pid, input_dataset[per_size * pid : per_size * (pid + 1)], result_queue)) for pid in range(num_processes)]
    pool.close()
    checking_results = [i.get() for i in checking_results]
    pool.join()
    result_queue.put(None)
    p.join()
    with open('small_temp.pickle', 'rb') as f:
        checking_results = pickle.load(f)
    dataset = list(map(lambda x : [i[0] for i in x], zip(*checking_results)))
    chroma_client = chromadb.PersistentClient(path=f"revision_library/skill/vectordb")
    vectordb = chroma_client.create_collection(name="skill_library")
    vectordb.add(metadatas=dataset[0], embeddings=dataset[1], documents=dataset[2], ids=dataset[3])

if __name__ == "__main__":
    main()