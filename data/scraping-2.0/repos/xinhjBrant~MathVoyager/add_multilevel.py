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
import networkx as nx

def remove_line_numbers(seq):
    result = []
    for line in seq.split('\n'):
        if m := lean_code_regex.match(line):
            result.append(line[m.span()[1] : ])
        else:
            result.append(line)
    return '\n'.join(result)

num_processes = 30

@retry(wait=wait_random_exponential(min=0.3, max=10))
def get_embedding(text, model="text-embedding-ada-002"):
   return [openai.Embedding.create(input=text, model=model, api_key=EMBEDDING_API_KEY)['data'][0]['embedding']]

def get_embedding_list(pid, dataset, result_queue):
    for count, name, context, proof_state, interpretation, sketch, output_proofstep, desc in tqdm(dataset, position=pid+1, desc=f"#{pid}"):
        text = f"- Description: \n\n{desc}\n\n- A tentative Lean 3 file:\n```lean\n{context}\n```\n\n- Proof states corresponding to the `sorry` keywords:\n```\n{proof_state}\n```\n\n- Interpretation of Proof States:\n\n{interpretation}\n\n- Drafting:\n\n{sketch}\n\n- Implementation:\n\n```lean\n{output_proofstep}\n```"
        text = re.sub(r'\n\n\n+', '\n\n', text)
        embeddings = get_embedding([sketch])
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

lean_code_regex = re.compile(r'#\d+ ')

def remove_line_numbers(seq):
    result = []
    for line in seq.split('\n'):
        if m := lean_code_regex.match(line):
            result.append(line[m.span()[1] : ])
        else:
            result.append(line)
    return '\n'.join(result)

def get_env(env):
    buffer = []
    namespaces = []
    opens = []
    for _, _, t, name in env:
        if t == 'namespace':
            namespaces.append(name)
            buffer.insert(0, name.split()[-1])
        elif t == 'open':
            opens.append(name)
    return namespaces, [f'end {item}' for item in buffer], opens            

def main():
    selected_path = ['data', 'algebra', 'analysis', 'field_theory', 'geometry', 'linear_algebra', 'init.data', 'logic', 'number_theory', 'topology.instances', 'order.filter', 'order.well_founded']
    total_len = 0
    input_dataset = []
    manager = Manager()
    result_queue = manager.Queue()
    with open('description_temp.json') as f:
        desp = json.load(f)
    with open('export_db.json', 'r') as f:
        decls = json.load(f)
    file_graph = nx.read_gexf("import.gexf")
    with open("dependency_graph.pickle", "rb") as f:
        decl_graph = pickle.load(f)
    with open('open_env.json') as f:
        env = json.load(f)
    init_end = json.loads(open('decl_init_end.json').read())
    with jsonlines.open('multilevel_proofstep.jsonl') as reader, trange(0) as tbar:
        for i in reader.iter():
            if not any(decls[i[1]]['filename'].startswith(j) for j in selected_path):
                continue
            codes = [i for i in re.findall(r'```Lean3(.*?)```', i[6], re.DOTALL)]
            if not len(codes) == 2 or re.sub(r'\s', '', codes[0]) in ['bysorry', 'beginsorryend', 'beginsorry,end']:
                continue
            i.append(desp[i[1]])
            precedent_files = []
            try:
                for pf in file_graph.successors('mathlib:'+decls[i[1]]['filename']):
                    if pf.startswith('mathlib:'):
                        precedent_files.append(pf[len('mathlib:') : ])
            except:
                pass
            premises = []
            for premise in [i for i in decl_graph.successors(i[1]) if decls[i]['kind'] == 'theorem']:
                if premise in init_end:
                    namespaces, ends, opens = get_env(env[decls[premise]['filename']][premise])
                    premises.append('\n\n'.join(namespaces + opens + [init_end[premise][0]] + ends))
            namespaces, ends, opens = get_env(env[decls[i[1]]['filename']][i[1]])
            i[6] = i[6][ : i[6].find('to get:')].replace('Lean3', 'lean').strip() + \
                (('\nand import files\n```lean\n' + '\n'.join(f'import {i}' for i in precedent_files) + '\n```') if precedent_files else '') + \
                (('\nand add helper lemmas\n```lean\n' + '\n\n'.join(premises) + '\n```') if premises else '') + \
                (('\nand open namespaces\n' + ' '.join([f"`{i.split()[1]}`" for i in namespaces] + [' '.join(f"`{j}`" for j in i.split()[1 : ]) for i in opens])) if namespaces or opens else '') + \
                '\nto get\n```lean\n' + '\n\n'.join([
                '\n'.join(f'import {i}' for i in precedent_files), 
                '\n\n'.join(premises)] + namespaces + opens + [remove_line_numbers(codes[1]).strip()] + ends
                ) + '\n```'
            input_dataset.append(i)
            total_len += 1
            tbar.update()
    # input_dataset = input_dataset[ : 12000]
    # random.shuffle(input_dataset)
    # input_dataset = sorted(input_dataset, key=lambda x : sum(len(i) for i in x[1:]))[ : 10000]
    # total_len = 10000
    # per_size = total_len // num_processes + 1
    # p = Process(target=add_item, args=(result_queue, total_len))
    # p.start()
    # pool = Pool(processes=num_processes, initargs=(RLock(),), initializer=tqdm.set_lock)
    # checking_results = [pool.apply_async(get_embedding_list, args=(pid, input_dataset[per_size * pid : per_size * (pid + 1)], result_queue)) for pid in range(num_processes)]
    # pool.close()
    # checking_results = [i.get() for i in checking_results]
    # pool.join()
    # result_queue.put(None)
    # p.join()
    # with open('small_temp.pickle', 'rb') as f:
    #     checking_results = pickle.load(f)
    # for item in tqdm(checking_results):
    #     code = re.search(r'```lean(.*)```', item[2][0][item[2][0].find('- Implementation:') + len('- Implementation:') : ], re.DOTALL).group(1).replace('Lean3', 'lean').strip()
    #     item[2][0] = item[2][0][item[2][0].find('- A tentative Lean 3 file:') : item[2][0].find('- Interpretation of Proof States:')] + '- Implementation:\n\n' + remove_line_numbers(code)
    # dataset = list(map(lambda x : [i[0] for i in x], zip(*checking_results)))
    # U.f_remove('./multi_level_library')
    # chroma_client = chromadb.PersistentClient(path=f"multi_level_library/skill/vectordb")
    # vectordb = chroma_client.create_collection(name="skill_library")
    # vectordb.add(metadatas=dataset[0], embeddings=dataset[1], documents=dataset[2], ids=dataset[3])

if __name__ == "__main__":
    main()