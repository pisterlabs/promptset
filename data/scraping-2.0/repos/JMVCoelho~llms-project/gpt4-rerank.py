import openai  # for calling the OpenAI API
import pandas as pd  # for storing text and embeddings data
from tqdm import tqdm
import os

base = './data'
GPT_MODEL = "gpt-4"
openai.api_key = ""

top_100 = {}
n=100
run_to_rerank = "./runs/distilbert/dev.run"
docs_needed = []
with open(run_to_rerank, 'r') as h:
    for line in h:
        qid, _, did, _, _, _ = line.split()
        qid = str(qid)
        did = str(did)
        if qid not in top_100:
            top_100[qid] = [did]
            docs_needed.append(did)
        else:
            if len(top_100[qid]) < n:
                top_100[qid].append(did)
                docs_needed.append(did)

#os.makedirs(f"./runs/{GPT_MODEL}rerank-distilbert/")

jsonObjdata = pd.read_json(f"{base}/corpus.jsonl", lines=True)
ans = []
count=0
corpus = dict()
for index, row in jsonObjdata.iterrows():
    if str(row["doc_id"]) in docs_needed:
        corpus[str(row["doc_id"])] = row["page_title"]

del jsonObjdata

jsonObj = pd.read_json(path_or_buf=f"{base}/dev/queries.jsonl", lines=True)
ans = []
actual = []
count=0




done = 0
todo =  150

os.makedirs(f"./runs/{GPT_MODEL}-rerank-distilbert/")

with open(f"./runs/{GPT_MODEL}-rerank-distilbert/dev.run", 'w') as out_file:
    for index,row in tqdm(jsonObj.iterrows(), total=todo):
        if done == todo:
            break 
        query_id = str(row['id'])

        query_top_100 = top_100[query_id][:100]

        small_id_mapper = {str(idx): str(x) for idx, x in enumerate(query_top_100)}
        inv_mapper = {v:k for k,v in small_id_mapper.items()}

        query = f"I will provide you with {n} movies, each indicated by a numerical identifier []. Rank the moves based on their relevance to the user description: {row['text']}.\n\n\n"
        
        for doc in query_top_100:
            doc_title = corpus[str(doc)]
            small_id = inv_mapper[str(doc)]
            query += f"[{small_id}] {doc_title}\n"

        query += f"\nUser Description:\n{row['text']}\n"
        query += f"Rank the {n} movies above based on their relevance to the search query. Use your best knowledge about the movie given only their titles. All the movies should be included and listed using identifiers, in descending order of relevance. The output format should be [] > [], e.g., [4] > [2]. Only respond with the ranking results, do not say any word or explain."

        response = openai.ChatCompletion.create(
            messages=[
                {'role': 'system', 'content': 'You help someone re-rank a list of movies with respect to relevancy towards a description.'},
                {'role': 'user', 'content': query},
            ],
            model=GPT_MODEL,
            temperature=0,
        )
        ranked_list = response['choices'][0]['message']['content']

        try:
            int_list = [int(part.strip("[] ")) for part in ranked_list.split(">") if part.strip("[] ").isdigit()]
            if any(number > 99 or number < 0 for number in int_list):
                print(f"bad list for query {query_id}")
                int_list = list(small_id_mapper.keys())
            if len(int_list) > 100:
                print(f"bad list (too big) for query  {query_id}")
                int_list = list(small_id_mapper.keys())

        except Exception as e:
            print(f"bad list for query {query_id}")
            int_list = list(small_id_mapper.keys())
        
        doc_id_list = [small_id_mapper[str(small_id)] for small_id in int_list]


        for pos, rd in enumerate(doc_id_list):
            out_file.write(f"{query_id} Q0 {rd} {pos} {1/(pos+1)} gpt4-rerank-distilbert-top100\n")
        
        done += 1