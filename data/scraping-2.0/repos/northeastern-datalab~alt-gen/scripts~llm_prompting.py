import torch, baukit
from transformers import RobertaTokenizer, RobertaModel
from transformers import AutoTokenizer, AutoModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import os
import pickle
import pickle5 as p
import ast
import numpy as np
from csv import writer
import random
from langchain.llms import OpenAI
from langchain import PromptTemplate, LLMChain
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
import tqdm
import time

new_sample_path = "../labeled_benchmark"
new_sample_datalake = new_sample_path + "/datalake"
new_sample_query_table = new_sample_path + "/query"

gt_union_csv = "../table_union_data/new_labeled_union_gt.csv"

os.environ['OPENAI_API_KEY'] = ''

def loadDictionaryFromPickleFile(dictionaryPath):
    ''' Load the pickle file as a dictionary
    Args:
        dictionaryPath: path to the pickle file
    Return: dictionary from the pickle file
    '''
    filePointer=open(dictionaryPath, 'rb')
    dictionary = p.load(filePointer)
    filePointer.close()
    return dictionary

def saveDictionaryAsPickleFile(dictionary, dictionaryPath):
    ''' Save dictionary as a pickle file
    Args:
        dictionary to be saved
        dictionaryPath: filepath to which the dictionary will be saved
    '''
    filePointer=open(dictionaryPath, 'wb')
    pickle.dump(dictionary,filePointer, protocol=pickle.HIGHEST_PROTOCOL)
    filePointer.close()
    
def average_cosine_similarity(query_texts, texts):
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix_texts = tfidf_vectorizer.fit_transform(texts)
    tfidf_matrix_queries = tfidf_vectorizer.transform(query_texts)
    cosine_similarities = cosine_similarity(tfidf_matrix_queries, tfidf_matrix_texts)
    avg_cosine_similarities = np.mean(cosine_similarities, axis=0)
    return avg_cosine_similarities

def find_closest_texts(query_texts, texts, k=5):
    avg_cosine_similarities = average_cosine_similarity(query_texts, texts)
    closest_indices = np.argsort(avg_cosine_similarities)[0:k]
    print(avg_cosine_similarities[closest_indices])
    closest_texts = [texts[i] for i in closest_indices]
    return closest_texts

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

def decode(model, tok, corpus):
    embeddings = [] 
    for corpus_tmp in baukit.pbar(chunks(corpus, 32)):
        encoding = tok.batch_encode_plus(corpus_tmp, padding=True, truncation=True)
        sentence_batch, attn_mask = encoding["input_ids"], encoding["attention_mask"]
        sentence_batch, attn_mask = torch.LongTensor(sentence_batch).to(device), torch.LongTensor(attn_mask).to(device)

        with torch.no_grad():
            embedding_output_batch = model(sentence_batch, attn_mask)            
            sentence_embeddings = mean_pooling(embedding_output_batch, attn_mask)
        embeddings.append(sentence_embeddings.detach().cpu().numpy())

#         embedding_output_batch = model(sentence_batch, attn_mask)
#         embeddings.append(embedding_output_batch[0][:, 0, :].detach().cpu())
        del sentence_batch, attn_mask, embedding_output_batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return np.concatenate(embeddings, axis=0)

device = 'cuda:0'
def decode_all(queries, texts):
    print("Loading roberta")
    tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
    model = RobertaModel.from_pretrained("roberta-large").to(device)
    print("Loaded roberta")
    query_embeddings = decode(model, tokenizer, queries)
    text_embeddings = decode(model, tokenizer, texts)
    return query_embeddings, text_embeddings

def average_euclidean_distance(query_embeddings, text_embeddings):
    euclidean_distances_arr = euclidean_distances(query_embeddings, text_embeddings)
    avg_euclidean_distances = np.mean(euclidean_distances_arr, axis=0)
    return avg_euclidean_distances

def find_new_closest_texts(avg_euclidean_distances, texts, k):
    closest_indices = np.argsort(avg_euclidean_distances)[0:k]
    closest_texts = [texts[i] for i in closest_indices]
    return closest_texts


gt_union = pd.read_csv(gt_union_csv)
gpt3_path = "../ugen_v1"
gpt3_path_datalake = gpt3_path + "/datalake"
gpt3_path_query_table = gpt3_path + "/query"

gpt_3_union_csv = gpt3_path + "/groundtruth.csv"
gpt_3_union_df = pd.read_csv(gpt_3_union_csv)

new_sample_path = "../labeled_benchmark"
new_sample_datalake = new_sample_path + "/datalake"
new_sample_query_table = new_sample_path + "/query"

gt_union_csv = "../table_union_data/new_labeled_union_gt.csv"

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
#     for i in range(0, len(lst), n):
#         yield lst[i:i + n]
    return [lst[i:i + n] for i in range(0, len(lst), n)]

def get_pair_texts():
    query_texts = []
    texts = []
    
    for index, row in baukit.pbar(gpt_3_union_df.iterrows()):
        curr_text = ""
        q_table = row["query_table"]
        q_table_path = gpt3_path_query_table + "/" + q_table
        q_file = open(q_table_path, "r")
        curr_text += "Table:\n"
        curr_text += q_file.readlines()[1].replace(',', '|')
        q_file.close()

        d_table = row["data_lake_table"]
        d_table_path = gpt3_path_datalake + "/" + d_table
        d_file = open(d_table_path, "r")
        curr_text += "Table:\n"
        d_rows = d_file.readlines()
        if len(d_rows) >= 2:
            curr_text += d_rows[1].replace(',','|')
        else:
            curr_text += d_rows[0].replace(',','|')
        d_file.close()

        curr_text += "Unionable: "
        curr_text += "yes" if str(row["unionable"]) == '1' else "no"
        query_texts.append(curr_text)

    print("Now getting texts")
    for index, row in baukit.pbar(gt_union.iterrows()):
        curr_text = ""
        q_table = row["query_table"]
        q_table_path = new_sample_query_table + "/" + q_table
        q_file = open(q_table_path, "r")
        curr_text += "Table:\n"
        curr_text += q_file.readlines()[1].replace(',', '|')
        q_file.close()

        d_table = row["data_lake_table"]
        d_table_path = new_sample_datalake + "/" + d_table
        d_file = open(d_table_path, "r")
        curr_text += "Table:\n"
        curr_text += d_file.readlines()[1].replace(',','|')
        d_file.close()

        curr_text += "Unionable: "
        curr_text += row["unionable"]
        texts.append(curr_text)
    texts = list(set(texts))
    return query_texts, texts
    
def get_closest_texts(query_texts, texts):
    print("Now finding closest texts")
    query_embeddings, text_embeddings = decode_all(query_texts, texts)
    avg_euclid_dist = average_euclidean_distance(query_embeddings, text_embeddings)
    closest_texts = find_new_closest_texts(avg_euclid_dist,texts,k=5)
    #closest_texts = find_closest_texts(query_texts,texts,k=5)
    return closest_texts


closest_texts_filepath = "../data/ugen_v1_icl_examples.pickle"
if os.path.exists(closest_texts_filepath):
    print("found closest text file")
    closest_texts = list(loadDictionaryFromPickleFile(closest_texts_filepath).values())
else:
    query_texts, texts = get_pair_texts()
    closest_texts = get_closest_texts(query_texts, texts)
    print(closest_texts)
    closest_text_dict = {}
    for i in range(5):
        closest_text_dict[f'icl_ind_{i}'] = closest_texts[i]
    saveDictionaryAsPickleFile(closest_text_dict, "../data/ugen_v1_icl_examples.pickle")
    
#MODEL_NAME = "gpt2-xl"
MODEL_NAME = "lmsys/vicuna-7b-v1.3"
#MODEL_NAME = "gpt-3"
#MODEL_NAME = "circulus/alpaca-7b"
model = None
tok = None
if MODEL_NAME != "gpt-3":
    if "falcon" in MODEL_NAME:
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True).to("cuda")
    else:
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, low_cpu_mem_usage=False).to("cuda")
    if "opt" in MODEL_NAME or "vicuna" in MODEL_NAME or "alpaca" in MODEL_NAME or "falcon" in MODEL_NAME:
        print("came to use fast false")
        tok = AutoTokenizer.from_pretrained(MODEL_NAME,use_fast=False)
    else:
        tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    baukit.set_requires_grad(False, model)
else:
    model = OpenAI(model_name='text-davinci-003', temperature=0, max_tokens=5)
    
    
def generate(model, tok, prefix, n=10):
    #inp = {k: torch.tensor(v)[None].cuda() for k, v in tok(prefix).items()}
    inp = {k: torch.tensor(v)[None].cuda() for k, v in tok(prefix, return_token_type_ids=False).items()}
    initial_length = len(inp['input_ids'][0])
    pkv = None
    prob_stat = []
    for _ in range(n):
        full_out = model(**inp)
        out = full_out['logits']
#         print("out", out)
        #pred = out[0, -1].argmax()
        probs = torch.nn.functional.softmax(out[0, -1], dim=-1)
        favorite_probs, favorite_tokens = probs.topk(k=1, dim=-1)
        prob_stat.append((tok.decode(favorite_tokens),favorite_probs))
        inp['input_ids'] = torch.cat((inp['input_ids'], torch.tensor([favorite_tokens])[None].cuda()), dim=1)
        inp['attention_mask'] = torch.cat((inp['attention_mask'], torch.ones(1, 1).cuda()), dim=1)
#     print(tok.decode(inp['input_ids'][0, initial_length:]))
    return tok.decode(inp['input_ids'][0, initial_length:]), prob_stat


def parse_str(dict_str):
    return ast.literal_eval(dict_str)

def llm_search(benchmark, fromsearch, qt_dir, dl_dir, num_query_tables=50, icl=None, extra_k = 10):
    total_correct = 0
    size = 0
    initial_text = "Are the following tables unionable? Answer in the following format:\nUnionable: {yes/no}\n"
    if icl is not None:
        initial_text += "\n\n".join(icl) + "\n\n"
    test_table_pairs = fromsearch[-num_query_tables:]
    query_tables_seen = 0
    result_dict_list = []
    for index, row in baukit.pbar(test_table_pairs.iterrows()):
        #REMEMBER TO COMMENT THIS OUT!!!!
#         if query_tables_seen >= 5:
#             break
        query_table = row['query_table']
        datalake_tables = row['groundtruth_set']
        search_tables = row['result_set']
        actual_subset_found = []
        confidence_vals = []
        for curr_search in search_tables:
            new_text = ""
            q_table = query_table
            q_table_path = qt_dir + "/" + q_table
            q_file = open(q_table_path, "r")
            new_text += "Table:\n"
            q_rows = q_file.readlines()
            if len(q_rows) >= 2:
                new_text += q_rows[1]
            else:
                new_text += q_rows[0]            
            q_file.close()

            d_table = curr_search
            d_table_path = dl_dir + "/" + d_table + ".csv"
            d_file = open(d_table_path, "r")
            new_text += "Table:\n"
            d_rows = d_file.readlines()
            if len(d_rows) >= 2:
                new_text += d_rows[1]
            else:
                new_text += d_rows[0]
            d_file.close()

            new_text += "Unionable:"
            prompt = initial_text + new_text
#             print('Prompt: ', prompt)
#             print("-----------------------")
            try:
                answer = None
                prob_stats = []
                if MODEL_NAME != "gpt-3":
                    answer, prob_stats = generate(model, tok, prompt, n=4)
                else:
                    answer = model(prompt)
                if answer is None:
                    print("Something is wrong. Answer is None.")
                    return
                expected = "yes"
                not_expected = "no"
                gt_expected = "yes" if d_table in datalake_tables else "no"
                gt_not_expected = "no" if d_table in datalake_tables else "yes"
#                 print("CAME TO ANSWER", answer, "CAME TO EXPECTED", expected)
#                 print("-----------------------")
                size += 1
#                 print(answer)
#                 print("------------")
                if expected in answer.lower() and not_expected not in answer.lower():
                    if gt_expected in answer.lower():
                        total_correct += 1
                    if expected == "yes":
                        actual_subset_found.append(curr_search)
                        added_confidence = False
                        for (curr_token, curr_prob) in prob_stats:
                            if expected == curr_token:
                                added_confidence = True
                                confidence_vals.append(curr_prob.item())
                        if not added_confidence:
                            confidence_vals.append(prob_stats[0][1])
                                
            except Exception as error:
                print("error reached:", error)
                continue
        query_tables_seen += 1
        result_dict_list.append({'query_table': query_table, 
                                 "groundtruth_set": datalake_tables, 
                                 "starmie_result_set": search_tables,
                                 "result_set": actual_subset_found,
                                 "confidence_set": confidence_vals})
    
    actual_size = size - (extra_k*query_tables_seen)
    accuracy = total_correct/actual_size
    return accuracy, total_correct, size, actual_size, result_dict_list

# get_average_context_length_size toooo
#ugen_gt = loadDictionaryFromPickleFile("../data/ugen_v1/santosUnionBenchmark.pickle")
#print(ugen_gt)
# print(accuracy, total_correct, size)

sparse_vals = [0]
#sparse_vals = [0,5,10,15,20]
icl_range = [0,1,2,3]
num_query_tables = 50
for i in range(len(sparse_vals)):
    print("Sparse Val",sparse_vals[i], flush=True)
    result_file = None
    query_file_path = ""
    datalake_file_path = ""
    if i == 0:
        result_file = pd.read_csv(f"../experiment_run_results/ugen_v1_results_k20.csv",
                              converters={'result_set': parse_str, 'groundtruth_set': parse_str})
        num_query_tables = len(pd.unique(result_file["query_table"])) 
        query_file_path = f"../data/ugen_v1/query"
        datalake_file_path = f"../data/ugen_v1/datalake"
    else:
        print("shouldn't have come here")
        result_file = pd.read_csv(f"../experiment_run_results/ugen_v1_sparse_{sparse_vals[i]}_results_k20.csv",
                                  converters={'result_set': parse_str, 'groundtruth_set': parse_str})
        query_file_path = f"../data/ugen_v1_sparse_{sparse_vals[i]}/query"
        datalake_file_path = f"../data/ugen_v1_sparse_{sparse_vals[i]}/datalake"        
    #for icl_num in range(0,len(closest_texts)+1):
    for icl_num_ind in range(len(icl_range)):
        icl_num = icl_range[icl_num_ind]
        icl_val = None
        if icl_num >= 1:
            icl_val = closest_texts[0:icl_num]
        start_time = time.time()
        print("Number of query tables", num_query_tables)
        accuracy, total_correct, size, actual_size, result_dict_list = llm_search('ugen', result_file, query_file_path, datalake_file_path, icl=icl_val, num_query_tables = num_query_tables)
        end_time = time.time()
        saveDictionaryAsPickleFile(result_dict_list, f"../starmie-llm-results/gpt2xl_ugen_v1_tus_sparse_{sparse_vals[i]}_icl-{icl_num}_result.pickle")
        time_taken = end_time - start_time
        print("ICL SIZE", icl_num, flush=True)
        print(accuracy, flush=True)
        print("TIME TAKEN", time_taken, flush=True)




