import os
import openai
from alpaca_farm.auto_annotations import PairwiseAutoAnnotator
import pandas as pd
from transformers import AutoTokenizer
import re
from datasets import load_dataset

from datasets import load_dataset


toker = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
toker.pad_token_id = toker.eos_token_id

# get openai key from file
def oai_kwargs():
    with open("../secret/openaikey.txt", 'r') as f:
        keyval = f.read()
    os.environ['OPENAI_API_KEY'] = keyval
    openai.api_key = keyval
    if 'OPENAI_API_KEY' not in os.environ:
        decoding_kwargs = dict(
            openai_api_key = keyval, #"sk-...",
            openai_organization_ids = None, # ["org-...","org-..."] you can set multiple orgs to avoid rate limits
        )
        assert decoding_kwargs["openai_api_key"] is not None, "OPENAI_API_KEY not found you should set it in environment or above"
    else:
        decoding_kwargs = {}
    return decoding_kwargs

def count_words(indf, k):
    wcnt = [len(toker(s).input_ids) for s in indf[k]]
    return wcnt

# given list, get num tokens for all and return that list
def tok_dist(lis, toker):
    res = []
    for l in lis:
        res.append(len(toker(l).input_ids))
    return res

def reconvert(inp):
    strnew = inp.replace("</s>", "")
    strnew = strnew.replace("<s>", "")
    strnew = strnew.replace("<unk>", "")
    return strnew

def getapfsft(inp, tostack=False):
    instruction_match = re.search(r'### Instruction:\n(.*?)(### Response:|\Z)', inp, re.DOTALL)
    instruction = instruction_match.group(1).strip() if instruction_match else None
    
    # Extract Response
    response_match = re.search(r'### Response:.*?(.*?)(### |\Z)', inp, re.DOTALL)
    response = response_match.group(1).strip() if response_match else None
    if tostack:
        return "Question: " + instruction + "\n\nAnswer: " + response
    return instruction, response

def proctmp(tmp):
    if "<s>" in tmp['response'][0]:
        tmp['response'] = [reconvert(r) for r in tmp['response']]
        tmp['question'] = [reconvert(r) for r in tmp['question']]
    if "### Instruction" in tmp['response'][0]:
        resps = [getapfsft(r)[1] for r in tmp['response']]
        qs = [getapfsft(r)[0] for r in tmp['response']]
        tmp['response'] = resps
        tmp['question'] = qs
    elif "### Instruction" in tmp['question'][0]: 
        tmp['question'] = [getapfsft(r)[0] for r in tmp['question']]
        
    if "Question:" in tmp['response'][0]:
        tmp['response'] = [r[len(q):] for r, q in zip(tmp['response'], tmp['question'])]
    if "Question:" in tmp['question'][0]:
        tmp['question'] = [s[len("Question: "):-1*len("\n\nAnswer: ")] for s in tmp['question']]
        
    if "Continue the conversation:\n\n" in tmp['question'][0]:
        tmp['question'] = [s.replace("Continue the conversation:\n\n", "") for s in tmp['question']]

    return tmp
    
def tulu_to_qa(resps):
    qlist = []
    rlist = []
    for r in resps:
        tmp = r[7:]
        q, a = tmp.split("\n<assistant>\n")
        qlist.append(q)
        rlist.append(a)
    return qlist, rlist

def load_alldfs(base="use_outs/", limit=100, matching=True):
    alldfs = {}
    for f in os.listdir(base):
        print(f)
        if ".jsonl" in f:
            tmp = pd.read_json(base+f, lines=True, orient='records')
            tmp = proctmp(tmp)
            if "<s>" in tmp['response'][0]:
                tmp['response'] = [reconvert(r) for r in tmp['response']]
                tmp['question'] = [reconvert(r) for r in tmp['question']]
            if "### Instruction" in tmp['response'][0]:
                resps = [getapfsft(r)[1] for r in tmp['response']]
                qs = [getapfsft(r)[0] for r in tmp['response']]
                tmp['response'] = resps
                tmp['question'] = qs
            if "<user>" in tmp['response'][0]:
                qs, resps = tulu_to_qa(tmp['response'])
                tmp['question'] = qs
                tmp['response'] = resps
            if "Continue the conversation:\n\n" in tmp['question'][0]:
                print('here')
                tmp['question'] = [s.replace("Continue the conversation:\n\n", "") for s in tmp['question']]
            if "Question:" in tmp['response'][0]:
                tmp['response'] = [r[len(q):] for r, q in zip(tmp['response'], tmp['question'])]
            if "Question:" in tmp['question'][0]:
                tmp['question'] = [s[len("Question: "):-1*len("\n\nAnswer: ")] for s in tmp['question']]
            # constrain to only 200 examples for everything
            #tmp = tmp.loc[:199]
            tmp = tmp.dropna()
            tmp['wcnt'] = count_words(tmp, "question")
            tmp['rcnt'] = count_words(tmp, "response")
            #tmp = tmp[tmp['wcnts']<200].reset_index()
            #tmp = tmp[tmp['rcnts']<200].reset_index()
            alldfs[f.replace("generated_", "").replace(".jsonl", "")] = tmp
    
    valid_indices_sets = []
    if matching:
        for df_key in alldfs:
            if df_key=="davinciwebgpt":
                continue
            df = alldfs[df_key]
            valid_indices_for_df = set(df[(df['wcnt'] < 1000) & (df['rcnt'] < 1000)].index)
            valid_indices_sets.append(valid_indices_for_df)
        
        # Step 2: Find intersection of all sets to get indices valid across all dataframes
        common_valid_indices = set.intersection(*valid_indices_sets)
        
        # Step 3: Filter each dataframe using the common valid indices
        for df_key in alldfs:
            if df_key=="davinciwebgpt":
                continue
            alldfs[df_key] = alldfs[df_key].loc[list(common_valid_indices)].reset_index(drop=True)
            alldfs[df_key] = alldfs[df_key].sort_values(by='question').reset_index(drop=True)
            alldfs[df_key] = alldfs[df_key].loc[:limit]
    #alldfs['davinci']=pd.read_json("../outputs/ckpt_generations/generated_davinci.jsonother", lines=True, orient='records')
    #alldfs['davinci'] = alldfs['davinci'].sort_values(by='question').reset_index(drop=True)
    return alldfs

def apf_format(indf):
    result = []
    for i, row in indf.iterrows():
        result.append({
            'instruction':row['question'],
            'input':"",
            'output':row['response']
        })
    return result

def annotate_apfarm(alldfs, baseline, test, start, end, dec_kwargs):
    # make a fresh annotator each time (is this a good idea?, is it consistent?)
    ann = PairwiseAutoAnnotator(annotators_config="annotator_pool_v0/configs.yaml", **dec_kwargs)
    base_outs = apf_format(alldfs[baseline])
    test_outs = apf_format(alldfs[test])
    assert end>0
    assert end>start
    print("sanity check")
    for i in range(start, end, 100):
        print(i)
    for i in range(start, end, 100):
        print(i, baseline)
        if os.path.exists("../outputs/apeval/"+baseline+"_"+test+"_"+str(i)+".jsonl"): 
            print('already exists')
            continue
        tmp_base_outs = base_outs[i:i+100]
        tmp_test_outs = test_outs[i:i+100]
        assert len(base_outs)==len(test_outs)
        annotated = ann.annotate_head2head(outputs_1=tmp_base_outs, outputs_2=tmp_test_outs)
        dftmp = pd.DataFrame(annotated)
        try:
            print(dftmp.preference.mean())
        except:
            ""
        dftmp.to_json("../outputs/apeval/"+baseline+"_"+test+"_"+str(i)+".jsonl", orient='records', lines=True)
    #pd.DataFrame(annotated).to_json("../outputs/apeval/"+baseline+"_"+test+".jsonl")
    return annotated

def preproc_wgpt(example):
    ex = {}
    ex['question'] = example['question']['full_text']
    if example['score_0']>example['score_1']:
        ex['response_j'] = example['answer_0']
        ex['response_k'] = example['answer_1']
    else:
        ex['response_k'] = example['answer_0']
        ex['response_j'] = example['answer_1']
    return ex

def load_wgpt(topval, bottom=0):
    # take eval set using specific seed so it's not polluted by RM initial stuff
    tdset = load_dataset("openai/webgpt_comparisons", split="train")
    tdset = tdset.map(preproc_wgpt)
    tdset = tdset.shuffle(seed=100)
    # take the last portion as a test set
    dset = tdset.select(range(18000, len(tdset)))
    dset = dset.select(range(bottom, topval))
    results = []
    for d in range(len(dset)):
        # use apfarm prompt format here
        results.append({
            'question':dset['question'][d],
            # assume that better one is the response
            'response':dset['response_j'][d],
            'response_k':dset['response_k'][d],
        })
    return results

def filter_and_sort_df(a, b):
    # Find the unique questions in DataFrame 'a'
    unique_questions_in_a = a['question'].unique()
    
    # Filter DataFrame 'b' to keep only rows where the 'question' value is in 'a'
    filtered_b = b[b['question'].isin(unique_questions_in_a)]
    
    # Sort the filtered DataFrame 'b' by the 'question' column
    sorted_b = filtered_b.sort_values(by='question')
    
    return sorted_b