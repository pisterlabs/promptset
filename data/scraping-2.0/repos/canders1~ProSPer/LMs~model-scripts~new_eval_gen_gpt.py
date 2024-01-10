# coding=utf-8
from transformers import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer, GPT2LMHeadModel, GPT2Tokenizer, BertTokenizer
import torch
import math
import sys
import csv
import logging
import itertools

logging.basicConfig(level=logging.INFO)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if len(sys.argv) < 4:
    print("Usage: python new_eval_gen_gpt.py modelname data datatype",file=sys.stderr)

model_name = "openai-gpt"
model_names = {"gpt":"openai-gpt","gpt2":"gpt2", "gpt2-med":"gpt2-medium","gpt2-large":"gpt2-large","gpt2-xl":"gpt2-xl"}
models = {"gpt":OpenAIGPTLMHeadModel,"gpt2":GPT2LMHeadModel,"gpt2-med":GPT2LMHeadModel,"gpt2-large":GPT2LMHeadModel,"gpt2-xl":GPT2LMHeadModel}
tokenizers = {"gpt":OpenAIGPTTokenizer,"gpt2":GPT2Tokenizer,"gpt2-med":GPT2Tokenizer,"gpt2-large":GPT2Tokenizer,"gpt2-xl":GPT2Tokenizer}
model_name = model_names[sys.argv[1]]
model = models[sys.argv[1]].from_pretrained(model_name)
tokenizer = tokenizers[sys.argv[1]].from_pretrained(model_name)
print("using model: {}".format(model_name), file=sys.stderr)

#split_words = False
#if 'split' in sys.argv:
#    split_words = True
#    print("We don't split words", file=sys.stderr)

use_postfix = False
if 'use_postfix' in sys.argv:
    use_postfix = True
    print("We compute probabilities over the entire sentence", file=sys.stderr)

#model = OpenAIGPTLMHeadModel.from_pretrained(model_name)
#tokenizer = OpenAIGPTTokenizer.from_pretrained(model_name)
#bert_tokenizer=BertTokenizer.from_pretrained('bert-base-uncased')
model.eval()
model.to(device)


def get_probs_for_words(sent,forms):
    pre, target, post = sent.split("***")
    pre = pre.strip()
    if "mask" in target.lower():
        target = ["[MASK]"]
    else:
        target = tokenizer.tokenize(target)
    pre_tokens = tokenizer.tokenize(pre)
    target_idx = len(pre_tokens)

    # Filter answers based on BERT wordpieces to align with BERT results
    """
    try:
        word_ids=bert_tokenizer.convert_tokens_to_ids(forms)
    except KeyError:
        print("skipping",forms[0],"bad wins")
        return None
    """
    wids = []
    for i,w in enumerate(forms):
        t = tokenizer.encode(' '+w)
        wids.append(t)
        assert t == tokenizer.encode(' '+w)
    input_ids = tokenizer.convert_tokens_to_ids(pre_tokens)
    assert input_ids == tokenizer.encode(pre) == tokenizer(pre)['input_ids']
     
    if len(input_ids) == 0:
        print("skipping",pre,forms[0],"empty beginning")
        return None

    # Compute the scores
    add_tok_ws = []
    scores = []
    for w in wids:
        #add_tok_ws.append([])
        scores.append(0)
    #add_tok_w1 = []
    #score_w1 = 0
    for i,ids_w in enumerate(wids):
        #print("Handling form:",file=sys.stderr)
        #print(tokenizer.decode(ids_w),file=sys.stderr)
        if len(ids_w) > 1:
            print("Word is tokenized into multiple entries",file=sys.stderr)
            subwords_to_add = ids_w
            prev_subwords = []
            while len(subwords_to_add) > 1:
                subword = subwords_to_add.pop(0)
                print("Querying subword:",file=sys.stderr)
                print(subword,file=sys.stderr)
                print(tokenizer.decode(subword),file=sys.stderr)
                print("getting intermediate prob",file=sys.stderr)
                print(tokenizer.decode(input_ids+prev_subwords),file=sys.stderr)
                tens = torch.LongTensor([input_ids + prev_subwords]).to(device)
                query_idx = subword
                with torch.no_grad():
                    logits = model(tens)[0]
                    pred_logits = logits[:,-1,:]
                    res = torch.nn.functional.log_softmax(pred_logits,dim=-1)
                    scores[i] += res[0, query_idx].item()
                    print("Intermediate score: "+str(scores[i]),file=sys.stderr)
                    top_pred = torch.topk(pred_logits,1,dim=-1).indices[0].tolist()
                    print("top pred",file=sys.stderr)
                    print(top_pred[0],file=sys.stderr)
                    print(tokenizer.decode(top_pred[0]),file=sys.stderr)
                prev_subwords.append(subword)
            print("at end of adding subwords",file=sys.stderr)
            print(tokenizer.decode(input_ids + prev_subwords),file=sys.stderr)
            tens = torch.LongTensor([input_ids + prev_subwords]).to(device)
            query_idx = subwords_to_add[0]
        else:
            #print("Tokenizer did not split",file=sys.stderr)
            #print(tokenizer.decode(input_ids),file=sys.stderr)
            tens = torch.LongTensor([input_ids]).to(device)
            query_idx = ids_w[0]
            
        with torch.no_grad():
            # To double-check spacing/tokenization, uncomment below:
            # print(tokenizer.decode(input_ids + ids_w),file=sys.stderr)
            
            logits = model(tens)[0]
            pred_logits = logits[:,-1,:]
            #res = res[..., 0:model.config.vocab_size]
            res = torch.nn.functional.log_softmax(pred_logits,dim=-1)
            scores[i] += res[0, query_idx].item()
            #print("res: "+str(scores[i]),file=sys.stderr)
            #print("Final score: "+str(scores[i]),file=sys.stderr)
            top_pred = torch.topk(pred_logits,1,dim=-1).indices[0].tolist()
            #print("top pred",file=sys.stderr)
            #print(top_pred[0],file=sys.stderr)
            #print(tokenizer.decode(top_pred[0]),file=sys.stderr)
            
            # To double-check how logits are retrieved, use the following:
            """
            alt_inp = tokenizer(pre,return_tensors="pt").to(device)
            _, alt_logits, _ = model(**alt_inp,labels=alt_inp["input_ids"])
            alt_res = torch.nn.functional.log_softmax(alt_logits[:,-1,:],dim=-1)
            alt_score = alt_res[0,query_idx].item()
            alt_max = torch.topk(alt_logits[:,-1,:],1,dim=-1).indices[0].tolist()
            assert alt_score == scores[i]
            assert alt_max == top_pred
            """
    assert len(scores) == 5
    return [math.exp(float(s)) for s in scores]


from collections import Counter

def eval_new():
    extended = False
    datatype = sys.argv[3]
    if datatype not in ['corpus','annotated','old']:
        print("Error: unrecognized input data type!",file=sys.stderr)
        return
    for i, line in enumerate(open(sys.argv[2], encoding="utf8")):
        if datatype != 'old':
            if datatype == 'corpus':
                pref,word,postf,subgenre,source,stem,genre,corpus,tense,gof,comef,drivef,walkf,arrivef = line.strip().split("\t")
            else: #annotated
                pref,word,postf,subgenre,source,stem,genre,corpus,tense,embedding,subj,dest,notes,gof,comef,drivef,walkf,arrivef = line.strip().split('\t')
            masked = pref + " ***mask*** " + postf
            forms = [gof,comef,drivef,walkf,arrivef]
            pos = -1
            for i,f in enumerate(forms):
                if f == word:
                    pos = i
            if word not in forms:
                print("Error: form not identified correctly!",file=sys.stderr)
                pos = 0
        else: #Original data formatting
            na, _, masked, good, bad = line.strip().split("\t")
            forms = [good,bad]
            pos = 0
            word = None
        ps = get_probs_for_words(masked,forms)
        if ps is None:
            continue
        pairs = []
        for i,f in enumerate(forms):
            pairs.append(f)
            pairs.append(ps[i])
        if word == comef:
            comego = ps[1] > ps[0]
            win = ps[1] > ps[0] and ps[1] > ps[2] and ps[1] > ps[3] and ps[1] > ps[4]
        elif word == gof:
            comego = ps[0] > ps[1]
            win = ps[0] > ps[1] and ps[0] > ps[2] and ps[0] > ps[3] and ps[0] > ps[4]
        elif word == arrivef:
            comego = ps[4] > ps[1]
            win = ps[4] > ps[0] and ps[4] > ps[1] and ps[4] > ps[2] and ps[4] > ps[3]
        elif word == walkf:
            comego = ps[3] > ps[2]
            win = ps[3] > ps[0] and ps[3] > ps[1] and ps[3] > ps[2] and ps[3] > ps[4]
        elif word == drivef:
            comego = ps[2] > ps[3]
            win = ps[2] > ps[0] and ps[2] > ps[1] and ps[2] > ps[3] and ps[2] > ps[4]
        else:
            comego = None
        
        fields = [str(win),str(comego)]+pairs+[masked.encode("utf8")]
        if datatype != 'old':
            fields += [stem,pref,word,postf,corpus,genre,subgenre,source,tense]
        if datatype == 'annotated':
            fields += [embedding,subj,dest,notes]
        print(*fields,sep=u"\t")
        if i % 100 == 0:
            print(i, file=sys.stderr)
            sys.stdout.flush()

eval_new()
