import os, sys, time, ljqpy, math, re, json
import unicodedata
import torch
import torch.nn as nn
import torch.functional as F
from tqdm import tqdm
import numpy as np
import h5py
from functools import partial
from collections import defaultdict
import argparse
from config import config
import random
import copy
from transformers import LlamaForCausalLM, LlamaTokenizer
from peft import PeftModel


import openai
from diskcache import Cache

openai.api_key = 'your_api_key'
cache = Cache("cache")

parser = argparse.ArgumentParser()
parser.add_argument("--dname", help='dataset for training and testing', choices=['ske2019','HacRED','NYT10-HRL','NYT11-HRL','NYT21-HRL', 'WebNLG', 'WikiKBP', 'NYT10', 'WebNLG_star','CoNLL04'], default='HacRED')
parser.add_argument("--do_train", help='training model for EE model', type=bool, default=False)
parser.add_argument("--do_eval", help='evaluate the output file', type=bool, default=False)
parser.add_argument("--cons_candidate", help='construct the candidate pairs for whole test set',type=bool, default=False)
parser.add_argument("--filter", help='whether use triple filter module',type=bool, default=False)
parser.add_argument("--do_predict", help='to predict a single sentence',type=bool, default=False)
parser.add_argument("--negative_rate", help='how many ground_truth pair become unlabeled pair.', type=float, default=0)
args = parser.parse_args()
#print(args)
negative_threshold = args.negative_rate
dname = args.dname
datadir = './dataset/' + dname
dsplits = 'train test valid'.split()
fns = {x:os.path.join(datadir, f'new_{x}.json') for x in dsplits}

maxlen = config[dname]['maxlen']
if not os.path.isdir(dname): os.makedirs(dname)
def wdir(x): return os.path.join(dname, x)
rc_threshold = config[dname]['thre_rc']
ee_threshold = config[dname]['thre_ee']

from transformers import BertTokenizer, BertModel, set_seed
set_seed(52)

if dname in ['HacRED', 'ske2019']: plm_name = 'hfl/chinese-roberta-wwm-ext'
else: plm_name = 'bert-base-cased'
tokenizer = BertTokenizer.from_pretrained(plm_name, model_max_length=maxlen)
with open(os.path.join(datadir, 'rel2id.json')) as fin:
    rel_map = json.load(fin)
rev_rel_map = {v:k for k,v in rel_map.items()}
relation_list = list(rel_map.keys())
rels = None

from utils import TN, restore_token_list, GetTopSpans, FindValuePos

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, negative_threshold):
        self.y = 0
        #global rel
        global rel_map
        print('rels:', len(rel_map))
        #print(rels.t2id)
        self.items = []
        for z in data:
            item = {}
            item['tid'] = torch.tensor(tokenizer.encode(z['sentText'])[:512])
            item['yrc'] = list(set(rel_map[x['label']] for x in z['relationMentions']))
            self.items.append(item)
            self.y += len(item['yrc'])
    def __len__(self): return len(self.items)
    def __getitem__(self, k): 
        item = self.items[k]
        return item['tid'], item['yrc']

class PU_mid_loss(nn.Module):
    def __init__(self, mid=0, pi=0.1):
        super().__init__()
        self.mid = mid
        self.pi = pi

    def forward(self,y_true,y_pred):
        eps = torch.tensor(1e-6).cuda()
        y_true = y_true.double()
        pos = torch.sum(y_true * y_pred, 1) / torch.maximum(eps, torch.sum(y_true, 1))
        pos = - torch.log(pos + eps)
        neg = torch.sum((1-y_true) * y_pred, 1) / torch.maximum(eps, torch.sum(1-y_true, 1))
        neg = torch.abs(neg - self.mid) 
        neg = - torch.log(1 - neg + eps)
        return torch.mean(self.pi*pos + neg)

class DatasetEE(torch.utils.data.Dataset):
    def __init__(self, data, negative_threshold):
        self.items = []
        for i, z in enumerate(data):
            text, spo_list = z['sentText'], z['relationMentions']
            labels = z.get('rc_pred', list(set(x['label'] for x in spo_list)))  
            tokens = tokenizer.tokenize(text)[:maxlen]
            otokens = restore_token_list(text, tokens)
            tid = [tokenizer.cls_token_id] + tokenizer.convert_tokens_to_ids(tokens) + [tokenizer.sep_token_id]
            for label in labels:
                if random.random() <= negative_threshold:
                    continue
                prompt = tokenizer.encode(label)
                plen = len(prompt)
                item = {'text':text, 'spo_list':spo_list}
                item['id'] = i
                item['plen'] = plen
                item['otokens'] = otokens
                item['label'] = label
                item['tid'] = torch.tensor(prompt + tid[1:])
                slen = item['tid'].size(0)
                ss = set(TN(x['em1Text']) for x in spo_list if x['label'] == label)
                oo = set(TN(x['em2Text']) for x in spo_list if x['label'] == label)
                yy = torch.zeros((slen, 4)).float()
                for s in ss:
                    for u, v in FindValuePos(otokens, s): 
                        yy[u+plen,0] = yy[v-1+plen,1] = 1
                for o in oo:
                    for u, v in FindValuePos(otokens, o): 
                        yy[u+plen,2] = yy[v-1+plen,3] = 1
                item['yy'] = yy
                self.items.append(item)
    def __len__(self): 
        return len(self.items)
    def __getitem__(self, k): 
        item = self.items[k%len(self.items)]
        return item['tid'], item['yy']


class DatasetonlyEE(torch.utils.data.Dataset):
    def __init__(self, data, negative_threshold):
        self.items = []
        for i, z in enumerate(data):
            text, spo_list = z['sentText'], z['relationMentions']
            tokens = tokenizer.tokenize(text)[:maxlen]
            otokens = restore_token_list(text, tokens)
            tid = [tokenizer.cls_token_id] + tokenizer.convert_tokens_to_ids(tokens) + [tokenizer.sep_token_id]
            if random.random() <= negative_threshold:
                continue
            item = {'text':text, 'spo_list':spo_list}
            item['id'] = i
            item['otokens'] = otokens
            item['tid'] = torch.tensor(tid[1:])
            slen = item['tid'].size(0)
            ss = set(TN(x['em1Text']) for x in spo_list)
            oo = set(TN(x['em2Text']) for x in spo_list)
            yy = torch.zeros((slen, 4)).float()
            for s in ss:
                for u, v in FindValuePos(otokens, s): 
                    yy[u,0] = yy[v-1,1] = 1
            for o in oo:
                for u, v in FindValuePos(otokens, o): 
                    yy[u,2] = yy[v-1,3] = 1
            item['yy'] = yy
            self.items.append(item)
    def __len__(self): 
        return len(self.items)
    def __getitem__(self, k): 
        item = self.items[k%len(self.items)]
        return item['tid'], item['yy']


def rc_collate_fn(items):
    xx = nn.utils.rnn.pad_sequence([x for x,y in items], batch_first=True)
    yy = torch.zeros((len(items), len(rel_map)))
    for i, (x, ys) in enumerate(items):
        for y in ys: yy[i,y] = 1
    return xx, yy

def ee_collate_fn(items):
    xx = nn.utils.rnn.pad_sequence([x for x,y in items], batch_first=True)
    yy = nn.utils.rnn.pad_sequence([y for x,y in items], batch_first=True)
    return xx, yy.float()


class EEModel(nn.Module):
    def __init__(self, outd=4):
        super().__init__()
        self.bert = BertModel.from_pretrained(plm_name)
        self.fc = nn.Linear(768, outd)
    def forward(self, x):
        z = self.bert(x).last_hidden_state
        out = self.fc(z)
        out = torch.sigmoid(out)
        return out

class MetricF1:
    def __init__(self):
        self.correct = self.output = self.golden = 0
    def append(self, out, ans):
        out, ans = set(out), set(ans)
        mid = out & ans
        self.correct += len(mid)
        self.output += len(out)
        self.golden += len(ans)

    def compute(self, show=True):
        correct, output, golden = self.correct, self.output, self.golden
        prec = correct / max(output, 1);  reca = correct / max(golden, 1);
        f1 = 2 * prec * reca / max(1e-9, prec + reca)
        pstr = 'Prec: %.4f %d/%d, Reca: %.4f %d/%d, F1: %.4f' % (prec, correct, output, reca, correct, golden, f1)
        if show: print(pstr)
        return f1

    def compute_and_record(self, fout):
        correct, output, golden = self.correct, self.output, self.golden
        prec = correct / max(output, 1);  reca = correct / max(golden, 1);
        f1 = 2 * prec * reca / max(1e-9, prec + reca)
        pstr = 'Prec: %.4f %d/%d, Reca: %.4f %d/%d, F1: %.4f' % (prec, correct, output, reca, correct, golden, f1)
        fout.write(pstr+'\n')
        return (prec, reca, f1)

def tt(t):
    return t['em1Text']+' | '+t['em2Text']

sys.path.append('../')
import pt_utils

def ComputeOne(item, pairs, f1, fout, label=None):
    # spos = [x for x in item['spo_list'] if x['label'] == item['label']]
    spos = [{'em1Text': x['em1Text'], 'em2Text':x['em2Text']} for x in item['spo_list']]
    pairs = set(tt(x) for x in pairs)
    spos = set(tt(x) for x in spos)
    print('-'*30, file=fout)
    print(item['text'], file=fout)
    for x in pairs&spos: print('o', x, file=fout)
    for x in pairs-spos: print('-', x, file=fout)
    for x in spos-pairs: print('+', x, file=fout)
    f1.append(pairs, spos)

def test_ee(): 
    outs = [] 
    with torch.no_grad():
        for x, y in dl_dev:
            out = ee(x.cuda()).detach().cpu()
            for z in out: outs.append(z.numpy())
    f1 = MetricF1()
    fout = open('ret.txt', 'w', encoding='utf-8')
    for item, rr in zip(dss['test'].items, outs):
        pairs = decode_entitypair(item, rr, ee_threshold)
        ComputeOne(item, pairs, f1, fout)
    f1.compute()
    fout.close()


def decode_entitypair(item, rr, ee_threshold, gpout=None):
    otokens = item['otokens']
    subs = GetTopSpans(otokens, rr[:,:2])
    objs = GetTopSpans(otokens, rr[:,2:])
    vv1 = [x for x,y in subs if y >= 0.1]
    vv2 = [x for x,y in objs if y >= 0.1]
    subv = {x:y for x,y in subs}
    objv = {x:y for x,y in objs}
    pairs = []
    for sv1, sv2 in [(sv1, sv2) for sv1 in vv1 for sv2 in vv2]:
        if gpout is not None:
            loc1, loc2 = FindValuePos(otokens, sv1), FindValuePos(otokens, sv2)
            vals = []
            for u1, v1 in loc1:
                for u2, v2 in loc2:
                    vals.append([])
                    for i in range(1+u1, 1+v1):
                        for j in range(1+u2, 1+v2):
                            vals[-1].append(gpout[0,i,j])
            ind = item['id']
            tdata[ind].setdefault('gp_detail', []).append( (sv1, sv2, vals) )
            vals = [np.array(x).mean() for x in vals]
            tdata[ind].setdefault('gp', []).append( (sv1, sv2, vals) )
            if len(vals) == 0: continue
            if max(vals) < 0:
                continue
        score = min(subv[sv1], objv[sv2])
        if score < ee_threshold: continue
        pairs.append({'em1Text': sv1, 'em2Text':sv2})
    return pairs

def tt2(t):
    try:
        ans = t['em1Text']+' | '+t['em2Text']+' | '+t['label']
    except:
        ans = 'wrongcase'
    return ans

def tt2_reverse(t):
    try:
        ans = t['em2Text']+' | '+t['em1Text']+' | '+t['label'] 
    except:
        ans = 'wrongcase'
    return ans

def ComputeOne_2(item, preds, f1, fout, label=None):
    spos = item['std_ans']
    triples = preds['relationMentions']
    triples_1 = [tt2(x) for x in triples]
    triples_reverse = [tt2_reverse(x) for x in triples]
    spos = set(tt2(x) for x in spos)
    triples2 = []
    for i in range(len(triples_1)):
        if triples_1[i] in spos:
            triples2.append(triples_1[i])
        elif triples_reverse[i] in spos:
            triples2.append(triples_reverse[i])
        else:
            triples2.append(triples_1[i])
    triples2 = set(triples2)
    print('-'*30, file=fout)
    print(item['sentText'], file=fout)
    for x in triples2&spos: print('o', x, file=fout)
    for x in triples2-spos: print('-', x, file=fout)
    for x in spos-triples2: print('+', x, file=fout)
    f1.append(triples2, spos)

def test_LLM(filterprompt = False, spo_limit = 1):
    with open(wdir('your_candidate_path'),'r',encoding='utf-8') as fin:
        outs = json.load(fin)
    if filterprompt:
        path2 = wdir('your_file_path')
    else:
        path2 = wdir('your_file_path')
    with open(path2,'r',encoding='utf-8') as fin:
        preds = json.load(fin)
    f1 = MetricF1()
    fout = open('ret.txt', 'w', encoding='utf-8')
    for i in range(min(len(outs),len(preds))):
        if len(outs[i]['std_ans']) >= spo_limit:
            ComputeOne_2(outs[i], preds[i], f1, fout)
    f1.compute()
    fout.close()

# Train ee
if args.do_train:
    epochs = 25
    dss = {x:DatasetonlyEE(ljqpy.LoadJsons(fn), negative_threshold if x == 'train' else 0) for x, fn in fns.items()}
    dl_train = torch.utils.data.DataLoader(dss['train'], batch_size=16, shuffle=True, collate_fn=ee_collate_fn)
    dl_dev = torch.utils.data.DataLoader(dss['test'], batch_size=16, shuffle=False, collate_fn=ee_collate_fn)
    total_steps = len(dl_train) * epochs

    ee = EEModel().cuda()
    eemfile = wdir(f'ee_negative_{negative_threshold}.pt')
    #ee.load_state_dict(torch.load(eemfile))
    pt_utils.lock_transformer_layers(ee.bert, 3)
    optimizer, scheduler = pt_utils.get_bert_optim_and_sche(ee, 5e-5, total_steps)

    loss_fct = lambda y_pred, y_true: - (y_true*torch.log(y_pred+1e-9) + (1-y_true)*torch.log(1-y_pred+1e-9)).mean()

    def train_func(model, ditem):
        x, y = ditem
        y = y.cuda()
        out = model(x.cuda())
        loss = loss_fct(out, y)# + 0.1 * out.mean()
        oc = (out > 0.5).float()
        prec = (oc + y > 1.5).sum() / max(oc.sum().item(), 1)
        reca = (oc + y > 1.5).sum() / max(y.sum().item(), 1)
        f1 = 2 * prec * reca / (prec + reca)
        r = {'loss': loss, 'prec': prec, 'reca': reca, 'f1':f1}
        return r

    pt_utils.train_model(ee, optimizer, dl_train, epochs, train_func, test_ee, 
                   scheduler=scheduler, save_file=eemfile)

if args.cons_candidate:
    tdata = ljqpy.LoadJsons(fns['test'])
    ee = EEModel().cuda()
    ee.load_state_dict(torch.load(wdir(f'ee_negative_{negative_threshold}.pt')),strict=False)
    if args.filter:
        import relcomb
        gp = relcomb.GlobalPointerModel(plm_name).cuda()
        gp.load_state_dict(torch.load(wdir(f'relcomb_BCE_sh_{negative_threshold}.pt')),strict=False)
    ds_ee = DatasetonlyEE(tdata, 0)
    dl_ee = torch.utils.data.DataLoader(ds_ee, batch_size=16, shuffle=False, collate_fn=ee_collate_fn)
    gpouts = []
    if args.filter:
        for x, y in dl_ee:
            out = gp(x.cuda()).detach().cpu()
            for z in out: gpouts.append(z.numpy())
        print('gp enabled')
    else:
        gpouts = [None] * len(tdata)
    outs = [] 
    with torch.no_grad():
        for x, y in dl_ee:
            out = ee(x.cuda()).detach().cpu()
            for z in out: outs.append(z.numpy())
    for item, rr in zip(ds_ee.items, outs):
        triples = decode_entitypair(item, rr, ee_threshold, gpouts[item['id']])
        tdata[item['id']].setdefault('preds', []).extend(triples)
    with open(wdir('candidate.json'), 'w', encoding='utf-8') as fout:
        wdata = [{'sentText':x['sentText'],'preds':x['preds'],'std_ans':x['relationMentions']} for x in tdata] 
        json.dump(wdata, fout, ensure_ascii=False, indent=2)

if args.do_predict:
    ee = EEModel().cuda()
    ee.load_state_dict(torch.load(wdir(f'ee_negative_{negative_threshold}.pt')))
    if args.filter:
        import relcomb
        gp = relcomb.GlobalPointerModel(plm_name).cuda()
        gp.load_state_dict(torch.load(wdir(f'relcomb_BCE_sh_{negative_threshold}.pt')))

    while True:
        sent = input('>')
        tdata = [{'sentText': sent, 'relationMentions':[]}]
        
        ds_ee = DatasetonlyEE(tdata, 0)
        dl_ee = torch.utils.data.DataLoader(ds_ee, batch_size=1, shuffle=False, collate_fn=ee_collate_fn)
        gpouts = []
        if args.filter:
            for x, y in dl_ee:
                out = gp(x.cuda()).detach().cpu()
                for z in out: gpouts.append(z.numpy())
            print('gp enabled')
        else:
            gpouts = [None] * len(tdata)
        
        outs = [] 
        with torch.no_grad():
            for x, y in dl_ee:
                out = ee(x.cuda()).detach().cpu()
                for z in out: outs.append(z.numpy())
        for item, rr in zip(ds_ee.items, outs):
            triples = decode_entitypair(item, rr, ee_threshold, gpouts[item['id']])
            tdata[item['id']].setdefault('preds', []).extend(triples)
        for x in tdata[0].get('preds', []):
            print(x)

if args.do_eval:
    for limit in [1,2,4,5,7]:
        print(f'limit={limit}')
        print('without filter prompt \n')
        test_LLM(spo_limit=limit)
        print('with filter prompt\n')
        test_LLM(True, limit)