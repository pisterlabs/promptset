import argparse
import os
import random
import json
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle

from analysis import rocstories as rocstories_analysis
from datasets import rocstories
from model_pytorch_analysis import DoubleHeadModel, load_openai_pretrained_model
#from ..pytorch_pretrained_bert.modeling_openai import OpenAIGPTDoubleLMHeadModel, OpenAIGPTConfig

from opt import OpenAIAdam
from text_utils import TextEncoder
from utils import (encode_dataset, iter_data,
                   ResultLogger, make_path, encode_dataset_whole)
from loss import ClassificationLossCompute
def flatten_list(l):
    flat_list = []
    for sublist in l:
        for item in sublist:
            flat_list.append(item)
    return flat_list
def transform_recipe_whole(X1, X2, X3):
    n_batch = len(X1)
    xmb = np.zeros((n_batch, n_ctx, 2), dtype=np.int32)
    mmb = np.zeros((n_batch, n_ctx), dtype=np.float32)
    start = encoder['_start_']
    delimiter = encoder['_delimiter_']
    extra1 = encoder['_extra1_']
    for i, (x1, x2, x3), in enumerate(zip(X1, X2, X3)):
        #x12 = [start] + x1[:max_len] + [delimiter] + x2[:max_len] + [delimiter] + x3[:max_len]+ [clf_token]
        x12 = [start] + x2 +[delimiter] 
        '''
        for x in x3:
            x12+= x + [extra1]
        '''
        for x in x1:
            x12+= x +[clf_token]
        
        l12 = len(x12)
        if l12 == 0:
            print('O length train para\n')
            continue
        if l12 > 512:
            continue
        '''
        if l12 < 5:
            continue
        '''
        #l13 = len(x13)
        xmb[i, :l12, 0] = x12
        #xmb[i, 1, :l13, 0] = x13
        mmb[i, :l12] = 1
        #mmb[i, 1, :l13] = 1
    # Position information that is added to the input embeddings in the TransformerModel
    xmb[:, :, 1] = np.arange(n_vocab + n_special, n_vocab + n_special + n_ctx)
    return xmb, mmb

def transform_recipe_whole_just_recipe(X1, X2, X3):
    n_batch = len(X1)
    xmb = np.zeros((n_batch, n_ctx, 2), dtype=np.int32)
    mmb = np.zeros((n_batch, n_ctx), dtype=np.float32)
    start = encoder['_start_']
    delimiter = encoder['_delimiter_']
    extra1 = encoder['_extra1_']
    for i, (x1, x2, x3), in enumerate(zip(X1, X2, X3)):
        #x12 = [start] + x1[:max_len] + [delimiter] + x2[:max_len] + [delimiter] + x3[:max_len]+ [clf_token]
        x12 = x1
        '''
        for x in x3:
            x12+= x + [extra1]
        '''

        #l12 = len(x12)
        x12+=[clf_token]

        l12 = len(x12)
        if l12 <= 2:
            print('fucked\n')
            quit()
        if l12 == 0:
            print('O length train para\n')
            quit()
        if l12 > 512:
            print('512+ length paragraph\n')
            xmb[i, :n_ctx, 0] = x12[:n_ctx]
            mmb[i, :n_ctx] = 1
            continue
        '''
        if l12 < 7:
            continue
        '''
        #l13 = len(x13)
        xmb[i, :l12, 0] = x12
        #xmb[i, 1, :l13, 0] = x13
        mmb[i, :l12-1] = 1
        #mmb[i, 1, :l13] = 1
    # Position information that is added to the input embeddings in the TransformerModel
    xmb[:, :, 1] = np.arange(n_vocab + n_special, n_vocab + n_special + n_ctx)
    return xmb, mmb

def transform_recipe(X1, X2, X3):
    n_batch = len(X1)
    xmb = np.zeros((n_batch, n_ctx, 2), dtype=np.int32)
    mmb = np.zeros((n_batch, n_ctx), dtype=np.float32)
    start = encoder['_start_']
    delimiter = encoder['_delimiter_'] 
    for i, (x1, x2, x3), in enumerate(zip(X1, X2, X3)):
        #x12 = [start] + x1[:max_len] + [delimiter] + x2[:max_len] + [delimiter] + x3[:max_len]+ [clf_token]
        x12 = [start] + x1 + [delimiter]  + x3+ [delimiter] + x2 + [delimiter] + x3+ [clf_token]
        #x12 = [start]  + x2 + [delimiter] + x3+ [clf_token]
        #x13 = [start] + x1[:max_len] + [delimiter] + x3[:max_len] + [clf_token]
        l12 = len(x12)
        if l12 == 0:
            print('O length train para\n')
            continue
        if l12 > 512:
            continue
        #l13 = len(x13)
        xmb[i, :l12, 0] = x12
        #xmb[i, 1, :l13, 0] = x13
        mmb[i, :l12] = 1
        #mmb[i, 1, :l13] = 1
    # Position information that is added to the input embeddings in the TransformerModel
    xmb[:, :, 1] = np.arange(n_vocab + n_special, n_vocab + n_special + n_ctx)
    return xmb, mmb

def transform_recipe_additional(X1, X2, X3):
    n_batch = len(X1)
    xmb = np.zeros((n_batch, n_ctx, 3), dtype=np.int32)
    mmb = np.zeros((n_batch, n_ctx), dtype=np.float32)
    start = encoder['_start_']
    delimiter = encoder['_delimiter_']
    for i, (x1, x2, x3), in enumerate(zip(X1, X2, X3)):
        #x12 = [start] + x1[:max_len] + [delimiter] + x2[:max_len] + [delimiter] + x3[:max_len]+ [clf_token]
        x12 = [start] + x1 + [delimiter] + x2 + [delimiter] + x3+ [clf_token]
        #x12 = [start]  + x2 + [delimiter] + x3+ [clf_token]
        #x13 = [start] + x1[:max_len] + [delimiter] + x3[:max_len] + [clf_token]
        l12 = len(x12)
        if l12 == 0:
            print('O length train para\n')
            continue
        if l12 > 512:
            continue
        #l13 = len(x13)
        xmb[i, :l12, 0] = x12
        #xmb[i, 1, :l13, 0] = x13
        mmb[i, :l12] = 1
        #mmb[i, 1, :l13] = 1
    # Position information that is added to the input embeddings in the TransformerModel
    xmb[:, :, 1] = np.arange(n_vocab + n_special, n_vocab + n_special + n_ctx)
    xmb[:,: len(x1)+2,2] = encoder['_extra1_']
    xmb[:, len(x1)+2: len(x1)+2 + len(x2)+1,2] = encoder['_extra2_']
    xmb[:, len(x1)+2 + len(x2)+1:len(x1)+2 + len(x2)+1 + len(x3)+1,2] = encoder['_extra3_']
    return xmb, mmb

def transform_recipe3(X1, X2, X3, X1_helper, X2_helper):
    n_batch = len(X1)
    xmb = np.zeros((n_batch, n_ctx, 4), dtype=np.int32)
    mmb = np.zeros((n_batch, n_ctx), dtype=np.float32)
    start = encoder['_start_']
    delimiter = encoder['_delimiter_']
    for i, (x1, x2, x3, x4, x5), in enumerate(zip(X1, X2, X3, X1_helper, X2_helper)):
        #x12 = [start] + x1[:max_len] + [delimiter] + x2[:max_len] + [delimiter] + x3[:max_len]+ [clf_token]
        x12 = [start] + x1 + [delimiter] + x2 + [delimiter] + x3+ [clf_token]
        x14 = [ing_not_present_token] + x4 + [ing_not_present_token] + x5 + [ing_not_present_token] + [ing_present_token]*len(x3) + [ing_not_present_token]

        assert len(x1) == len(x4)
        assert len(x2) == len(x5)
        #x12 = [start]  + x2 + [delimiter] + x3+ [clf_token]
        #x13 = [start] + x1[:max_len] + [delimiter] + x3[:max_len] + [clf_token]
        l12 = len(x12)
        if l12 == 0:
            print('O length train para\n')
            continue
        if l12 > 512:
            continue
        #l13 = len(x13)
        xmb[i, :l12, 0] = x12
        xmb[i, :l12, 3] = x14
        #xmb[i, 1, :l13, 0] = x13
        mmb[i, :l12] = 1
        #mmb[i, 1, :l13] = 1
    # Position information that is added to the input embeddings in the TransformerModel
    xmb[:, :, 1] = np.arange(n_vocab + n_special, n_vocab + n_special + n_ctx)
    xmb[:,: len(x1)+2,2] = encoder['_extra1_']
    xmb[:, len(x1)+2: len(x1)+2 + len(x2)+1,2] = encoder['_extra2_']
    xmb[:, len(x1)+2 + len(x2)+1:len(x1)+2 + len(x2)+1 + len(x3)+1,2] = encoder['_extra3_']
    return xmb, mmb
def transform_recipe_stories(X1):
    n_batch = len(X1)
    xmb = np.zeros((n_batch, n_ctx, 2), dtype=np.int32)
    mmb = np.zeros((n_batch, n_ctx), dtype=np.float32)
    start = encoder['_start_']
    delimiter = encoder['_delimiter_']
    for i, x1 in enumerate(X1):
        #x12 = [start] + x1[:max_len] + [delimiter] + x2[:max_len] + [delimiter] + x3[:max_len]+ [clf_token]
        x12 = [start] + x1 + [clf_token]
        #x12 = [start]  + x2 + [delimiter] + x3+ [clf_token]
        #x13 = [start] + x1[:max_len] + [delimiter] + x3[:max_len] + [clf_token]
        l12 = len(x12)
        if l12 == 0:
            print('O length train para\n')
            continue
        if l12 > 512:
            continue
        #l13 = len(x13)
        xmb[i, :l12, 0] = x12
        #xmb[i, 1, :l13, 0] = x13
        mmb[i, :l12-1] = 1
        #mmb[i, 1, :l13] = 1
    # Position information that is added to the input embeddings in the TransformerModel
    xmb[:, :, 1] = np.arange(n_vocab + n_special, n_vocab + n_special + n_ctx)
    return xmb, mmb


def iter_apply(Xs, Ms, Ys):
    # fns = [lambda x: np.concatenate(x, 0), lambda x: float(np.sum(x))]
    logits = []
    cost = 0
    with torch.no_grad():
        #dh_model.eval()
        for xmb, mmb, ymb in iter_data(Xs, Ms, Ys, n_batch=n_batch_train, truncate=False, verbose=True):
            n = len(xmb)
            XMB = torch.tensor(xmb, dtype=torch.long).to(device)
            YMB = torch.tensor(flatten_list(ymb), dtype=torch.long).to(device)
            MMB = torch.tensor(mmb).to(device)
            _, clf_logits = dh_model(XMB)
            #print("+"*80)
            #print(clf_logits)
            #print("="*80)
            clf_logits *= n
            #print(clf_logits)
            #print("+"*80)
            
            
            clf_losses = compute_loss_fct(XMB, YMB, MMB, clf_logits, only_return_losses=True)
            clf_losses *= n
            logits.append(clf_logits.to("cpu").numpy())
            cost += clf_losses.sum().item()
        logits = np.concatenate(logits, 0)
    return logits, cost

def iter_apply_lm(Xs, Ms, denom):
    # fns = [lambda x: np.concatenate(x, 0), lambda x: float(np.sum(x))]
    logits = []
    cost = 0
    total_loss = 0
    total_preds = 0
    with torch.no_grad():
        dh_model.eval()
        for xmb, mmb in iter_data(Xs, Ms, n_batch=n_batch_train, truncate=False, verbose=True):
            n = len(xmb)
            XMB = torch.tensor(xmb, dtype=torch.long).to(device)
            MMB = torch.tensor(mmb).to(device)
            lm_logits, clf_logits = dh_model(XMB)

            
            
            lm_losses = compute_loss_fct(XMB, None, MMB, None, lm_logits, only_return_losses=True)
            total_loss+= lm_losses.item()
            total_preds+= torch.sum(MMB[:,1:]).item()

    return total_loss / total_preds

def iter_predict(Xs, Ms):
    logits = []
    with torch.no_grad():
        dh_model.eval()
        for xmb, mmb in iter_data(Xs, Ms, n_batch=n_batch_train, truncate=False, verbose=True):
            n = len(xmb)
            XMB = torch.tensor(xmb, dtype=torch.long).to(device)
            MMB = torch.tensor(mmb).to(device)
            _, clf_logits = dh_model(XMB)
            logits.append(clf_logits.to("cpu").numpy())
    logits = np.concatenate(logits, 0)
    return logits

def log_lm(save_dir, desc):
    print("Logging")
    tr_cost = iter_apply_lm(trlmX[:374], trlmM[:374], 1.0)
    va_cost = iter_apply_lm(valmX, valmM, 1.0)
    logger.log(n_epochs=n_epochs, n_updates=n_updates, tr_cost=tr_cost, va_cost = va_cost)
    print('%d %d %.3f %.3f' % (n_epochs, n_updates, tr_cost, va_cost))


def log(save_dir, desc):
    global best_score
    print("Logging")
    tr_logits, tr_cost = iter_apply(trX[:n_valid], trM[:n_valid], trY[:n_valid])
    va_logits, va_cost = iter_apply(vaX, vaM, vaY)
    tr_cost = tr_cost / len(trY[:n_valid])
    va_cost = va_cost / n_valid
    tr_acc = accuracy_score(flatten_list(trY[:n_valid]), np.argmax(tr_logits, 1)) * 100.
    va_acc = accuracy_score(flatten_list(vaY), np.argmax(va_logits, 1)) * 100.
    logger.log(n_epochs=n_epochs, n_updates=n_updates, tr_cost=tr_cost, va_cost=va_cost, tr_acc=tr_acc, va_acc=va_acc)
    print('%d %d %.3f %.3f %.2f %.2f' % (n_epochs, n_updates, tr_cost, va_cost, tr_acc, va_acc))
    score = va_acc
    print(args)
    if score > best_score:
        best_score = score
        #path = os.path.join(save_dir, desc, 'best_params')
        #torch.save(dh_model.state_dict(), make_path(path))
        np.save('./train_transformer_recipes_replaced_lm{}_{}_{}_{}_{}.npy'.format(args.n_iter_lm, args.n_layer, args.n_head, args.n_embd, args.lmtotal), va_logits)

def predict(dataset, submission_dir):
    filename = filenames[dataset]
    pred_fn = pred_fns[dataset]
    label_decoder = label_decoders[dataset]
    predictions = pred_fn(iter_predict(teX, teM))
    if label_decoder is not None:
        predictions = [label_decoder[prediction] for prediction in predictions]
    path = os.path.join(submission_dir, filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write('{}\t{}\n'.format('index', 'prediction'))
        for i, prediction in enumerate(predictions):
            f.write('{}\t{}\n'.format(i, prediction))


def run_epoch():
    for xmb, mmb, ymb in iter_data(*shuffle(trlmX, trlmM, trlmM, random_state=np.random),
                                   n_batch=n_batch_train, truncate=True, verbose=True):
        global n_updates
        dh_model.train()
        XMB = torch.tensor(xmb, dtype=torch.long).to(device)
        YMB = torch.tensor(flatten_list(ymb), dtype=torch.long).to(device)
        MMB = torch.tensor(mmb).to(device)
        lm_logits, clf_logits = dh_model(XMB)
        compute_loss_fct(XMB, YMB, MMB, None,lm_logits)
        
        n_updates += 1




def run_epoch2():
    for xmb, mmb, ymb in iter_data(*shuffle(trX, trM, trYt, random_state=np.random),
                                   n_batch=n_batch_train, truncate=True, verbose=True):
        global n_updates
        dh_model.train()
        XMB = torch.tensor(xmb, dtype=torch.long).to(device)
        YMB = torch.tensor(flatten_list(ymb), dtype=torch.long).to(device)
        MMB = torch.tensor(mmb).to(device)
        lm_logits, clf_logits = dh_model(XMB)

        #if n_updates < 1400 or n_updates > 1500:
        compute_loss_fct(XMB, YMB, MMB, clf_logits,lm_logits)
        
        n_updates += 1
        if n_updates in [ 8000, 16000, 32000] and n_epochs == 0:
            log(save_dir, desc)

def run_epoch_lm():

    for xmb, mmb in iter_data(*shuffle(trlmX, trlmM, random_state=np.random),
                                   n_batch=n_batch_train, truncate=True, verbose=True):

        global n_updates
        dh_model.train()
        XMB = torch.tensor(xmb, dtype=torch.long).to(device)
        MMB = torch.tensor(mmb).to(device)
        lm_logits, _ = dh_model(XMB)
        compute_loss_fct(XMB, MMB, lm_logits)
        
        n_updates += 1
        '''
        if n_updates in [ 8000, 16000, 32000] and n_epochs == 0:
            log(save_dir, desc)

        '''    
argmax = lambda x: np.argmax(x, 1)

pred_fns = {
    'rocstories': argmax,
}

filenames = {
    'rocstories': 'ROCStories.tsv',
}

label_decoders = {
    'rocstories': None,
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--desc', type=str, help="Description")
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--log_dir', type=str, default='log/')
    parser.add_argument('--save_dir', type=str, default='save/')
    parser.add_argument('--data_dir', type=str, default='data/')
    parser.add_argument('--submission_dir', type=str, default='submission/')
    parser.add_argument('--submit', action='store_true')
    parser.add_argument('--analysis', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_iter_lm', type=int, default=10)
    parser.add_argument('--lmval', type=int, default=2000)
    parser.add_argument('--lmtotal', type=int, default=20000)
    parser.add_argument('--n_iter', type=int, default=25)
    parser.add_argument('--n_batch', type=int, default=4)
    parser.add_argument('--max_grad_norm', type=int, default=1)
    parser.add_argument('--lr', type=float, default=6.25e-5)
    parser.add_argument('--lr_warmup', type=float, default=0.002)
    parser.add_argument('--n_ctx', type=int, default=512)
    parser.add_argument('--n_embd', type=int, default=768)
    parser.add_argument('--n_head', type=int, default=8)
    parser.add_argument('--n_l ayer', type=int, default=8)
    parser.add_argument('--embd_pdrop', type=float, default=0.1)
    parser.add_argument('--attn_pdrop', type=float, default=0.1)
    parser.add_argument('--resid_pdrop', type=float, default=0.1)
    parser.add_argument('--clf_pdrop', type=float, default=0.1)
    parser.add_argument('--l2', type=float, default=0.01)
    parser.add_argument('--vector_l2', action='store_true')
    parser.add_argument('--opt', type=str, default='adam')
    parser.add_argument('--afn', type=str, default='gelu')
    parser.add_argument('--lr_schedule', type=str, default='warmup_linear')
    parser.add_argument('--encoder_path', type=str, default='model/encoder_bpe_40000.json')
    parser.add_argument('--bpe_path', type=str, default='model/vocab_40000.bpe')
    parser.add_argument('--n_transfer', type=int, default=12)
    parser.add_argument('--lm_coef', type=float, default=0.5)
    parser.add_argument('--b1', type=float, default=0.9)
    parser.add_argument('--b2', type=float, default=0.999)
    parser.add_argument('--e', type=float, default=1e-8)
    parser.add_argument('--n_valid', type=int, default=374)

    args = parser.parse_args()
    print(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Constants
    submit = args.submit
    dataset = args.dataset
    n_ctx = args.n_ctx
    save_dir = args.save_dir
    desc = 'train_transformer_recipes_lm{}_{}_{}_{}_{}'.format(args.n_iter_lm, args.n_layer, args.n_head, args.n_embd, args.lmtotal)
    print(desc)
    data_dir = args.data_dir
    log_dir = args.log_dir
    submission_dir = args.submission_dir

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    print("device", device, "n_gpu", n_gpu)

    logger = ResultLogger(path=os.path.join(log_dir, '{}.jsonl'.format(desc + 'replaced')), **args.__dict__)
    text_encoder = TextEncoder(args.encoder_path, args.bpe_path)
    encoder = text_encoder.encoder
    n_vocab = len(text_encoder.encoder)


    '''
    print("Encoding dataset...")
    ((trX1, trX2, trX3, trY),
     (vaX1, vaX2, vaX3, vaY),
     (teX1, teX2, teX3)) = encode_dataset(*rocstories(data_dir, n_valid=args.n_valid),
                                          encoder=text_encoder)


    '''


    #LM-pretraining

    encoder['_start_'] = len(encoder)
    encoder['_delimiter_'] = len(encoder)
    encoder['_classify_'] = len(encoder)
    encoder['_extra1_'] = len(encoder)
    encoder['_extra2_'] = len(encoder)
    encoder['_extra3_'] = len(encoder)
    encoder['_ing_present_'] = len(encoder)
    encoder['_ing_not_present_'] = len(encoder)

    clf_token = encoder['_classify_']
    ing_present_token = encoder['_ing_present_']
    ing_not_present_token = encoder['_ing_not_present_']
    n_special = 8

    '''
    train_file = json.load(open('../train_recipes.json','r'))
    val_file = json.load(open('../val_recipes.json','r'))


    t_passage = []
    v_passage = []
    for ins in train_file:
        t_passage.append(ins['story'])
    for ins in val_file:
        v_passage.append(ins['story'])

    a = (t_passage),(v_passage)
    ((trX1),(vaX1)) = encode_dataset(*a,encoder = text_encoder)

    '''
    train_lm_file = json.load(open('./train_gpt_whole_just_recipes.json','r'))
    train_file = json.load(open('./test_gpt_whole.json','r'))
    val_file = json.load(open('./val_gpt_whole_verbimputed.json','r'))

    #print(train_file[0])
    t_passage = []
    t_context = []
    t_ing = []
    t_gold = []
    tlm_passage = []
    tlm_context = []
    tlm_ing = []
    tlm_gold = []
    v_passage = []
    v_context = []
    v_ing = []
    v_gold = []
    t_all_ings = []
    v_all_ings = []
    print(len(train_lm_file))
    print(len(train_file))
    print(len(val_file))

    lmval = args.lmval
    lmtotal =args.lmtotal
    '''
    for idx, ins in enumerate(train_lm_file):

        max_len = 0
        tmp  = " ".join(ins['text'])
        curr_len = len(tmp.split())
        if curr_len > max_len:
            max_len = curr_len
        if idx%1000 == 0:
            print(idx)
            print(max_len)
            print('\n')
            max_len = 0
    '''
    '''
    f = open('./recipes.txt', 'w')

    for ins in train_lm_file[:lmtotal]:
        f.write(" ".join(ins['text'])+'\n')
    f.close()
    exit
    '''
    for ins in train_lm_file[:200]:
        tmp = ""
        tmp = " ".join(ins['text'])
        tmp = tmp.replace('-lrb-', '(')
        tmp = tmp.replace('-rrb-', ')')
        if len(tmp.split()) < 5:
            continue
        tlm_passage.append(tmp)
        tlm_ing.append(ins['ing'])
    print(tlm_passage[0])

    for ins in train_file[:500]:

        text= [step.replace('-lrb-','(').replace('-rrb-', ')') for step in ins['text']]
        t_passage.append(text)
        t_ing.append(ins['ing'])
        t_gold.append(ins['gold'])
        t_all_ings.append(ins['all_ings'])

    '''
    v_passage.append(['$'])
    v_ing.append('ok')
    v_gold.append([0])
    v_all_ings.append(['a','b'])
    '''
    for ins in val_file:
        text= [step.replace('-lrb-','(').replace('-rrb-', ')') for step in ins['ve_$replaced_text']]
        v_passage.append(text)
        v_ing.append(ins['ing'])
        v_gold.append(ins['gold'])
        v_all_ings.append(ins['all_ings'])
    #print(tlm_passage[0])


    a = (tlm_passage, tlm_ing,), (t_ing,t_gold),(v_ing,v_gold)

    ((trlmX1, trlmX2,),(trX2, trY),(vaX2, vaY)) = encode_dataset(*a,encoder = text_encoder)

    #trlmX1 = encode_dataset_whole(tlm_passage, encoder = text_encoder)
    trX1 = encode_dataset_whole(t_passage, encoder = text_encoder)
    vaX1 = encode_dataset_whole(v_passage, encoder = text_encoder)
    print(vaX1[0][1])
    trX3 = encode_dataset_whole(t_all_ings, encoder = text_encoder)
    vaX3 = encode_dataset_whole(v_all_ings, encoder = text_encoder)
    n_batch_train = args.n_batch * max(n_gpu, 1)
    print(n_ctx)
    vocab = n_vocab + n_special + n_ctx
    trlmX, trlmM = transform_recipe_whole_just_recipe(trlmX1, trlmX2,trlmX2)

    trlmX, valmX = trlmX[:-lmval], trlmX[-lmval:]
    trlmM, valmM = trlmM[:-lmval], trlmM[-lmval:]
    trX, trM = transform_recipe_whole(trX1, trX2, trX3)
    vaX, vaM = transform_recipe_whole(vaX1, vaX2, vaX3)
    n_train_lm  = len(trlmX)
    n_train = len(trY)
    n_valid = len(vaY)
    print(len(trlmX))
    print(len(valmX))
  
    dh_model = DoubleHeadModel(args, clf_token, 'custom', vocab, n_ctx)
    #load_openai_pretrained_model(dh_model.transformer, n_ctx=n_ctx, n_special=n_special)
    dh_model = nn.DataParallel(dh_model)
    path = os.path.join(save_dir, desc, 'best_params')
    dh_model.load_state_dict(torch.load(path))

    dh_model.module.transformer.embed.weight.data[289,:] = torch.zeros([768,], dtype= torch.long)
    #=  torch.zeros(768)
    dh_model.to(device)
    
    


    dh_model.eval()

    criterion = nn.CrossEntropyLoss(reduce=False)
    model_opt = OpenAIAdam(dh_model.parameters(),
                           lr=6.25e-5,
                           schedule=args.lr_schedule,
                           warmup=.002,
                           t_total=1000,
                           b1=args.b1,
                           b2=args.b2,
                           e=args.e,
                           l2=args.l2,
                           vector_l2=args.vector_l2,
                           max_grad_norm=args.max_grad_norm)


    compute_loss_fct = ClassificationLossCompute(criterion,
                                                 criterion,
                                                 args.lm_coef,
                                                 model_opt)

    n_updates = 0
    n_epochs = 0
    best_score = 0
    log(save_dir, desc)

