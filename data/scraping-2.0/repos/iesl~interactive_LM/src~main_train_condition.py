import argparse
import os, sys
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.cuda as cutorch

import gc

#import data
#import random
import model as model_code
import nsd_loss
#import coherency_eval
from gpt2_model.modeling_gpt2_condition import GPT2LMHeadModel
from gpt2_model.configuration_gpt2 import GPT2Config
from gpt2_model.optimization import AdamW

from utils import seed_all_randomness, create_exp_dir, save_checkpoint, load_idx2word_freq, load_emb_file_to_tensor, load_corpus, output_parallel_models, str2bool

parser = argparse.ArgumentParser(description='PyTorch Train Future Topic Prediction')
parser.add_argument('--data', type=str, default='./data/processed/wiki2016_gpt2/',
                    help='location of the data corpus')
parser.add_argument('--tensor_folder', type=str, default='tensors_all_min100',
                    help='location of the data corpus')
parser.add_argument('--save', type=str,  default='./models/',
                    help='path to save the final model')
parser.add_argument('--emb_file', type=str, default='./resources/glove.840B.300d.txt',
                    help='path to the file of a word embedding file')

parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=256,
                    help='sequence length')
#parser.add_argument('--dilated_head_span', type=int, default=50,
#                    help='The span of each head which generates the topics')
parser.add_argument('--n_further', type=int, default=50,
                    help='number of words we want to look ahead when we do sampling')
parser.add_argument('--turn_off_condition', type=str2bool, nargs='?', default=False,
                    help='Turn off the condition for debugging purpose')
parser.add_argument('--num_insert', type=int, default=5,
                    help='The number of places we insert the topics')
parser.add_argument('--min_rest_seq_len', type=int, default=20,
                    help='The future words can only insert from 0 to (bptt - min_rest_seq_len)')
parser.add_argument('--max_chosen_topics', type=int, default=10,
                    help='the number of chosen topics is sampled uniformly between 0 and max_chosen_topics')
#parser.add_argument('--avg_word_num', type=int, default=1,
#                    help='the number of word embeddings we take average before sending to GPT2 model')

parser.add_argument('--optimizer', type=str, default="SGD",
                    help='optimization algorithm. Could be SGD or Adam')
parser.add_argument('--lr', type=float, default=1,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=100,
                    help='upper epoch limit')
parser.add_argument('--nonmono', type=int, default=2,
                    help='random seed')
parser.add_argument('--wdecay', type=float, default=1.2e-6,
                    help='weight decay applied to all weights')
parser.add_argument('--small_batch_size', type=int, default=-1,
                    help='the batch size for computation. batch_size should be divisible by small_batch_size.\
                     In our implementation, we compute gradients with small_batch_size multiple times, and accumulate the gradients\
                     until batch_size is reached. An update step is then performed.')
parser.add_argument('--training_split_num', type=int, default=2,
                    help='We want to split training corpus into how many subsets. Splitting training dataset seems to make pytorch run much faster and we can store and eval the model more frequently')
parser.add_argument('--valid_per_epoch', type=int, default=2,
                    help='Number of times we want to run through validation data and save model within an epoch')
parser.add_argument('--start_training_split', type=int, default=0,
                    help='We want to split training corpus into how many subsets. Splitting training dataset seems to make pytorch run much faster and we can store and eval the model more frequently')

parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', type=str2bool, nargs='?', default=True,
#parser.add_argument('--cuda', action='store_false',
#parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--continue_train', action='store_true',
                    help='continue train from a checkpoint')
parser.add_argument('--single_gpu', default=False, action='store_true', 
                    help='use single GPU')


args = parser.parse_args()

########################
print("Set up environment")
########################

assert args.training_split_num >= args.valid_per_epoch

if args.small_batch_size < 0:
    args.small_batch_size = args.batch_size
assert args.batch_size % args.small_batch_size == 0, 'batch_size must be divisible by small_batch_size'

if not args.continue_train:
    args.save = '{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
create_exp_dir(args.save, scripts_to_save=['./src/main_train_topics.py', './src/model.py', './src/nsd_loss.py'])


def logging(s, print_=True, log_=True):
    if print_:
        print(s)
        sys.stdout.flush()
    if log_:
        with open(os.path.join(args.save, 'log.txt'), 'a+') as f_log:
            f_log.write(s + '\n')

# Set the random seed manually for reproducibility.
seed_all_randomness(args.seed,args.cuda)

logging('Args: {}'.format(args))

###############################################################################
# Load data
###############################################################################

device = torch.device("cuda" if args.cuda else "cpu")

idx2word_freq, dataloader_train_arr, dataloader_val = load_corpus(args.data, args.batch_size, args.batch_size, args.bptt, args.n_further, -1, device, args.tensor_folder, args.training_split_num, condition = True)

#corpus = data.Corpus(args.data)

#eval_batch_size = 10
#test_batch_size = 1
#train_data = batchify(corpus.train, args.batch_size, args)
#val_data = batchify(corpus.valid, eval_batch_size, args)
#test_data = batchify(corpus.test, test_batch_size, args)

def counter_to_tensor(idx2word_freq,device):
    total = len(idx2word_freq)
    w_freq = torch.zeros(total, dtype=torch.float, device = device, requires_grad = False)
    for i in range(total):
        w_freq[i] = 1
        #w_freq[i] = math.sqrt(idx2word_freq[x][1])
        #w_freq[i] = idx2word_freq[x][1]
    w_freq[0] = -1
    return w_freq

external_emb, output_emb_size, extra_init_idx = load_emb_file_to_tensor(args.emb_file, device, idx2word_freq)
external_emb = external_emb / (0.000000000001 + external_emb.norm(dim = 1, keepdim=True))
external_emb.requires_grad = False
print("loading ", args.emb_file)

w_freq = counter_to_tensor(idx2word_freq,device)

#print("Memory ",str(cutorch.memory_allocated(0)) , ' ', str(cutorch.max_memory_allocated(0)) , ' ' , str(cutorch.get_device_properties(0).total_memory))
###############################################################################
# Build the model
###############################################################################

model_name = 'gpt2'
gpt2_config = GPT2Config.from_pretrained(model_name)
gpt2_config.word_emb_dim = output_emb_size
if not args.continue_train:
    encoder = GPT2LMHeadModel.from_pretrained(model_name, config = gpt2_config)
else:
    encoder_state_dict = torch.load(os.path.join(args.save, 'encoder.pt'), map_location=device)
    encoder = GPT2LMHeadModel.from_pretrained(model_name, state_dict = encoder_state_dict, config = gpt2_config)
#if args.continue_train:
#    #model = torch.load(os.path.join(args.save, 'model.pt'))
#    model.load_state_dict(torch.load(os.path.join(args.save, 'model.pt')))
#    model_further.load_state_dict(torch.load(os.path.join(args.save, 'model_further.pt')))

if args.cuda:
    if args.single_gpu:
        parallel_encoder = encoder.cuda()
    else:
        parallel_encoder = nn.DataParallel(encoder, dim=0).cuda()
else:
    parallel_encoder = encoder

#print("Memory ",str(cutorch.memory_allocated(0)) , ' ', str(cutorch.max_memory_allocated(0)) , ' ' , str(cutorch.get_device_properties(0).total_memory))

total_params = sum(x.data.nelement() for x in encoder.parameters())
logging('Encoder total parameters: {}'.format(total_params))


###############################################################################
# Training code
###############################################################################


def evaluate(dataloader, external_emb):
    # Turn on evaluation mode which disables dropout.
    encoder.eval()
    total_loss = 0
    with torch.no_grad():
        for i_batch, sample_batched in enumerate(dataloader):
            feature, future_toks, idx_gpt2_to_spacy_small = sample_batched
            if args.turn_off_condition:
                outputs = encoder(feature, labels=feature)
                loss = outputs[0]
            else:
                insert_loc = np.random.choice(args.bptt - args.min_rest_seq_len, size=args.num_insert, replace = False)
                insert_loc.sort()
                chosen_topic_num = np.random.randint(0, args.max_chosen_topics+1, size=(args.num_insert))
                future_emb_chosen_arr = []
                for j in range(args.num_insert):
                    start = idx_gpt2_to_spacy_small[:, insert_loc[j]]
                    batch_size = future_toks.size(0)
                    idx_space = torch.tensor([list(range(start[k],start[k]+args.n_further)) for k in range(batch_size)],dtype = torch.long, device = device)
                    #future_sample_space = future_toks[:,start:start+args.n_further]
                    future_sample_space = future_toks.gather(dim = 1, index= idx_space)
                    select_idx = torch.randint(0, args.n_further, size=(batch_size,chosen_topic_num[j]),dtype = torch.long, device = device )
                    future_toks_chosen = future_sample_space.gather(dim = 1, index= select_idx)
                    future_emb_chosen_arr.append( external_emb[future_toks_chosen,:] )
                outputs = encoder(feature, labels=feature, insert_loc=insert_loc, future_emb_chosen_arr=future_emb_chosen_arr)
                loss = outputs[0]

            batch_size = feature.size(0)
            total_loss += loss * batch_size

    return total_loss.item() / len(dataloader.dataset)


def train_one_epoch(dataloader_train, external_emb, lr, split_i):
    start_time = time.time()
    total_loss = 0.
    
    encoder.train()
    for i_batch, sample_batched in enumerate(dataloader_train):
        #init_head_posi = random.randint(0, args.dilated_head_span - 1)
        #dilated_heads_posi = list(range(init_head_posi, len(feature), args.dilated_head_span))
        #if dilated_heads_posi[-1] != len(feature)-1:
        #    dilated_heads_posi.append(len(feature)-1)
        feature, future_toks, idx_gpt2_to_spacy_small = sample_batched
        if args.turn_off_condition:
            outputs = parallel_encoder(feature, labels=feature)
            loss = outputs[0]
        else:
            #insert_loc = random.sample( range(args.bptt - args.min_rest_seq_len), args.num_insert) #without replacement
            insert_loc = np.random.choice(args.bptt - args.min_rest_seq_len, size=args.num_insert, replace = False)
            insert_loc.sort()
            chosen_topic_num = np.random.randint(0, args.max_chosen_topics+1, size=(args.num_insert))
            future_emb_chosen_arr = []
            for j in range(args.num_insert):
                start = idx_gpt2_to_spacy_small[:, insert_loc[j]]
                batch_size = future_toks.size(0)
                idx_space = torch.tensor([list(range(start[k],start[k]+args.n_further)) for k in range(batch_size)],dtype = torch.long, device = device)
                #future_sample_space = future_toks[:,start:start+args.n_further]
                future_sample_space = future_toks.gather(dim = 1, index= idx_space)
                select_idx = torch.randint(0, args.n_further, size=(batch_size,chosen_topic_num[j]),dtype = torch.long, device = device )
                future_toks_chosen = future_sample_space.gather(dim = 1, index= select_idx)
                future_emb_chosen_arr.append( external_emb[future_toks_chosen,:] )
                #future_toks_chosen_emb = external_emb[future_toks_chosen,:]
                #if args.avg_word_num == 1 or chosen_topic_num[j]<=1:
                #    future_emb_chosen_arr.append( future_toks_chosen_emb )
                #else:
                #    future_toks_all_emb = external_emb[future_sample_space,:]
                #    merge_num = int(chosen_topic_num[j]/2)
                #    emb_sim = torch.bmm(future_toks_chosen_emb[:,:merge_num,:], future_toks_all_emb.permute(0,2,1))
                #    top_value, top_index = torch.topk(emb_sim, args.avg_word_num, dim = 2, sorted=False)
                #    #top_index should have size (batch_size, merge_num, args.avg_word_num)
                #    emb_size = future_toks_all_emb.size(2)
                #    future_toks_all_emb_expanded = future_toks_all_emb.unsqueeze(dim=1).expand(batch_size, merge_num, future_toks_all_emb.size(1), future_toks_all_emb.size(2))
                #    future_toks_topk_emb = future_toks_all_emb_expanded.gather(dim = 2, index = top_index.unsqueeze(dim=-1).expand(batch_size, merge_num, args.avg_word_num, emb_size))
                #    #future_toks_topk_emb should have size (batch_size, merge_num, args.avg_word_num, emb_size)
                #    future_toks_topk_sum_emb = torch.sum( future_toks_topk_emb * top_value.unsqueeze(-1), dim = 2) 
                #    future_toks_topk_sum_norm_emb = future_toks_topk_sum_emb / (0.000000000001 + future_toks_topk_sum_emb.norm(dim = -1, keepdim=True))
                #    future_toks_merge_emb = torch.cat( (future_toks_chosen_emb[:,merge_num:,:], future_toks_topk_sum_norm_emb), dim = 1)
                #    future_toks_merge_perm_emb = future_toks_merge_emb[:, torch.randperm(chosen_topic_num[j].item(),device=device),:]
                #    future_emb_chosen_arr.append( future_toks_merge_perm_emb )
            outputs = parallel_encoder(feature, labels=feature, insert_loc=insert_loc, future_emb_chosen_arr=future_emb_chosen_arr)
            loss = outputs[0]
        #feature.size(0) % args.dilated_head_span
        #target_idx = idx_feature_to_target[:,init_head_posi:].unfold(dim=1, args.n_further, args.dilated_head_span)
        #target_idx = torch.cat( (target_idx, idx_feature_to_target[:,-1]), dim=1)
        
        #print(feature.size())
        #print(target_unfold.size()) 
        #print(future_mask.size())
        #target should have size (batch, num_head, n_further)
        #target = target_unfold.view(-1, target_unfold.size(2) )

        optimizer_e.zero_grad()
        #output_emb, hidden, output_emb_last = parallel_encoder(feature.t())
        #output_emb, hidden, output_emb_last = parallel_encoder(feature)
        #output_emb_last, output_emb = parallel_encoder(feature)
        #output_emb, past = parallel_encoder(feature)
        #output_emb should have size (batch, seq_len, hidden_size)
        #hidden_size = output_emb.size(2)
        #batch_size, num_head, seq_len = future_mask.size()
        
        #print(inner_idx_tensor)
        #output_emb_head = output_emb.gather(dim=1,index=inner_idx_tensor.unsqueeze(dim=-1).expand(batch_size,num_head,hidden_size))
        #output_emb_head should have size (batch, num_head, hidden_size)
        #output_emb_last = output_emb_head.view(-1, output_emb_head.size(2))

        
        loss *= args.small_batch_size / args.batch_size
        total_loss += loss.item()

        loss.backward()

        gc.collect()
        
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), args.clip)
        optimizer_e.step()
        
        #del sample_batched
        #break
    
        #if args.update_target_emb:
        #    if args.optimizer == 'SGD':
        #        external_emb.data -= lr/10.0 * external_emb.grad.data
        #    else:
        #        external_emb.data -= 10/10.0 * external_emb.grad.data
        #    external_emb.data[0,:] = 0
        #    external_emb.grad.data.zero_()
        #    #with torch.no_grad():
        #    external_emb.data = external_emb.data / (0.000000000001 + external_emb.data.norm(dim = 1, keepdim=True))

        if i_batch % args.log_interval == 0 and i_batch > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            logging('| e {:3d} {:3d} | {:5d}/{:5d} b | lr {:02.2f} | ms/batch {:5.2f} | '
                    'l {:5.5f}  '.format(
                epoch, split_i, i_batch, len(dataloader_train.dataset) // args.batch_size, optimizer_e.param_groups[0]['lr'],
                elapsed * 1000 / args.log_interval, cur_loss))

            total_loss = 0.
            start_time = time.time()

if args.optimizer == 'SGD':
    optimizer_e = torch.optim.SGD(encoder.parameters(), lr=args.lr, weight_decay=args.wdecay)
elif args.optimizer == 'Adam':
    optimizer_e = torch.optim.Adam(encoder.parameters(), lr=args.lr, weight_decay=args.wdecay)
elif args.optimizer == 'AdamW':
    optimizer_e = AdamW(encoder.parameters(), lr=args.lr, weight_decay=args.wdecay)

if args.continue_train:
    optimizer_e_state_dict = torch.load(os.path.join(args.save, 'optimizer_e.pt'), map_location=device)
    optimizer_e.load_state_dict(optimizer_e_state_dict)

lr = args.lr
best_val_loss = None
nonmono_count = 0
saving_freq = int(math.floor(args.training_split_num / args.valid_per_epoch))

for epoch in range(1, args.epochs+1):
    epoch_start_time = time.time()
    for i in range(len(dataloader_train_arr)):
        if epoch == 1 and i < args.start_training_split:
            print("Skipping epoch "+str(epoch) + ' split '+str(i) )
            continue
        train_one_epoch(dataloader_train_arr[i], external_emb, lr, i)
        
        if i != args.training_split_num - 1 and (i + 1) % saving_freq != 0:
            continue

        val_loss_all = evaluate(dataloader_val, external_emb)
        logging('-' * 89)
        logging('| end of epoch {:3d} split {:3d} | time: {:5.2f}s | lr {:5.2f} | valid loss {:5.5f} '
                .format(epoch, i, (time.time() - epoch_start_time), lr, val_loss_all))
        
        val_loss_important = val_loss_all
        
        if not best_val_loss or val_loss_important < best_val_loss:
            #save_checkpoint(encoder, decoder, optimizer_e, optimizer_d, external_emb, args.save)
            torch.save(encoder.state_dict(), os.path.join(args.save, 'encoder.pt'))
            torch.save(external_emb, os.path.join(args.save, 'target_emb.pt'))
            torch.save(optimizer_e.state_dict(), os.path.join(args.save, 'optimizer_e.pt'))
            best_val_loss = val_loss_important
            logging('Models Saved')
        else:
            nonmono_count += 1
        
        if nonmono_count >= args.nonmono:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            nonmono_count = 0
            lr /= 4.0
            for param_group in optimizer_e.param_groups:
                param_group['lr'] = lr
