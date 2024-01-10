import argparse
import os
import numpy as np
import random
import torch
from csv import writer
#import torch.nn as nn
#import torch.utils.data
#import coherency_eval

import sys
sys.path.insert(0, sys.path[0]+'/..')
from utils import seed_all_randomness, load_corpus, loading_all_models, str2bool, raw_sent_dataset, load_idx2word_freq
#from run_pplm import pplm
from run_pplm_fine_tuned import pplm
import ngram
import utils_testing
from gpt2_model.tokenization_gpt2 import GPT2Tokenizer

from gpt2_model.modeling_gpt2_condition import GPT2LMHeadModel
from gpt2_model.modeling_gpt2 import GPT2LMHeadModel as GPT2LM
from gpt2_model.configuration_gpt2 import GPT2Config

parser = argparse.ArgumentParser(description='PyTorch Interactive LM')

###path
parser.add_argument('--data', type=str, default='./data/processed/wiki2016_gpt2/',
                    help='location of the data corpus')
parser.add_argument('--tensor_folder', type=str, default='tensors_10000000_min100',
                    help='location of the data corpus')
parser.add_argument('--checkpoint_topics', type=str, default='',
                    help='topical model checkpoint to use')
parser.add_argument('--checkpoint_conditional', type=str, default='',
                    help='conditional LM model checkpoint to use')
parser.add_argument('--checkpoint_org', type=str, default='',
                    help='original LM model checkpoint to use')
parser.add_argument('--checkpoint_org_for_pplm', type=str, default='',
                    help='PPLM model checkpoint to use')
parser.add_argument('--emb_file', type=str, default='target_emb.pt',
                    help='path to the file of a word embedding file')
#parser.add_argument('--gpt2_vocab_file', type=str, default='./resources/distilgpt2-vocab.json',
#                    help='path to the file of a word embedding file')
parser.add_argument('--outf', type=str, default='gen_log/generated.txt',
                    help='output file for generated text')
parser.add_argument('--csv_outf', type=str, default='gen_log/output.csv',
                    help='output file for generated text')
parser.add_argument('--use_corpus', type=str, default='wiki',
                    help='which corpus format we want to load. Could be wiki or STS')
parser.add_argument('--STS_location', type=str, default='',
                    help='If --use_corpus is STS, load the testing file from this path')

###system
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda_topics', type=str2bool, nargs='?', default=True,
                    help='use CUDA for topical model')
parser.add_argument('--cuda_conditional', type=str2bool, nargs='?', default=True,
                    help='use CUDA for conditional LM')
parser.add_argument('--single_gpu', default=True, action='store_true',
                    help='use single GPU')
parser.add_argument('--batch_size', type=int, default=1, metavar='N',
                    help='batch size')
parser.add_argument('--num_sent_gen', type=int, default=3, metavar='N',
                    help='In each prompt, generate how many sentences')
parser.add_argument('--gen_sent_len', type=int, default=50, metavar='N',
                    help='In each prompt, generate sentences with length gen_sent_len')
parser.add_argument('--bptt', type=int, default=512,
#parser.add_argument('--bptt', type=int, default=256,
                    help='sequence length')
parser.add_argument('--bptt_conditional', type=int, default=256,
                    help='sequence length')
parser.add_argument('--dilated_head_span', type=int, default=80,
                    help='The span of each head which generates the topics')
parser.add_argument('--max_batch_num', type=int, default=100, 
                    help='number of batches for evaluation')
parser.add_argument('--topic_mode', type=str, default='NSD',
                    help='topical model we want to use. Could be NSD, radnom_word, cluster, LDA')
parser.add_argument('--LDA_model_path', type=str, default='',
                    help='path to the file of a LDA mdoel')
parser.add_argument('--word_emb_center_path', type=str, default='',
                    help='path to the file of a clustering results in a word embedding space')
parser.add_argument('--stop_word_file', type=str, default='./resources/stop_word_list',
                    help='path to the file of a stop word list')
#parser.add_argument('--readable_context', type=str2bool, nargs='?', default=True,
parser.add_argument('--readable_context', type=str2bool, nargs='?', default=False,
                    help='use CUDA for conditional LM')

utils_testing.add_model_arguments(parser)

args = parser.parse_args()

if args.emb_file == "target_emb.pt":
    args.emb_file =  os.path.join(args.checkpoint_topics,"target_emb.pt")

#if args.nhidlast2 < 0:
#    args.nhidlast2 = args.emsize

#if args.trans_nhid < 0:
#    args.trans_nhid = args.emsize

# Set the random seed manually for reproducibility.
seed_all_randomness(args.seed,args.cuda_topics)


########################
print("Loading data")
########################

#device_topics = torch.device("cuda:0" if args.cuda_topics else "cpu")
#device_conditional = torch.device("cuda:1" if args.cuda_conditional else "cpu")
device_topics = torch.device("cuda" if args.cuda_topics else "cpu")
device_conditional = torch.device("cuda" if args.cuda_conditional else "cpu")
device_pplm = torch.device("cuda" if args.cuda_conditional else "cpu")
device_gpt2 = "cuda" if torch.cuda.is_available() else "cpu"
#print(args.cuda_conditional, device_conditional)

#with open(args.gpt2_vocab_file) as f_in:
#    idx_l2_w_gpt2 = utils_testing.load_gpt2_vocab(f_in)
tokenizer_GPT2 = GPT2Tokenizer.from_pretrained('gpt2')

def load_STS_filtered(data_path):
    sent_list = []
    type_idx_list = []
    type_d2_idx = {}
    
    with open(data_path) as f_in:
        for line in f_in:
            fields = line.rstrip().split('\t')
            sent_list.append(fields[0])
            sent_type = fields[-1]
            if sent_type in type_d2_idx:
                type_idx = type_d2_idx[sent_type]
            else:
                type_idx = len(type_d2_idx)
                type_d2_idx[sent_type] = type_idx
            type_idx_list.append(type_idx)
    idx_l2_type = [''] * len(type_d2_idx)
    for type_name, type_idx in type_d2_idx.items():
        idx_l2_type[type_idx] = type_name
    return sent_list, type_idx_list, idx_l2_type

def load_STS_org(data_path):
    sent_list = []
    type_idx_list = []
    type_d2_idx = {}
    
    with open(data_path) as f_in:
        for line in f_in:
            fields = line.rstrip().split('\t')
            sent_list.append(fields[-2])
            sent_list.append(fields[-1])
            sent_type = fields[0]
            if sent_type in type_d2_idx:
                type_idx = type_d2_idx[sent_type]
            else:
                type_idx = len(type_d2_idx)
                type_d2_idx[sent_type] = type_idx
            type_idx_list.append(type_idx)
            type_idx_list.append(type_idx)
    idx_l2_type = [''] * len(type_d2_idx)
    for type_name, type_idx in type_d2_idx.items():
        idx_l2_type[type_idx] = type_name
    return sent_list, type_idx_list, idx_l2_type

#idx2word_freq, dataloader_train_arr, dataloader_val, dataloader_val_shuffled, max_sent_len = load_corpus(args.data, args.batch_size, args.batch_size, device )
#idx2word_freq, dataloader_train_arr, dataloader_val, dataloader_val_shuffled, max_sent_len = load_corpus(args.data, args.batch_size, args.batch_size, device, skip_training = True, want_to_shuffle_val = False )
random_start = True
#random_start = False
#if args.topic_mode == "NSD":
#    random_start = True
if args.use_corpus == 'STS':
    dictionary_input_name = args.data + "/" + args.tensor_folder + "/dict_idx_compact"
    with open(dictionary_input_name) as f_in:
        idx2word_freq = load_idx2word_freq(f_in)
    STS_sents, type_idx_list, idx_l2_type = load_STS_filtered(args.STS_location)
    dataset = raw_sent_dataset(STS_sents, type_idx_list, tokenizer_GPT2, args.bptt, device_topics)   
    want_to_shuffle_val = True
    dataloader_test = torch.utils.data.DataLoader(dataset, batch_size = args.batch_size, shuffle = want_to_shuffle_val, drop_last=True)
    dataloader_train = 0
else:
    idx2word_freq, dataloader_train_arr, dataloader_val, dataloader_test = load_corpus(args.data, args.batch_size, args.batch_size, args.bptt, -1, args.dilated_head_span, device_topics, args.tensor_folder, skip_training = True, want_to_shuffle_val = True, load_testing = True, random_start = random_start )
    #idx2word_freq, dataloader_train_arr, dataloader_val, dataloader_test = load_corpus(args.data, args.batch_size, args.batch_size, args.bptt, -1, args.dilated_head_span, device_topics, args.tensor_folder, skip_training = True, want_to_shuffle_val = False, load_testing = True, random_start = random_start )
    dataloader_train = dataloader_train_arr[0]
    idx_l2_type = []


#if args.topic_mode != 'NSD':
def convert_stop_to_ind_lower(f_in, idx2word_freq):
    stop_word_org_set = set()
    for line in f_in:
        w = line.rstrip()
        stop_word_org_set.add(w)
    stop_word_set = set()
    for idx, (w, freq) in enumerate(idx2word_freq):
        if w.lower() in stop_word_org_set:
            stop_word_set.add(idx)
    return stop_word_set
with open(args.stop_word_file) as f_in:
    stop_word_set = convert_stop_to_ind_lower(f_in, idx2word_freq)

########################
print("Loading Models")
########################

#load topical model
parallel_encoder, parallel_decoder, encoder, decoder, word_norm_emb = loading_all_models(args, idx2word_freq, device_topics)
output_emb_size = word_norm_emb.size(1)

print("encoder:", next(encoder.parameters()).device)

#load conditional LM model
model_name = 'gpt2'

encoder_state_dict = torch.load(os.path.join(args.checkpoint_conditional, 'encoder.pt'), map_location=device_conditional)
gpt2_config = GPT2Config.from_pretrained(model_name)
gpt2_config.word_emb_dim = output_emb_size
model_condition = GPT2LMHeadModel.from_pretrained(model_name, state_dict = encoder_state_dict, config = gpt2_config).to(device_conditional)
print("model condition:", next(model_condition.parameters()).device)



#load pplm model
if args.topic_mode == 'NSD':
    model_name_pplm = 'gpt2'

    #pplm_model = pplm(seed = 0, pretrained_model = model_name_pplm, device = device_pplm)
    pplm_model = pplm(seed = 0, pretrained_model = model_name_pplm, checkpoint = args.checkpoint_org_for_pplm, device = device_pplm)



#load gpt-2 model for generating perplexity
#gpt2_model = GPT2LMHeadModel.from_pretrained(model_name, state_dict = encoder_state_dict, config = gpt2_config).to(device_gpt2)
gpt2_model = GPT2LM.from_pretrained('gpt2-medium').to(device_gpt2)
print("gpt-2:", next(gpt2_model.parameters()).device)


model_org = None
if len(args.checkpoint_org) > 0:
    device_org = torch.device("cuda:1" if args.cuda_conditional else "cpu")
    encoder_state_dict = torch.load(os.path.join(args.checkpoint_org, 'encoder.pt'), map_location=device_org)
    model_org = GPT2LMHeadModel.from_pretrained(model_name, state_dict = encoder_state_dict, config = gpt2_config).cuda(device_conditional)
    print("gpt-2 finetued on wikipedia:", next(model_org.parameters()).device)



encoder.eval()
decoder.eval()
model_condition.eval()
gpt2_model.eval()

with open(args.csv_outf, 'w', encoding='utf-8') as csvOutf:
    with open(args.outf, 'w') as outf:
        if args.topic_mode == 'NSD':
            outf.write('Testing Prompts:\n\n')
            csvOutf = writer(csvOutf)
            csvOutf.writerow(['paragraph_previous', 'paragraph_last', 'topic_0', 'topic_1', 'topic_2', 'topic_3', 'topic_4', 'topic_5', 'topic_6', 'topic_7', 'topic_8', 'topic_9', 'selected_topics', 'sentence', 'model', 'other'])
            run_eval = True   
            if args.use_corpus == 'STS':
                run_eval = False
            utils_testing.visualize_interactive_LM(model_condition, pplm_model, gpt2_model, device_conditional, args.num_sent_gen, args.gen_sent_len, dataloader_test, parallel_encoder, parallel_decoder, word_norm_emb, idx2word_freq, outf, args.n_basis, args.max_batch_num, args.de_en_connection, tokenizer_GPT2, stop_word_set, args.bptt_conditional, csvOutf, model_org, args.readable_context, run_eval, args.use_corpus, idx_l2_type)
            #outf.write('Validation Prompts:\n\n')
            #utils_testing.visualize_interactive_LM(model_condition, pplm_model, gpt2_model, device_conditional, args.num_sent_gen, args.gen_sent_len, dataloader_val, parallel_encoder, parallel_decoder, word_norm_emb, idx2word_freq, outf, args.n_basis, args.max_batch_num, args.de_en_connection, tokenizer_GPT2, args.bptt_conditional)
            if dataloader_train:
                outf.write('Training Prompts:\n\n')
                utils_testing.visualize_interactive_LM(model_condition, pplm_model, gpt2_model, device_conditional, args.num_sent_gen, dataloader_train, parallel_encoder, parallel_decoder, word_norm_emb, idx2word_freq, outf, args.n_basis, args.max_batch_num, args.de_en_connection, tokenizer_GPT2, args.bptt_conditional, csvOutf, model_org, args.readable_context, run_eval, args.use_corpus, idx_l2_type)
        else:
            outf.write('Testing Prompts:\n\n')
            csvOutf = writer(csvOutf)
            csvOutf.writerow(['paragraph_previous', 'paragraph_last', 'topic_0', 'topic_1', 'topic_2', 'topic_3', 'topic_4', 'topic_5', 'topic_6', 'topic_7', 'topic_8', 'topic_9', 'selected_topics', 'sentence', 'topic_mode', 'other'])
            run_eval = True   
            if args.use_corpus == 'STS':
                run_eval = False
            utils_testing.testing_topic_baseline(model_condition, gpt2_model, device_conditional, args.num_sent_gen, args.gen_sent_len, dataloader_test, word_norm_emb, idx2word_freq, outf, args.n_basis, args.max_batch_num, tokenizer_GPT2, args.bptt_conditional, args.topic_mode, stop_word_set, parallel_encoder, parallel_decoder, args.de_en_connection, args.LDA_model_path, args.word_emb_center_path, csvOutf, args.readable_context, run_eval, args.use_corpus, idx_l2_type)
