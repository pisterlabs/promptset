#

# python -u main_gpt.py --cuda --batch_size 20 --dropout 0.45 --dropouth 0.3 --dropouti 0.5 --wdrop 0.45 --chunk_size 10 --seed 141 --epoch 1000 --feature fixGPT

import argparse
import time
import math
import numpy as np
import os
import hashlib
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
import pickle
import pdb

import data
from model import RNNModel
from GPT_model import GPTRNNModel
import pickle as pkl
from splitcross import SplitCrossEntropyLoss
from utils import batchify, get_batch, get_batch_gpt, repackage_hidden
import tools
from pytorch_pretrained_bert import OpenAIGPTTokenizer, OpenAIGPTModel, OpenAIAdam
from torch.nn import CrossEntropyLoss

parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')

# data
parser.add_argument('--data', type=str, default='data/penn',
                    help='location of the data corpus')
parser.add_argument('--debug', default=False, action='store_true',
                    help='debug mode')
parser.add_argument('--log_interval', type=int, default=200,
                    help='log interval')
parser.add_argument('--wvec', type=str, default='',
                    help='load pretrained word vector')
parser.add_argument('--maxvocab', type=int, default=50000,
                    help='maximum word vector to load')
parser.add_argument('-n', '--name', default=tools.date_hash(),
                    help='checkpoint file name')
parser.add_argument('--output', metavar='SAVE DIR',
                    default='res/',
                    help='path to save result')

# GPT
parser.add_argument('--learning_rate', type=float, default=6.25e-5)
parser.add_argument('--lm_coef', type=float, default=0.9)
parser.add_argument('--warmup_proportion', type=float, default=0.002)
parser.add_argument('--max_grad_norm', type=int, default=1)
parser.add_argument('--weight_decay', type=float, default=0.01)
parser.add_argument('--lr_schedule', type=str, default='warmup_linear')

# model
parser.add_argument('--mode', type=str, default='',
                    help='whether to use GPT pre-trained model')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (LSTM, QRNN, GRU)')
parser.add_argument('--emsize', type=int, default=400,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=1150,
                    help='number of hidden units per layer')
parser.add_argument('--chunk_size', type=int, default=10,
                    help='number of units per chunk')
parser.add_argument('--nlayers', type=int, default=3,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=30,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=8000,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=80, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=70,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.4,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--dropouth', type=float, default=0.25,
                    help='dropout for rnn layers (0 = no dropout)')
parser.add_argument('--dropouti', type=float, default=0.5,
                    help='dropout for input embedding layers (0 = no dropout)')
parser.add_argument('--dropoute', type=float, default=0.1,
                    help='dropout to remove words from embedding layer (0 = no dropout)')
parser.add_argument('--wdrop', type=float, default=0.4,
                    help='amount of weight dropout to apply to the RNN hidden to hidden matrix')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--nonmono', type=int, default=5,
                    help='random seed')

parser.add_argument('--cuda', default=False, action='store_true',
                    help='use CUDA')

parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
# parser.add_argument('--save', type=str, default=randomhash + '.pt',
#                     help='path to save the final model')
parser.add_argument('--alpha', type=float, default=2,
                    help='alpha L2 regularization on RNN activation (alpha = 0 means no regularization)')
parser.add_argument('--beta', type=float, default=1,
                    help='beta slowness regularization applied on RNN activiation (beta = 0 means no regularization)')
parser.add_argument('--wdecay', type=float, default=1.2e-6,
                    help='weight decay applied to all weights')
parser.add_argument('--resume', type=str, default='',
                    help='path of model to resume')
parser.add_argument('--optimizer', type=str, default='sgd',
                    help='optimizer to use (sgd, adam)')
parser.add_argument('--when', nargs="+", type=int, default=[-1],
                    help='When (which epochs) to divide the learning rate by 10 - accepts multiple')
parser.add_argument('--finetuning', type=int, default=500,
                    help='When (which epochs) to switch to finetuning')
parser.add_argument('--philly', action='store_true',
                    help='Use philly cluster')
parser.add_argument('--feature', default='', type=str)
args = parser.parse_args()
args.tied = True

if not os.path.exists(args.output):
    os.makedirs(args.output)

save_dir = os.path.join(args.output, args.name)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
args.save_dir = save_dir
args.save = os.path.join(save_dir, args.name)

# Set the random seed manually for reproducibility.
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        tools.print_log(args.save, "WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)


###############################################################################
# Load data
###############################################################################


def model_save(fn):
    if args.philly:
        fn = os.path.join(os.environ['PT_OUTPUT_DIR'], fn)
    with open(fn, 'wb') as f:
        torch.save([model, criterion, optimizer, epoch], f)


def model_load(fn):
    global model, criterion, optimizer, start_epoch
    if args.philly:
        fn = os.path.join(os.environ['PT_OUTPUT_DIR'], fn)
    with open(fn, 'rb') as f:
        model, criterion, optimizer, start_epoch = torch.load(f)


if args.wvec:
    word_vec_dir = 'data/wordvec/'
    if not os.path.exists(word_vec_dir):
        os.makedirs(word_vec_dir)
    fn = 'corpus.{}.data'.format(hashlib.md5((args.data + args.wvec).encode()).hexdigest())  # 1ce....
    # Load preprocessed vocab dict
    if os.path.exists(fn):
        tools.print_log(args.save, 'Loading cached dataset...')
        corpus = torch.load(fn)
    elif args.wvec == 'glove':
        wvec_dir = word_vec_dir + args.wvec
        if not os.path.exists(wvec_dir):
            os.makedirs(wvec_dir)
            tools.load_wvec(args.wvec, max_vocab=args.maxvocab)
        tools.print_log(args.save, 'Producing dataset with pretrained word vectors...')
        word2idx = tools.pkl_loader(os.path.join('data/wordvec', args.wvec, 'words2idx'))
        idx2word = tools.pkl_loader(os.path.join('data/wordvec', args.wvec, 'idx2words'))
        corpus = data.Corpus(args.data, args.wvec, word2idx, idx2word)
        torch.save(corpus, fn)
    pre_emb = tools.pkl_loader(os.path.join('data/wordvec', args.wvec, 'word_vecs'))

else:
    fn = 'corpus.{}.data'.format(hashlib.md5(args.data.encode()).hexdigest())
    if args.philly:
        fn = os.path.join(os.environ['PT_OUTPUT_DIR'], fn)
    if os.path.exists(fn):
        tools.print_log(args.save, 'Loading cached dataset...')
        corpus = torch.load(fn)
    else:
        tools.print_log(args.save, 'Producing dataset...')
        corpus = data.Corpus(args.data)
        torch.save(corpus, fn)

# Generate data

eval_batch_size = 10
test_batch_size = 1

train_data = batchify(corpus.train, args.batch_size, args)  # tensor (46479 * 20) 929589 / tot words887521
val_data = batchify(corpus.valid, eval_batch_size, args)  # 7376 * 10  / 70390
test_data = batchify(corpus.test, test_batch_size, args)  # 82430 * 1 / 78669 (tot tokens) + 3761 ('eos')

if args.debug:
    train_data = train_data[:50]
    val_data = val_data[:50]
    test_data = test_data[:50]

###############################################################################
# Build the model
###############################################################################

################################################3

criterion = None

ntokens = len(corpus.dictionary)  # 10000

if args.mode == 'GPT':
    model = GPTRNNModel(args.model, ntokens, args.emsize, args.nhid, args.chunk_size, args.nlayers,
                        args.dropout, args.dropouth, args.dropouti, args.dropoute, args.wdrop, args.tied, args=args, )
else:
    if args.wvec:
        model = RNNModel(args.model, ntokens, args.emsize, args.nhid, args.chunk_size, args.nlayers,
                         args.dropout, args.dropouth, args.dropouti, args.dropoute, args.wdrop, args.tied,
                         pre_emb=pre_emb, )
    else:
        model = RNNModel(args.model, ntokens, args.emsize, args.nhid, args.chunk_size, args.nlayers,
                         args.dropout, args.dropouth, args.dropouti, args.dropoute, args.wdrop, args.tied, )

if 'fixLastBlock' in args.feature.split('_'):
    # pdb.set_trace()
    transformer = [m for i, m in model.named_children() if i == 'transformer'][0]
    blocks = [m for i, m in transformer.named_children() if i == 'h'][0]
    for i, m in blocks.named_children():
        if i == '11':
            break
        else:
            for p in m.parameters():
                p.requries_grad = False

     # TODO there is BERTLayerNorm()

###
if args.resume:
    tools.print_log(args.save, 'Resuming model ...')
    model_load(args.resume)
    optimizer.param_groups[0]['lr'] = args.lr
    model.dropouti, model.dropouth, model.dropout, args.dropoute = args.dropouti, args.dropouth, args.dropout, args.dropoute
    if args.wdrop:
        for rnn in model.rnn.cells:
            rnn.hh.dropout = args.wdrop
###
if not criterion:
    splits = []
    if ntokens > 500000:
        # One Billion
        # This produces fairly even matrix mults for the buckets:
        # 0: 11723136, 1: 10854630, 2: 11270961, 3: 11219422
        splits = [4200, 35000, 180000]
    elif ntokens > 75000:
        # WikiText-103
        splits = [2800, 20000, 76000]
    tools.print_log(args.save, splits)
    criterion = SplitCrossEntropyLoss(args.emsize, splits=splits, verbose=False)
    if args.mode == 'GPT':
        criterion_gpt = CrossEntropyLoss(ignore_index=-1)

###
if args.cuda:
    model = model.cuda()
    criterion = criterion.cuda()
###
params = list(filter(lambda x: x.requires_grad, model.parameters())) + list(criterion.parameters())
total_params = sum(p.data.nelement() for p in params if p.requires_grad)
if args.mode == 'GPT':
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight', 'ln_'] # Add 'ln_1' to test if it's better
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay) and 'transformer' in n],
         'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay) and 'transformer' in n],
         'weight_decay': 0.0}
    ]
    num_train_optimization_steps = train_data.size(0) * args.epochs // args.batch_size
    optimizer_gpt = OpenAIAdam(optimizer_grouped_parameters,
                               lr=args.learning_rate,
                               warmup=args.warmup_proportion,
                               max_grad_norm=args.max_grad_norm,
                               weight_decay=args.weight_decay,
                               t_total=num_train_optimization_steps)
    params = [p for n, p in param_optimizer if 'transformer' not in n]

tools.print_log(args.save, args)
tools.print_log(args.save, 'Model total parameters:{}'.format(total_params))
if args.mode == 'GPT':
    tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
    with open('GPT_index.pkl', 'rb') as handle:
        gptdic = pickle.load(handle)


###############################################################################
# Training code
###############################################################################

def evaluate(data_source, batch_size=10):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    with torch.no_grad():
        if args.model == 'QRNN': model.reset()
        total_loss = 0
        ntokens = len(corpus.dictionary)
        hidden = model.init_hidden(batch_size)
        for i in range(0, data_source.size(0) - 1, args.bptt):
            # data, targets = get_batch(data_source, i, args, evaluation=True)
            if args.mode == 'GPT':
                data, targets, gpt_ids, fl_ids, tgt_gpt_id = get_batch_gpt(data_source, i, args, gptdic, tokenizer, )
            else:
                data, targets = get_batch(data_source, i, args)

            if args.mode == 'GPT':
                output, hidden = model(data, hidden, gpt_ids, fl_ids)
            else:
                output, hidden = model(data, hidden)
            total_loss += len(data) * criterion(model.decoder.weight, model.decoder.bias, output, targets).data
            hidden = repackage_hidden(hidden)
            torch.cuda.empty_cache()
    return total_loss.item() / len(data_source)


def train():
    # Turn on training mode which enables dropout.
    if args.model == 'QRNN': model.reset()
    total_loss = 0
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(args.batch_size)
    batch, i = 0, 0
    model.train()
    while i < train_data.size(0) - 1 - 1:
        bptt = args.bptt if np.random.random() < 0.95 else args.bptt / 2.
        # Prevent excessively small or negative sequence lengths
        seq_len = max(5, int(np.random.normal(bptt, 5)))
        # There's a very small chance that it could select a very long sequence length resulting in OOM
        # seq_len = min(seq_len, args.bptt + 10)

        lr2 = optimizer.param_groups[0]['lr']  # 16 param
        optimizer.param_groups[0]['lr'] = lr2 * seq_len / args.bptt
        if args.mode == 'GPT':
            data, targets, gpt_ids, fl_ids, tgt_gpt_id = get_batch_gpt(train_data, i, args, gptdic, tokenizer,
                                                                   seq_len=seq_len)
        # data : SL * BS; target:(SL*BS); gpt_ids : BS * SL_GPT; fl_ids : BS * (2 * SL)
        else:
            data, targets = get_batch(train_data, i, args, seq_len=seq_len)

        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)
        optimizer.zero_grad()
        if args.mode == 'GPT':
            optimizer_gpt.zero_grad()
        if args.mode == 'GPT':
            output, hidden, rnn_hs, dropped_rnn_hs, gpt_out = model(data, hidden, gpt_ids, fl_ids, return_h=True)
        # output: target_len * ES;
            raw_loss = criterion(model.decoder.weight, model.decoder.bias, output, targets)
            gpt_loss = criterion_gpt(gpt_out, tgt_gpt_id)
            loss = args.lm_coef * raw_loss + gpt_loss
        else:
            output, hidden, rnn_hs, dropped_rnn_hs = model(data, hidden, return_h=True)
            raw_loss = criterion(model.decoder.weight, model.decoder.bias, output, targets)
            loss = raw_loss

        # Activiation Regularization
        if args.alpha:
            loss = loss + sum(
                args.alpha * dropped_rnn_h.pow(2).mean()
                for dropped_rnn_h in dropped_rnn_hs[-1:]
            )
        # Temporal Activation Regularization (slowness)
        if args.beta:
            loss = loss + sum(
                args.beta * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean()
                for rnn_h in rnn_hs[-1:]
            )
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        if args.clip: torch.nn.utils.clip_grad_norm_(params, args.clip)
        optimizer.step()
        if args.mode == 'GPT':
            optimizer_gpt.step()

        total_loss += raw_loss.data
        optimizer.param_groups[0]['lr'] = lr2
        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss.item() / args.log_interval
            elapsed = time.time() - start_time
            tools.print_log(args.save, '| epoch {:3d} | {:5d}/{:5d} batches | lr {:05.5f} | ms/batch {:5.2f} | '
                                       'loss {:5.2f} | ppl {:8.2f} | bpc {:8.3f}'.format(
                epoch, batch, len(train_data) // args.bptt, optimizer.param_groups[0]['lr'],
                              elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss), cur_loss / math.log(2)))
            total_loss = 0
            start_time = time.time()
            # Print GPU Memory Usage
            if args.cuda:
                print('GPU Memory Usage:' + os.popen('nvidia-smi | grep MiB').read().split('|')[2].strip())
            ###
        batch += 1
        i += seq_len
        torch.cuda.empty_cache()
    if args.mode == 'GPT':
        del data, targets, gpt_ids, fl_ids, hidden, loss, raw_loss
    torch.cuda.empty_cache()


# Loop over epochs.
lr = args.lr
best_val_loss = []
stored_loss = 100000000

# At any point you can hit Ctrl + C to break out of training early.
try:
    if not args.resume:
        optimizer = None
        # Ensure the optimizer is optimizing params, which includes both the model's weights as well as the criterion's weight (i.e. Adaptive Softmax)
        if args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(params, lr=args.lr, weight_decay=args.wdecay)
        if args.optimizer == 'adam':
            optimizer = torch.optim.Adam(params, lr=args.lr, betas=(0, 0.999), eps=1e-9, weight_decay=args.wdecay)
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', 0.5, patience=2, threshold=0)
        start_epoch = 1
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start_time = time.time()
        train()
        torch.cuda.empty_cache()
        if 't0' in optimizer.param_groups[0]:
            tmp = {}
            for prm in params:
                tmp[prm] = prm.data.clone()
                try:
                    prm.data = optimizer.state[prm]['ax'].clone()
                except KeyError:
                    pass

            val_loss2 = evaluate(val_data, eval_batch_size)
            torch.cuda.empty_cache()
            tools.print_log(args.save, '-' * 89)
            tools.print_log(args.save, '| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                                       'valid ppl {:8.2f} | valid bpc {:8.3f}'.format(
                epoch, (time.time() - epoch_start_time), val_loss2, math.exp(val_loss2), val_loss2 / math.log(2)))
            tools.print_log(args.save, '-' * 89)

            if val_loss2 < stored_loss:
                model_save(args.save + '.pt')
                tools.print_log(args.save, 'Saving Averaged!')
                stored_loss = val_loss2

            for prm in params:
                prm.data = tmp[prm].clone()

            if epoch == args.finetuning:
                tools.print_log(args.save, 'Switching to finetuning')
                optimizer = torch.optim.ASGD(params, lr=args.lr, t0=0, lambd=0., weight_decay=args.wdecay)
                best_val_loss = []

            if epoch > args.finetuning and len(best_val_loss) > args.nonmono and val_loss2 > min(
                    best_val_loss[:-args.nonmono]):
                tools.print_log(args.save, 'Done!')
                import sys

                sys.exit(1)

        else:
            val_loss = evaluate(val_data, eval_batch_size)
            tools.print_log(args.save, '-' * 89)
            tools.print_log(args.save, '| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                                       'valid ppl {:8.2f} | valid bpc {:8.3f}'.format(
                epoch, (time.time() - epoch_start_time), val_loss, math.exp(val_loss), val_loss / math.log(2)))
            tools.print_log(args.save, '-' * 89)

            if val_loss < stored_loss:
                model_save(args.save + '.pt')
                tools.print_log(args.save, 'Saving model (new best validation)')
                stored_loss = val_loss

            if args.optimizer == 'adam':
                scheduler.step(val_loss)

            if args.optimizer == 'sgd' and 't0' not in optimizer.param_groups[0] and (
                    len(best_val_loss) > args.nonmono and val_loss > min(best_val_loss[:-args.nonmono])):
                tools.print_log(args.save, 'Switching to ASGD')
                optimizer = torch.optim.ASGD(params, lr=args.lr, t0=0, lambd=0., weight_decay=args.wdecay)

            if epoch in args.when:
                tools.print_log(args.save, 'Saving model before learning rate decreased')
                model_save('{}.e{}'.format(args.save + '.pt', epoch))
                tools.print_log(args.save, 'Dividing learning rate by 10')
                optimizer.param_groups[0]['lr'] /= 10.

            best_val_loss.append(val_loss)

        tools.print_log(args.save, "PROGRESS: {}%".format((epoch / args.epochs) * 100))
        # torch.cuda.empty_cache()

except KeyboardInterrupt:
    tools.print_log(args.save, '-' * 89)
    tools.print_log(args.save, 'Exiting from training early')

try:
    # Load the best saved model.
    model_load(args.save + '.pt')

    # Run on test data.
    test_loss = evaluate(test_data, test_batch_size)
    tools.print_log(args.save, '=' * 89)
    tools.print_log(args.save, '| End of training | test loss {:5.2f} | test ppl {:8.2f} | test bpc {:8.3f}'.format(
        test_loss, math.exp(test_loss), test_loss / math.log(2)))
    tools.print_log(args.save, '=' * 89)

except FileNotFoundError:
    print('No model saved so skipped testing.')
