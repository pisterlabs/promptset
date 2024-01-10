#!/usr/bin/env python3

import argparse
import logging
from tqdm import trange
from tqdm import tqdm, trange
import torch
import torch.nn.functional as F
import numpy as np
import json, os
from pytorch_pretrained_bert import GPT2LMHeadModel, GPT2Tokenizer
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)

from pytorch_pretrained_bert.optimization_openai import OpenAIAdam
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)
def pre_process_datasets(encoded_datasets, input_len, cap_length):
    """ Pre-process datasets containing lists of tuples(story, 1st continuation, 2nd continuation, label)
        To Transformer inputs of shape (n_batch, n_alternative, length) comprising for each batch, continuation:
        input_ids[batch, alternative, :] = [start_token] + story[:cap_length] + [delimiter_token] + cont1[:cap_length] + [clf_token]
    """


    tensor_datasets = []
    for dataset in encoded_datasets:
        n_batch = len(dataset)
        input_ids = np.zeros((n_batch, input_len), dtype=np.int64)
        mc_token_ids = np.zeros((n_batch,), dtype=np.int64)
        lm_labels = np.full((n_batch, input_len), fill_value=-1, dtype=np.int64)
        #mc_labels = np.zeros((n_batch,), dtype=np.int64)
        for i, (story), in enumerate(dataset):
            with_cont1 = story[:cap_length]
            #with_cont2 = [start_token] + story[:cap_length] + [delimiter_token] + cont2[:cap_length] + [clf_token]
            #print(with_cont1)
            #print(input_ids[i, :len(with_cont1)] )
            input_ids[i, :len(with_cont1)] = with_cont1
            #input_ids[i, 1, :len(with_cont2)] = with_cont2
            mc_token_ids[i] = len(with_cont1) - 1
            #mc_token_ids[i, 1] = len(with_cont2) - 1
            lm_labels[i, :len(with_cont1)-1] = with_cont1[1:]
            #lm_labels[i, 1, :len(with_cont2)-1] = with_cont2[1:]
            #mc_labels[i] = mc_label
        all_inputs = (input_ids, mc_token_ids, lm_labels)
        tensor_datasets.append(tuple(torch.tensor(t) for t in all_inputs))
    return tensor_datasets
def load_recipes_dataset(dataset_path='./val_recipes.json'):
    train_file = json.load(open(dataset_path,'r'))
    output = []
    for ins in train_file:
        output.append(ins['story'])
    return output

def top_k_logits(logits, k):
    if k == 0:
        return logits
    values, _ = torch.topk(logits, k)
    min_values = values[:, -1]
    return torch.where(logits < min_values, torch.ones_like(logits, dtype=logits.dtype) * -1e10, logits)

def sample_sequence(model, length, start_token=None, batch_size=None, context=None, temperature=1, top_k=0, device='cuda', sample=True):
    if start_token is None:
        assert context is not None, 'Specify exactly one of start_token and context!'
        context = torch.tensor(context, device=device, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1)
    else:
        assert context is None, 'Specify exactly one of start_token and context!'
        context = torch.full((batch_size, 1), start_token, device=device, dtype=torch.long)
    prev = context
    output = context
    past = None
    with torch.no_grad():
        for i in trange(length):
            logits, past = model(prev, past=past)
            logits = logits[:, -1, :] / temperature
            logits = top_k_logits(logits, k=top_k)
            log_probs = F.softmax(logits, dim=-1)
            if sample:
                prev = torch.multinomial(log_probs, num_samples=1)
            else:
                _, prev = torch.topk(log_probs, k=1, dim=-1)
            output = torch.cat((output, prev), dim=1)
    return output
def tokenize_and_encode(obj):
        """ Tokenize and encode a nested object """
        if isinstance(obj, str):
            return enc.end(obj)
        elif isinstance(obj, int):
            return obj
        return list(tokenize_and_encode(o) for o in obj)
def run_model():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default='gpt2', help='pretrained model name or path to local checkpoint')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nsamples", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=-1)
    parser.add_argument("--length", type=int, default=-1)
    parser.add_argument("--temperature", type=int, default=1)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument('--unconditional', action='store_true', help='If true, unconditional generation.')


    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev set.")

    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument('--train_dataset', type=str, default='./train_recipes.json')
    parser.add_argument('--eval_dataset', type=str, default='./val_recipes.json')


    #parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_train_epochs', type=int, default=3)
    parser.add_argument('--train_batch_size', type=int, default=2)
    parser.add_argument('--eval_batch_size', type=int, default=2)
    parser.add_argument('--max_grad_norm', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=6.25e-5)
    parser.add_argument('--warmup_proportion', type=float, default=0.002)
    parser.add_argument('--lr_schedule', type=str, default='warmup_linear')
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--lm_coef', type=float, default=0.9)
    parser.add_argument('--n_valid', type=int, default=374)

    args = parser.parse_args()
    print(args)
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if args.batch_size == -1:
        args.batch_size = 1
    assert args.nsamples % args.batch_size == 0

    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    enc = GPT2Tokenizer.from_pretrained(args.model_name_or_path)
    model = GPT2LMHeadModel.from_pretrained(args.model_name_or_path)
    model.to(device)

    def tokenize_and_encode(obj):
        """ Tokenize and encode a nested object """
        if isinstance(obj, str):
            return enc.encode(obj)
        elif isinstance(obj, int):
            return obj
        return list(tokenize_and_encode(o) for o in obj)

    val_dataset = load_recipes_dataset(args.eval_dataset)
    train_dataset = load_recipes_dataset(args.train_dataset)

    datasets = (train_dataset[:50000],)
    encoded_datasets = tokenize_and_encode(datasets)


    max_length = model.config.n_positions
    print(max_length)
    print(encoded_datasets[0][0])
    input_length = max(len(story[:max_length]) + 2 for dataset in encoded_datasets for story in dataset)
    input_length = min(input_length, model.config.n_positions)  # Max size of input for the pre-trained model
    print(input_length)

    tensor_datasets = pre_process_datasets(encoded_datasets, input_length, max_length)
    train_tensor_dataset = tensor_datasets[0]
    #eval_tensor_dataset = tensor_datasets[1]

    train_data = TensorDataset(*train_tensor_dataset)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)


    param_optimizer = list(model.named_parameters())

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    num_train_optimization_steps = len(train_data) * args.num_train_epochs // args.train_batch_size
    optimizer = OpenAIAdam(optimizer_grouped_parameters,
                           lr=args.learning_rate,
                           warmup=args.warmup_proportion,
                           max_grad_norm=args.max_grad_norm,
                           weight_decay=args.weight_decay,
                           t_total=num_train_optimization_steps)
    #encoded_datasets = tokenize_and_encode(datasets)
    
    if args.do_train:
        model.train()
        nb_tr_steps, tr_loss, exp_average_loss = 0, 0, None
        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss = 0
            nb_tr_steps = 0
            tqdm_bar = tqdm(train_dataloader, desc="Training")
            for step, batch in enumerate(tqdm_bar):
                #print(batch)
                batch = tuple(t.to(device) for t in batch)
                input_ids, mc_token_ids, lm_labels = batch
                loss = model(input_ids, lm_labels = lm_labels)
                
                loss.backward()
                optimizer.step()
                tr_loss += loss.item()
                exp_average_loss = loss.item() if exp_average_loss is None else 0.7*exp_average_loss+0.3*loss.item()
                nb_tr_steps += 1
                tqdm_bar.desc = "Training loss: {:.2e} lr: {:.2e}".format(exp_average_loss, optimizer.get_lr()[0])
    








    model.eval()


    val_data_half = []
    for text in val_dataset:
        a = text.split()
        a_half = a[:int(len(a)//2)]
        val_data_half.append(" ".join(a_half))


    if args.length == -1:
        args.length = model.config.n_ctx // 2
    elif args.length > model.config.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % model.config.n_ctx)

    output_eval_file = os.path.join(args.output_dir, "val_samples.txt")
    writer = open(output_eval_file, "w")
    generated = 0
    for rec_text in val_data_half:

        context_tokens = enc.encode(rec_text)
        
        for _ in range(args.nsamples // args.batch_size):
            out = sample_sequence(
                model=model, length=args.length,
                context=context_tokens if not args.unconditional else None,
                start_token=enc.encoder['<|endoftext|>'] if args.unconditional else None,
                batch_size=args.batch_size,
                temperature=args.temperature, top_k=args.top_k, device=device
            )
            out = out[:, len(context_tokens):].tolist()
            for i in range(args.batch_size):
                generated += 1
                text = enc.decode(out[i])
                writer.write("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40)
                writer.write(rec_text + '\n')
                writer.write(text + '\n')
        writer.write("=" * 80 + '\n')
    writer.close()
if __name__ == '__main__':
    run_model()