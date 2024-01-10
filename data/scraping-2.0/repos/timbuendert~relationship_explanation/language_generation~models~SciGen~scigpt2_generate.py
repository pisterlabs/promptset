# adapted from https://github.com/Kel-Lu/SciGen/blob/master/val_generation.py

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import logging
from tqdm import trange

import torch
import torch.nn.functional as F
import numpy as np

from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer

from rouge_score import rouge_scorer
from bert_score import score

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop

MODEL_CLASSES = {
    'gpt2': (GPT2LMHeadModel, GPT2Tokenizer),
    'openai-gpt': (OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
}


def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size x vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits


def sample_sequence(model, length, context, tokenizer, num_samples=1, temperature=1, top_k=0, top_p=0.0, repetition_penalty=1.0,
                    is_xlnet=False, is_xlm_mlm=False, xlm_mask_token=None, xlm_lang=None, device='cpu'):
    context = torch.tensor(context, dtype=torch.long, device=device)
    context = context.unsqueeze(0).repeat(num_samples, 1)
    generated = context
    with torch.no_grad():
        for _ in range(length):
            inputs = {'input_ids': generated}
           
            outputs = model(**inputs)

            next_token_logits = outputs[0][:, -1, :] / (temperature if temperature > 0 else 1.)

            # repetition penalty from CTRL (https://arxiv.org/abs/1909.05858)
            for i in range(num_samples):
                for _ in set(generated[i].tolist()):
                    next_token_logits[i, _] /= repetition_penalty
                
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            if temperature == 0: # greedy sampling:
                next_token = torch.argmax(filtered_logits, dim=-1).unsqueeze(-1)
            else:
                next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            if next_token == tokenizer.convert_tokens_to_ids('<|endoftext|>'):
                return generated
            generated = torch.cat((generated, next_token), dim=1)
    return generated


def filter_output(text):
    stop_tokens = ['<|endoftext|>']
    counter_ex = ['i.e.', 'e.g.']
    
    found_idx = [text.find(t) for t in stop_tokens]

    valid_idx = []
    for i in found_idx:
        idx_valid = True
        if i == -1:
            continue
        text_portion = text[i-8:i+1]
        for c in counter_ex:
            if c in text_portion:
                idx_valid = False
                
        if idx_valid:
            valid_idx.append(i)
    if len(valid_idx) > 0:
        target_idx = min(valid_idx)
        text = text[: target_idx+1]
        return text
    else:
        return text

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True)
    parser.add_argument("--tokenizer_path", default=None, type=str, required=False,
                        help="Path to the tokenizer")
    parser.add_argument("--base_dir", type=str, default="")
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--output_file")
    parser.add_argument("--input_type", type=str, default='cond_sum')
    parser.add_argument("--split", type=str, default='test')
    parser.add_argument("--n_obs", type=int)

    parser.add_argument("--padding_text", type=str, default="")
    parser.add_argument("--length", type=int, default=20)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="temperature of 0 implies greedy sampling")
    parser.add_argument("--repetition_penalty", type=float, default=1.0,
                        help="primarily useful for CTRL model; in that case, use 1.2")
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--stop_token', type=str, default=None,
                        help="Token at which text generation is stopped")

    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()

    set_seed(args)
    if not args.tokenizer_path:
        args.tokenizer_path = args.model_name_or_path


    args.model_type = args.model_type.lower()
    model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer = tokenizer_class.from_pretrained(args.model_type,  truncation = True)
    
    special_tokens = {"additional_special_tokens": ["<|tgt|>"], 'sep_token': '<|SEP|>', 'pad_token': '<|PAD|>'}
    if args.input_type == 'intro_tfidf':
        special_tokens['additional_special_tokens'].append('<TFIDF>')
    if args.input_type == 'intro_entity':
        special_tokens['additional_special_tokens'].append('<TFIDF>')
        special_tokens['additional_special_tokens'].append('<ENT>')
    tokenizer.add_special_tokens( special_tokens )
    #print(tokenizer)

    model = model_class.from_pretrained(args.model_name_or_path)
    model.to(args.device)
    model.eval()
    model.resize_token_embeddings(len(tokenizer))

    print('Length tokenizer:', len(tokenizer))

    if args.length < 0 and model.config.max_position_embeddings > 0:
        args.length = model.config.max_position_embeddings
    elif 0 < model.config.max_position_embeddings < args.length:
        args.length = model.config.max_position_embeddings  # No generation bigger than model size 
    elif args.length < 0:
        args.length = MAX_LENGTH  # avoid infinite loop

    logger.info(args)

    target_lst = []
    output_lst = []

    ####

    n_counter = 0
    input_path = f"{args.base_dir}/{args.input_type}"
    with open(f"{input_path}/{args.split}.source", "r") as source_data, open(f"{input_path}/{args.split}.target", "r") as dataset_target:
        for sample, target in zip(source_data, dataset_target):

            if 'cited' in args.input_type:
                sample = '<|SEP|> ' + sample

            if args.n_obs:
                if n_counter == args.n_obs:
                    break

            if n_counter % 500 == 0:
                print(n_counter)

            sample = sample.replace('\n', '')
            target = target.replace('\n', '')

            raw_text = sample.replace('<DOC_SEP>', '<|SEP|>') + ' <|tgt|>'

            context = tokenizer.tokenize(raw_text)
            target = tokenizer.tokenize(target)
    
            context_tokens = tokenizer.convert_tokens_to_ids( context )
            target_tokens = tokenizer.convert_tokens_to_ids( target ) 


            sep_token_ind = tokenizer.convert_tokens_to_ids('<|SEP|>')

            # adapt sequence if too long
            if len(context_tokens) > 1024-(args.length+3):
                n_remove_tokens = len(context_tokens) - (1024-(args.length+3))
                ind_doc = context_tokens.index(sep_token_ind)
                len_princ = len(context_tokens[:ind_doc])
                len_cited = len(context_tokens[(ind_doc+1):])

                if (len_cited/(len_princ + len_cited)) > 0.75:
                    remove_cited = n_remove_tokens
                    remove_princ = 0

                elif (len_princ/(len_princ + len_cited)) > 0.75:
                    remove_princ = n_remove_tokens
                    remove_cited = 0

                else:
                    remove_princ = int(np.ceil(n_remove_tokens * (len_princ/(len_princ + len_cited))))
                    remove_cited = int(np.ceil(n_remove_tokens * (len_cited/(len_princ + len_cited))))

                context_tokens = context_tokens[:(ind_doc-remove_princ)] + context_tokens[(ind_doc):-(1+remove_cited)] + [context_tokens[-1]]
              
            sep_index = list(context_tokens).index(sep_token_ind)
            if 'cited' in args.input_type:
                if sep_index != 0:
                    print(f'Sep token index is {sep_index}')
                context_tokens.pop(sep_index)
                
            out = sample_sequence(
                model=model,
                context=context_tokens,
                tokenizer = tokenizer,
                num_samples=args.num_samples,
                length=args.length,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                device=args.device,
            )
            out = out[:, len(context_tokens):].tolist()

            for o in out:
                text = tokenizer.decode(o, clean_up_tokenization_spaces=True)
                text = filter_output(text)
            target = tokenizer.decode( target_tokens, clean_up_tokenization_spaces=True)
            target_lst.append( target.replace('<|endoftext|>', '') )             
            output_lst.append(text)
            n_counter += 1

    if args.output_file:
        with open(args.output_file+'.targets', 'w+') as w:
            for item in target_lst:
                w.write( item + '\n')
        with open(args.output_file+'.outputs', 'w+') as w:
            for item in output_lst:
                w.write( item + '\n')


    # evaluate
    rouges = ['rouge1', 'rougeL']
    scorer = rouge_scorer.RougeScorer(rouges, use_stemmer=True)
    scores = [scorer.score(target_lst[i], output_lst[i]) for i in range(len(output_lst))]
    #print(scores)
    
    avg_scores = []
    for r in rouges:
        r_scores = [scores[i][r] for i in range(len(scores))]
        f1 = np.mean([r_scores[i].fmeasure for i in range(len(r_scores))])
        prec = np.mean([r_scores[i].precision for i in range(len(r_scores))])
        rec = np.mean([r_scores[i].recall for i in range(len(r_scores))])
        avg_scores.append({'Score': r, 'Avg. F1': f1, 'Avg. Precision': prec, 'Avg. Recall': rec})
        #print(avg_scores[-1])

    print('Rouge Scores:\n{}'.format(avg_scores))

    P, R, F1 = score(output_lst, target_lst, lang="en-sci", verbose=True, idf=True)
    print('\nBert Scores:\nMean F1-score = {}\nMean Recall = {}\nMean Precision = {}\n'.format(F1.mean(), R.mean(), P.mean()))

    return target_lst, output_lst, avg_scores


if __name__ == '__main__':
    main()
