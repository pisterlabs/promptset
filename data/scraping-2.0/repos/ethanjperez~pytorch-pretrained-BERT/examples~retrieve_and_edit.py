import argparse
import json
import time
from multiprocessing.pool import ThreadPool
from functools import partial
import numpy as np
import os
import pickle
from pytorch_pretrained_bert import (GPT2Tokenizer, GPT2LMHeadModel, GPT2DoubleHeadsModel,
                                     OpenAIGPTTokenizer, OpenAIGPTLMHeadModel, OpenAIGPTDoubleHeadsModel)
import struct
from tqdm import tqdm
import torch

from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances_argmin_min
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans, AffinityPropagation, Birch, MeanShift, estimate_bandwidth

format_tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')


def read_npy_chunk(filename, start_row, num_rows):
    """
    Copyright (c) 2012 by David Warde-Farley

    Reads a partial array (contiguous chunk along the first
    axis) from an NPY file.

    Parameters
    ----------
    filename : str
        Name/path of the file from which to read.
    start_row : int
        The first row of the chunk you wish to read. Must be
        less than the number of rows (elements along the first
        axis) in the file.
    num_rows : int
        The number of rows you wish to read. The total of
        `start_row + num_rows` must be less than the number of
        rows (elements along the first axis) in the file.

    Returns
    -------
    out : ndarray
        Array with `out.shape[0] == num_rows`, equivalent to
        `arr[start_row:start_row + num_rows]` if `arr` were
        the entire array (note that the entire array is never
        loaded into memory by this function).
    """
    assert start_row >= 0 and num_rows > 0
    with open(filename, 'rb') as fhandle:
        major, minor = np.lib.format.read_magic(fhandle)
        shape, fortran, dtype = np.lib.format.read_array_header_1_0(fhandle)
        assert not fortran, "Fortran order arrays not supported"
        # Make sure the offsets aren't invalid.
        assert start_row < shape[0], (
            'start_row is beyond end of file'
        )
        assert start_row + num_rows <= shape[0], (
            'start_row + num_rows > shape[0]'
        )
        # Get the number of elements in one 'row' by taking
        # a product over all other dimensions.
        row_size = np.prod(shape[1:])
        start_byte = start_row * row_size * dtype.itemsize
        fhandle.seek(start_byte, 1)
        n_items = row_size * num_rows
        flat = np.fromfile(fhandle, count=n_items, dtype=dtype)
        return flat.reshape((-1,) + shape[1:])


def cosine_similarity_chunk(i, X, Y, num_chunks):
    Y_chunked = Y[(i * Y.shape[0]) // num_chunks: ((i + 1) * Y.shape[0]) // num_chunks]
    return cosine_similarity(X, Y_chunked)


def clean_up_tokenization_spaces(out_string):
    """Converts an output string (de-BPE-ed) using de-tokenization algorithm from OpenAI GPT."""
    out_string = out_string.replace('<unk>', '')
    out_string = out_string.replace(' .', '.').replace(' ?', '?').replace(' !', '!').replace(' ,', ','
            ).replace(" ' ", "'").replace(" n't", "n't").replace(" 'm", "'m").replace(" do not", " don't"
            ).replace(" 's", "'s").replace(" 've", "'ve").replace(" 're", "'re")
    return out_string


def word_tokenizer(text):
    return format_tokenizer.decode(format_tokenizer.encode(text), clean_up_tokenization_spaces=False).split()


def word_untokenizer(word_list):
    return clean_up_tokenization_spaces(' '.join(word_list)).strip()


def get_tokenizer_class(model_name):
    """ Returns a tokenizer's Python class """
    return OpenAIGPTTokenizer if model_name == 'openai-gpt' else GPT2Tokenizer


def get_model_class(model_name, task_name):
    """ Returns a model's Python class """
    if task_name == 'rocstories':
        return OpenAIGPTDoubleHeadsModel if model_name == 'openai-gpt' else GPT2DoubleHeadsModel
    else:
        return OpenAIGPTLMHeadModel if model_name == 'openai-gpt' else GPT2LMHeadModel


def load_model(saved_dir):
    """ Loads a previously saved model """
    output_args_file = os.path.join(saved_dir, 'training_args.bin')
    args = torch.load(output_args_file)
    print('Loaded args:', args)
    tokenizer_class = get_tokenizer_class(args.model_name)
    tokenizer = tokenizer_class.from_pretrained(saved_dir)
    model_class = get_model_class(args.model_name, args.task_name)
    model = model_class.from_pretrained(saved_dir)
    return model, tokenizer, args


def main():
    """
    D_q: Questions from SQuAD + HotpotQA Easy (used to train low-level QA model)
    q_1, ..., q_K: Q's KNNs from D_q, given by word overlap (TF-IDF weighted)
    w_1, ..., w_M: Q's top M TF-IDF weighted tokens
    p_q: Language model trained on D_q

    # Replace <= L words with ones from {w_1, ..., w_M}, while maintaining p_q(q_k') >= p_q(q_k)
    for each q_k \in q_1, ..., q_K:
        Try replacing q_k_i with w_m (M x |q_k| many possible pairs).
        If p_q(q_k'') >= p_q(q_k'), keep this replacement.
        Try this until all w_1, ..., w_M are exhausted.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--filter_name", default="bi-cond-lm", type=str, required=False, help="The name of the task to train.")
    parser.add_argument("--top_knn", default=10, type=int, required=False, help="# of top nearest neighbors to modify.")
    parser.add_argument("--max_mods_per_nn", default=4, type=int, required=False, help="Max # modified Qs to accept per NN Q.")
    parser.add_argument("--seed", default=42, type=int, required=False, help="Random seed")
    parser.add_argument("--num_shards", default=1, type=int, required=False,
                        help="# of total data splits for distributed eval")
    parser.add_argument("--shard_no", default=0, type=int, required=False, help="Distributed eval data split index.")
    args = parser.parse_args()
    args.filter_name = args.filter_name.lower()
    save_filename = 'data/hotpot-all/split=train-dev.filter_name={}.top_knn={}.max_mods_per_nn={}.num_shards={}.shard_no={}.json'.format(
        args.filter_name, args.top_knn, args.max_mods_per_nn, args.num_shards, args.shard_no)
    print('Saving to', save_filename)

    # Load HotpotQA
    qis = []
    cis = []
    idis = []
    for split in ['train', 'dev']:
        with open('data/hotpot-all/{}.json'.format(split), 'r') as f:
            data_hotpot = json.load(f)
        for article in tqdm(data_hotpot['data']):
            for paragraph in article['paragraphs']:
                for qa in paragraph['qas']:
                    qis.append(qa['question'].strip())
                    cis.append(paragraph['context'])
                    idis.append(qa['id'])
    print('HotpotQA Qs:', len(qis))

    # Load SQuAD 2
    qks = []
    for split in ['train', 'dev']:
        with open('data/squad/{}-v2.0.json'.format(split), 'r') as f:
            data_squad = json.load(f)
        for article in tqdm(data_squad['data']):
            for paragraph in article['paragraphs']:
                for qa in paragraph['qas']:
                    qks.append(qa['question'].strip())
    print('SQuAD 2 Qs:', len(qks))

    # Fit TFIDF
    start_time = time.time()
    tfidf_filepath = 'data/hotpot-all/tfidf.pkl'
    if os.path.exists(tfidf_filepath):
        tfidf = pickle.load(open(tfidf_filepath, 'rb'))
    else:
        print('TFIDF Fitting...')
        tfidf = TfidfVectorizer(tokenizer=word_tokenizer, min_df=2, analyzer='word')
        tfidf.fit(qis + qks)
        pickle.dump(tfidf, open(tfidf_filepath, 'wb'))
    print('Got TFIDF in {:.0f}s with vocab size {:d}'.format(
        time.time() - start_time, len(tfidf.get_feature_names())))
    
    # HotpotQA: Transform Qs into TFIDF vectors
    start_time = time.time()
    tfidf_hotpot_filepath = 'data/hotpot-all/tfidf_hotpot.npy'
    if os.path.exists(tfidf_hotpot_filepath):
        tfidf_hotpot = np.load(tfidf_hotpot_filepath).item()
    else:
        print('Hotpot vectorization...')
        tfidf_hotpot = tfidf.transform(qis)
        np.save(tfidf_hotpot_filepath, tfidf_hotpot).item()
    print('Got Hotpot vectors in {:.0f}s'.format(time.time() - start_time))

    # SQuAD: Transform Qs into TFIDF vectors
    start_time = time.time()
    tfidf_squad_filepath = 'data/hotpot-all/tfidf_squad.npy'
    if os.path.exists(tfidf_squad_filepath):
        tfidf_squad = np.load(tfidf_squad_filepath).item()
    else:
        print('SQuAD vectorization...')
        tfidf_squad = tfidf.transform(qks)
        np.save(tfidf_squad_filepath, tfidf_squad)
    print('Got SQuAD vectors in {:.0f}s'.format(time.time() - start_time))

    # HotpotQA-SQuAD TFIDF cosine distances
    start_time = time.time()
    tfidf_cosine_filepath = 'data/hotpot-all/tfidf_cosine.npy'
    start_row_index = (args.shard_no * len(qis)) // args.num_shards
    end_row_index = ((args.shard_no + 1) * len(qis)) // args.num_shards
    if os.path.exists(tfidf_cosine_filepath):
        # tfidf_cosine = np.load(tfidf_cosine_filepath)
        tfidf_cosine = np.zeros((len(qis), len(qks)))
        tfidf_cosine[start_row_index: end_row_index] = read_npy_chunk(
            tfidf_cosine_filepath, start_row_index, end_row_index - start_row_index)
    else:
        print('Cosine(Hotpot Qs, SQuAD Qs)...')
        num_chunks = 240
        with ThreadPool(48) as p:
            partial_cosine_similarity = partial(cosine_similarity_chunk, X=tfidf_hotpot, Y=tfidf_squad, num_chunks=num_chunks)
            tfidf_cosine = np.hstack(list(tqdm(p.imap(partial_cosine_similarity, list(range(num_chunks))), total=num_chunks)))
        np.save(tfidf_cosine_filepath, tfidf_cosine)
    print('Got Cosine(Hotpot Qs, SQuAD Qs) in {:.0f}s'.format(time.time() - start_time))

    # Load LM for filtering Q's
    if 'lm' in args.filter_name:
        if 'cond-lm' in args.filter_name:
            model, model_tokenizer, model_args = load_model(
                'checkpoint/tn=squad-questions-cond-lm.mn=gpt2-medium.tbs=8.lr=6.25e-05')
        else:  # Unconditional LM
            model, model_tokenizer, model_args = load_model(
                'checkpoint/tn=squad-questions-lm.mn=openai-gpt.tbs=64.lr=6.25e-5')
        model.half()
        model.to('cuda')
        model.eval()

        def eval_prob(text):
            indexed_tokens = model_tokenizer.encode(text)
            tokens_tensor = torch.tensor([indexed_tokens]).to('cuda')

            with torch.no_grad():
                predictions = model(tokens_tensor)
                if isinstance(predictions, tuple):
                    predictions, past = predictions

            prob_dists = torch.nn.functional.softmax(predictions, dim=-1)
            input_probs = prob_dists[0].gather(1, tokens_tensor[0].unsqueeze(-1))  # NB: Change for batch size > 1
            return input_probs.mean().item()

    # Find and modify NNs for each HotpotQA Q
    data_hotpot_new = {'data': []}
    for i, qi in tqdm(enumerate(qis)):
        if (i < start_row_index) or (i >= end_row_index):
            continue  # Handle by different shard_no

        qi_words = word_tokenizer(qi)
        qi_unwords = word_untokenizer(qi_words)
        qi_prob = eval_prob(qi_unwords) if 'lm' in args.filter_name else 1.0
        print('Original Q #{}: {} ({:.0%})'.format(i, qi_unwords.capitalize(), qi_prob))

        sorted_q_idxs_squad = tfidf_cosine[i].argsort()[::-1]
        data_hotpot_new['data'].append({
            'paragraphs': [{
                'context': cis[i],
                'original_question': qi,
                'nns': [],
                'qas': []
            }],
            'title': ''
        })

        for k_rank in range(args.top_knn):
            start_time = time.time()
            k = sorted_q_idxs_squad[k_rank]
            qk = qks[k]
            data_hotpot_new['data'][-1]['paragraphs'][0]['nns'].append(qk)
            qk_words = word_tokenizer(qk)
            qk_unwords = word_untokenizer(qk_words)
            if 'bi-cond-lm' in args.filter_name:
                qk_prob1 = eval_prob(qi_unwords + ' ' + qk_unwords)
                qk_prob2 = eval_prob(qk_unwords + ' ' + qi_unwords)
                qk_prob = (qk_prob1 + qk_prob2) / 2.
            elif 'cond-lm' in args.filter_name:
                qk_prob = eval_prob(qi_unwords + ' ' + qk_unwords)
            elif 'lm' in args.filter_name:
                qk_prob = eval_prob(qk_unwords)
            elif args.filter_name in {'none', 'all'}:
                qk_prob = 1.0
            else:
                raise NotImplementedError('filter_name {}'.format(args.filter_name))

            if args.filter_name == 'all':
                continue

            # qkp_infos = set([])  # Avoid duplicates
            max_ngram_size = 3
            all_qkp_unwords = set([])
            for qi_ngram_size in range(1, max_ngram_size + 1):
                for qk_ngram_size in range(1, max_ngram_size + 1):  # range(max_ngram_size + 1) to include insertions
                    for qi_start_pos in range(len(qi_words) - qi_ngram_size):  # Excludes '?'
                        for qk_start_pos in range(len(qk_words) - qk_ngram_size):  # Excludes '?'
                            qkp_words = qk_words[:qk_start_pos] + \
                                         qi_words[qi_start_pos: qi_start_pos + qi_ngram_size] + \
                                         qk_words[qk_start_pos + qk_ngram_size:]
                            qkp_unwords = word_untokenizer(qkp_words)
                            if len(qkp_unwords) > 0:
                                all_qkp_unwords.add(qkp_unwords)

            if len(all_qkp_unwords) == 0:
                continue
            all_qkp_unwords = list(all_qkp_unwords)
            all_qkp_tfidf = tfidf.transform(all_qkp_unwords)
            all_qi2qkp_tfidf_cosine = cosine_similarity(tfidf_hotpot[i], all_qkp_tfidf)[0]
            all_qk2qkp_tfidf_cosine = cosine_similarity(tfidf_squad[k], all_qkp_tfidf)[0]
            qkp_infos = list(zip(all_qkp_unwords, all_qi2qkp_tfidf_cosine, all_qk2qkp_tfidf_cosine))
            print('= NN #{}: {} ({:.4%}) (TFIDF: {:.2f}) (Calc TFIDFs: {:.2f}s)'.format(
                k_rank + 1, qk_unwords.capitalize(), qk_prob, tfidf_cosine[i, k], time.time() - start_time))
            start_time = time.time()

            max_lm_evals = 32 * args.max_mods_per_nn
            top_qkp_infos = sorted(qkp_infos, key=lambda x: x[1]-x[2], reverse=True)[:max_lm_evals]
            num_lm_approved_qkps = 0
            for qkp_unwords, qi2qkp_tfidf_cosine, qk2qkp_tfidf_cosine in top_qkp_infos:
                if 'bi-cond-lm' in args.filter_name:
                    qkp_prob1 = eval_prob(qi_unwords + ' ' + qkp_unwords)
                    if qkp_prob1 < qk_prob1:
                        continue
                    qkp_prob2 = eval_prob(qkp_unwords + ' ' + qi_unwords)
                    if qkp_prob2 < qk_prob2:
                        continue
                    qkp_prob = (qkp_prob1 + qkp_prob2) / 2.
                elif 'cond-lm' in args.filter_name:
                    qkp_prob = eval_prob(qi_unwords + ' ' + qkp_unwords)
                elif 'lm' in args.filter_name:
                    qkp_prob = eval_prob(qkp_unwords)
                else:  # No filtering
                    qkp_prob = 1.0

                if qkp_prob < qk_prob:
                    continue

                print('=== TFIDF-qi: {:.2f}, TFIDF-qk: {:.2f}, {:.4%}, {}'.format(
                    qi2qkp_tfidf_cosine, qk2qkp_tfidf_cosine, qkp_prob, qkp_unwords.capitalize()))
                data_hotpot_new['data'][-1]['paragraphs'][0]['qas'].append({
                    'question': qkp_unwords,
                    'answers': [[] for _ in range(len(cis[i]))],
                    'id': idis[i] + '-' + str(len(data_hotpot_new['data'][-1]['paragraphs'][0]['qas'])),
                    'k_rank': k_rank,
                    'mod_rank': num_lm_approved_qkps,
                })
                num_lm_approved_qkps += 1
                if num_lm_approved_qkps >= args.max_mods_per_nn:
                    break

            print('= NN #{}: {} LM-approved mods found ({:.2f}s)'.format(
                k_rank + 1, num_lm_approved_qkps, time.time() - start_time))

    with open(save_filename, 'w') as f:
        json.dump(data_hotpot_new, f)

    return


if __name__ == '__main__':
    main()
