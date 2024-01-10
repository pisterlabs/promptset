"""Data loading utilities."""

import glob
import os

import numpy as np
import portalocker
import torch

from utils.vocabulary import OpenAIVocab, Vocab


class LMOrderedIterator:
    def __init__(self, data, bsz, bptt, device='cpu', ext_len=None):
        """
            data -- LongTensor -- the LongTensor is strictly ordered
        """
        self.bsz = bsz
        self.bptt = bptt
        self.ext_len = ext_len if ext_len is not None else 0

        self.device = device

        # Work out how cleanly we can divide the dataset into bsz parts.
        self.n_step = data.size(0) // bsz

        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        data = data.narrow(0, 0, self.n_step * bsz)

        # Evenly divide the data across the bsz batches.
        self.data = data.view(bsz, -1).t().contiguous().to(device)

        # Number of mini-batches
        self.n_batch = (self.n_step + self.bptt - 1) // self.bptt

    def get_batch(self, i, bptt=None):
        if bptt is None: bptt = self.bptt
        seq_len = min(bptt, self.data.size(0) - 1 - i)

        end_idx = i + seq_len
        beg_idx = max(0, i - self.ext_len)

        data = self.data[beg_idx:end_idx]
        target = self.data[i+1:i+1+seq_len]

        return data, target, seq_len

    def get_fixlen_iter(self, start=0):
        for i in range(start, self.data.size(0) - 1, self.bptt):
            yield self.get_batch(i)
    
    def get_varlen_iter(self, start=0, std=5, min_len=5, max_deviation=3):
        max_len = self.bptt + max_deviation * std
        i = start
        while True:
            bptt = self.bptt if np.random.random() < 0.95 else self.bptt / 2.
            bptt = min(max_len, max(min_len, int(np.random.normal(bptt, std))))
            data, target, seq_len = self.get_batch(i, bptt)
            i += seq_len
            yield data, target, seq_len
            if i >= self.data.size(0) - 2:
                break

    def __iter__(self):
        """Wrapper for get_fixlen_iter."""
        return self.get_fixlen_iter()


class LMShuffledIterator:
    def __init__(self, data, bsz, bptt, device='cpu', ext_len=None, shuffle=False):
        """
            data -- list[LongTensor] -- there is no order among the LongTensors
        """
        self.data = data

        self.bsz = bsz
        self.bptt = bptt
        self.ext_len = ext_len if ext_len is not None else 0

        self.device = device
        self.shuffle = shuffle

    def get_sent_stream(self):
        # index iterator
        epoch_indices = np.random.permutation(len(self.data)) if self.shuffle \
            else np.array(range(len(self.data)))

        # sentence iterator
        for idx in epoch_indices:
            yield self.data[idx]

    def stream_iterator(self, sent_stream):
        # streams for each data in the batch
        streams = [None] * self.bsz

        data = torch.LongTensor(self.bptt, self.bsz)
        target = torch.LongTensor(self.bptt, self.bsz)

        n_retain = 0

        while True:
            # data   : [n_retain+bptt x bsz]
            # target : [bptt x bsz]
            data[n_retain:].fill_(-1)
            target.fill_(-1)

            valid_batch = True

            for i in range(self.bsz):
                n_filled = 0
                try:
                    while n_filled < self.bptt:
                        if streams[i] is None or len(streams[i]) <= 1:
                            streams[i] = next(sent_stream)
                        # number of new tokens to fill in
                        n_new = min(len(streams[i]) - 1, self.bptt - n_filled)
                        # first n_retain tokens are retained from last batch
                        data[n_retain+n_filled:n_retain+n_filled+n_new, i] = \
                            streams[i][:n_new]
                        target[n_filled:n_filled+n_new, i] = \
                            streams[i][1:n_new+1]
                        streams[i] = streams[i][n_new:]
                        n_filled += n_new
                except StopIteration:
                    valid_batch = False
                    break

            if not valid_batch:
                return

            data = data.to(self.device)
            target = target.to(self.device)

            yield data, target, self.bptt

            n_retain = min(data.size(0), self.ext_len)
            if n_retain > 0:
                data[:n_retain] = data[-n_retain:]
            data.resize_(n_retain + self.bptt, data.size(1))

    def __iter__(self):
        # sent_stream is an iterator
        sent_stream = self.get_sent_stream()

        for batch in self.stream_iterator(sent_stream):
            yield batch


class LMMultiFileIterator(LMShuffledIterator):
    def __init__(self, paths, vocab, bsz, bptt, device='cpu', ext_len=None,
        shuffle=False):

        self.paths = paths
        self.vocab = vocab

        self.bsz = bsz
        self.bptt = bptt
        self.ext_len = ext_len if ext_len is not None else 0

        self.device = device
        self.shuffle = shuffle

    def get_sent_stream(self, path):
        sents = self.vocab.encode_file(path, add_double_eos=True)
        if self.shuffle:
            np.random.shuffle(sents)
        # Create virtual sentences for wikipedia data.
        # TODO: Load a few files at a time, then chunk?
        if type(sents) == torch.Tensor:
            return iter(sents.split(len(sents) // self.bsz))
        return iter(sents)

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.paths)

        for path in self.paths:
            # sent_stream is an iterator
            sent_stream = self.get_sent_stream(path)
            for batch in self.stream_iterator(sent_stream):
                yield batch


class Corpus:
    def __init__(self, path, dataset, use_bpe, *args, **kwargs):
        self.dataset = dataset
        if use_bpe:
            self.vocab = OpenAIVocab(kwargs['max_size'], kwargs.get('vocab_file'))
        else:
            self.vocab = Vocab(*args, **kwargs)

        if self.dataset in ['ptb', 'wt2', 'enwik8', 'text8']:
            self.vocab.count_file(os.path.join(path, 'train.txt'))
            self.vocab.count_file(os.path.join(path, 'valid.txt'))
            self.vocab.count_file(os.path.join(path, 'test.txt'))
        elif self.dataset == 'wt103' or self.dataset == 'wt2':
            self.vocab.count_file(os.path.join(path, 'train.txt'))
        elif self.dataset == 'wt103-normal':
            self.vocab.count_file(os.path.join(path, 'wiki.train.tokens'))
        elif self.dataset == 'lm1b':
            train_path_pattern = os.path.join(
                path, '1-billion-word-language-modeling-benchmark-r13output',
                'training-monolingual.tokenized.shuffled', 'news.en-*')
            train_paths = glob.glob(train_path_pattern)
        elif self.dataset == 'wiki':
            file_path_pattern = os.path.join(path, '*/wiki_*.txt')
            file_paths = glob.glob(file_path_pattern)
            assert file_paths, f'Nothing found at {file_path_pattern}' 

        # the vocab will load from file when build_vocab() is called
        self.vocab.build_vocab()

        if self.dataset in ['ptb', 'wt2', 'wt103']:
            self.train = self.vocab.encode_file(
                os.path.join(path, 'train.txt'), ordered=True)
            self.valid = self.vocab.encode_file(
                os.path.join(path, 'valid.txt'), ordered=True)
            self.test = self.vocab.encode_file(
                os.path.join(path, 'test.txt'), ordered=True)
        elif self.dataset in ['enwik8', 'text8']:
            self.train = self.vocab.encode_file(
                os.path.join(path, 'train.txt'), ordered=True, add_eos=False)
            self.valid = self.vocab.encode_file(
                os.path.join(path, 'valid.txt'), ordered=True, add_eos=False)
            self.test = self.vocab.encode_file(
                os.path.join(path, 'test.txt'), ordered=True, add_eos=False)
        elif self.dataset == 'lm1b':
            self.train = train_paths
            self.valid = self.vocab.encode_file(
                os.path.join(path, 'valid.txt'), ordered=False, add_double_eos=True)
            self.test = self.vocab.encode_file(
                os.path.join(path, 'test.txt'), ordered=False, add_double_eos=True)
        elif self.dataset == 'wiki':
            valid_path = sorted(file_paths)[42]
            test_path = sorted(file_paths)[1337]
            self.valid = self.vocab.encode_file(valid_path, ordered=True)
            self.test = self.vocab.encode_file(test_path, ordered=True)
            self.train = list(set(file_paths) - set((valid_path, test_path)))
        elif self.dataset in ['wt103-normal']:
            self.train = self.vocab.encode_file(
                os.path.join(path, 'wiki.train.tokens'), ordered=True, add_eos=False)
            self.valid = self.vocab.encode_file(
                os.path.join(path, 'wiki.valid.tokens'), ordered=True, add_eos=False)
            self.test = self.vocab.encode_file(
                os.path.join(path, 'wiki.test.tokens'), ordered=True, add_eos=False)

    def get_dist_iterator(self, split: str, *args, rank: int = 0, max_rank: int = 1, **kwargs):
        """Get an iterator that only operates on rank//max_rank independent subset of the data."""
        data = self.__getattribute__(split)
        subset = list(chunk(data, max_rank))[rank]
        if self.dataset in ['lm1b', 'wiki']:
            if split == 'train':
                return LMMultiFileIterator(subset, self.vocab, *args, **kwargs)
        
        return LMOrderedIterator(subset, *args, **kwargs)

    def get_iterator(self, split: str, *args, **kwargs):
        """Get an iterator over the corpus.

        Each next() returns (data, target, seq_length).
        data and target have shape (bptt, bsz) and seq_length is a scalar.
        """
        data = self.__getattribute__(split)
        if self.dataset in ['ptb', 'wt2', 'wt103', 'enwik8', 'text8', 'wt103-normal']:
            return LMOrderedIterator(data, *args, **kwargs)
        if self.dataset == 'lm1b':
            if split in ['valid', 'test']:
                return LMShuffledIterator(data, *args, **kwargs)
            
            kwargs['shuffle'] = True
            return LMMultiFileIterator(data, self.vocab, *args, **kwargs)
        if self.dataset == 'wiki':
            if split == 'train':
                return LMMultiFileIterator(data, self.vocab, *args, **kwargs)
            return LMOrderedIterator(data, *args, **kwargs)


def get_lm_corpus(datadir: str, dataset: str, use_bpe=False, max_size=None) -> Corpus:
    """Factory method for Corpus.

    Arguments:
        datadir: Where does the data live?
        dataset: eg 'wt103' which tells the Corpus how to parse the data.
    """
    cache_filepath = os.path.join(datadir, 'cache.pt.bpe' if use_bpe else 'cache.pt')
    # Don't cache dataset for wiki, it's just a file list.
    if os.path.exists(cache_filepath) and dataset != 'wiki':
        print('Loading cached dataset...')
        corpus = torch.load(cache_filepath)
    else:
        print('Producing dataset {}...'.format(dataset))
        kwargs = {'max_size': max_size}
        if dataset in ['wt103', 'wt2', 'wt103-normal']:
            kwargs['special'] = ['<eos>']
            kwargs['lower_case'] = False
        elif dataset == 'ptb':
            kwargs['special'] = ['<eos>']
            kwargs['lower_case'] = True
        elif dataset == 'lm1b':
            kwargs['special'] = []
            kwargs['lower_case'] = False
            kwargs['vocab_file'] = os.path.join(datadir, '1b_word_vocab.txt')
        elif dataset in ['enwik8', 'text8']:
            pass

        corpus = Corpus(datadir, dataset, use_bpe, **kwargs)
        with portalocker.Lock(cache_filepath, timeout=60) as _:
            torch.save(corpus, cache_filepath)

    return corpus

def chunk(a: list, n: int):
    """Split `a` into `n` chunks, with the last bucket taking the remaining.
    
    https://stackoverflow.com/a/2135920
    """
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

def main():
    import argparse
    parser = argparse.ArgumentParser(description='unit test')
    parser.add_argument('--datadir', type=str, default='../data/text8',
                        help='location of the data corpus')
    parser.add_argument('--dataset', type=str, default='text8',
                        choices=['ptb', 'wt2', 'wt103', 'lm1b', 'enwik8', 'text8', 'wt103-normal', 'wiki'],
                        help='dataset name')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    corpus = get_lm_corpus(args.datadir, args.dataset, use_bpe=True)
    print(f'Vocab size : {len(corpus.vocab)}')
    #tr_iter = corpus.get_iterator('train', 16, 150, 'cpu', ext_len=0)
    # Loop through all the data to force caching.
    for split in ('train', 'valid', 'test'):
        tr_iter = corpus.get_dist_iterator(split, rank=0, max_rank=1, bsz=16, bptt=150, device='cpu', ext_len=0)
        for data, target, seq_len in tr_iter:
            if args.debug:
                print(data.shape, target.shape, seq_len, len(list(tr_iter)))
                break

if __name__ == '__main__':
    main()
