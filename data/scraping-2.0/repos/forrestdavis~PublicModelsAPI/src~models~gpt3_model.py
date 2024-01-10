import torch
import sys
import os
import time
import string

import tiktoken
import openai

from .RTModel import RTModel

openai.api_key = os.getenv("OPENAI_API_KEY")

class GPT3Model(RTModel):
    def __init__(self, version, useMPS=True):

        super().__init__(model_name=version, 
                         use_prefix_space=True)

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.has_mps and useMPS:
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self._tokenizer = GPT3Tokenizer(version)
        self._version = version

    @property
    def tokenizer(self):
        return self._tokenizer

    def token_is_sep(self, token):
        return token == self.tokenizer.pad_token_id

    def token_is_unk(self, token):
        return token == self.tokenizer.unk_token_id

    def token_is_punct(self, token):
        word = self.tokenizer.convert_ids_to_tokens(token)[0][0]
        if type(word) != str:
            word = word.decode('utf-8')
        return word in string.punctuation

    def word_to_idx(self, text, isFirstWord=False, isLastWord=False):
        if self.usePrefixSpace:
            if isFirstWord:
                indices = self.tokenizer.encode(text)
            else:
                indices = self.tokenizer.encode(' '+text)
        else:
            indices = self.tokenizer.encode(' '+text)

        if type(indices[0]) == list:
            return indices[0]
        return indices

    def word_in_vocab(self, text, isFirstWord=False, isLastWord=False):

        indices = self.word_to_idx(text, isFirstWord, isLastWord)
        if len(indices) > 1 or self.token_is_unk(indices[0]):
            return False
        return True

    def get_response(self, prompts):

        response = openai.Completion.create(
            model=self._version, 
            prompt = prompts, 
            max_tokens = 0, 
            echo=True, 
            logprobs=1, 
            temperature=0)

        return response.to_dict_recursive()

    @torch.no_grad()
    def get_by_token_surprisals(self, text): 
        """Returns surprisal of each token for inputted text.
           Note that this requires that you've implemented
           a tokenizer and get_output
           for the model instance.

        Args: 
            text (List[str] | str ): A batch of strings or a string.

        Returns:
            lists (token, surp): Lists of (token id, surprisal) that are 
            batch_size X len(tokenized text). 
            Meaning that the padding from get_surprisals is removed.
        """
        batchSize=100
        sleepInterval = 30

        if type(text) == str:
            text = [text]

        results = []
        for idx in range(0, len(text), batchSize):
            if idx > 0:
                # pause inbetween batches
                time.sleep(sleepInterval)
            batch = text[idx:idx+batchSize]
            responses = self.get_response(batch)
            #Extract tokens and surps
            for output in responses['choices']:
                logprobs = output['logprobs']['token_logprobs']
                tokens = output['logprobs']['tokens']

                assert len(logprobs) == len(tokens)

                #First word is Null
                logprobs[0] = 0.0

                ids = self.tokenizer.convert_tokens_to_ids(tokens)
                #flatten
                ids = [i for l in ids for i in l]

                assert len(ids) == len(tokens), tokens

                surps = -(torch.tensor(logprobs)/torch.log(torch.tensor(2.0)))
                surps[0] = 0
                surps = surps.tolist()

                by_token = list(zip(ids, surps))
                results.append(by_token)
        return results

    def get_targeted_word_probabilities(self, context, target):

        if type(context) == str:
            context = [context]
        if type(target) == str:
            target = [target]

        assert len(context) == len(target)

        sents = list(zip(context, target))
        sents = list(map(lambda x: ' '.join(x), sents))

        surprisals = self.get_by_token_surprisals(sents)

        output = []
        for surps in surprisals:
            output.append(2**-(surps[-1][-1]))
        return output

    def get_targeted_word_surprisals(self, context, target):
        if type(context) == str:
            context = [context]
        if type(target) == str:
            target = [target]

        assert len(context) == len(target)

        sents = list(zip(context, target))
        sents = list(map(lambda x: ' '.join(x), sents))

        surprisals = self.get_by_token_surprisals(sents)

        output = []
        for surps in surprisals:
            output.append(surps[-1][-1])
        return output

    @torch.no_grad()
    def get_by_sentence_perplexity(self, text):
        """Returns perplexity of each sentence for inputted text.

        Args: 
            text (List[str] | str ): A batch of strings or a string.

        Returns:
            lists (sent, ppl): List of the perplexity of each string in the
            batch. Padding is ignored in the calculation. 
        """

        surprisals = self.get_by_token_surprisals(text)
        ppls = []
        for surprisal in surprisals:
            #Ignore first word 
            surprisal.pop(0)

            surps = list(map(lambda x: x[1], surprisal))
            surps = torch.tensor(surps)
            log_avg = torch.sum(surps, dim=0)/surps.shape[0]
            ppl = float(torch.exp2(log_avg))
            ppls.append(ppl)

        assert len(ppls) == len(text)

        return list(zip(text, ppls))

class GPT3Tokenizer:

    def __init__(self, version):
        self._enc_base = tiktoken.encoding_for_model(version)
        self._enc = tiktoken.Encoding(
                name=version+'_with_pad',
                pat_str=self._enc_base._pat_str,
                mergeable_ranks=self._enc_base._mergeable_ranks,
                special_tokens={
                    **self._enc_base._special_tokens,
                    " <|pad|>": 100264,
                }
            )
        self._vocab = self._enc._mergeable_ranks.copy()
        self._vocab.update(self._enc._special_tokens)

    @property
    def vocab(self):
        return self._vocab

    @property
    def vocab_size(self):
        return len(self.vocab)

    @property
    def idx2word(self):
        return dict((v, k) for k, v in self.vocab.items())

    @property 
    def eos_token(self):
        return '<|endoftext|>'

    @property 
    def pad_token(self):
        """NOT USED BY TOKENIZER"""
        return " <|pad|>"

    @property
    def pad_token_id(self):
        return self.vocab[self.pad_token]

    @property
    def eos_token_id(self):
        if self.eos_token in self.vocab:
            return self.vocab[self.eos_token]
        else:
            return None

    @property
    def unk_token(self):
        return None

    @property
    def unk_token_id(self):
        return None

    def __len__(self):
        return self.vocab_size

    def __call__(self, text, return_tensors=None):

        encodings = self.encode(text)
        encodings = self.batchify(encodings)

        if return_tensors=='pt':
            return {'input_ids': 
                         torch.tensor(encodings['input_ids'], 
                                      dtype=torch.int64), 
                         'attention_mask':
                         torch.tensor(encodings['attention_mask'], 
                                      dtype=torch.int64)}
        elif return_tensors is None:
            return encodings
        else:
            sys.stderr.write('I have not implemented a return_tensors type: '+str(return_tensors)+'\n')
            sys.exit(1)

    def encode(self, text, lower=False,
            remove_trailing_spaces=False):
        """ Returns a list of encoded text"""

        if type(text) == str:
            text = [text]

        if lower or remove_trailing_spaces:
            for idx, line in enumerate(text):
                if lower:
                    text[idx] = line.lower()
                if remove_trailing_spaces:
                    text[idx] = line.strip()

        return self._enc.encode_batch(text, allowed_special="all")

    def batchify(self, encodings):

        assert self.pad_token_id is not None, 'Attempting to PAD with no token'
        max_seq_len = max(len(encoding) for encoding in encodings)
        padded_batch_outputs = {'input_ids': [], 'attention_mask': []}

        for encoding in encodings:
            difference = max_seq_len - len(encoding)
            input_ids = encoding + [self.pad_token_id]*difference
            attn_ids = [1]*len(encoding) + [0]*difference

            padded_batch_outputs['input_ids'].append(input_ids)
            padded_batch_outputs['attention_mask'].append(attn_ids)

        return padded_batch_outputs

    def decode(self, input_dict, convertByte=True):
        if type(input_dict) == dict:
            input_ids = input_dict['input_ids']
            attn_mask = input_dict['attention_mask']
        else:
            input_ids = input_dict

        if type(input_ids) != list:
            input_ids = input_ids.tolist()

        if type(input_ids[0]) != list:
            input_ids = [input_ids]

        decoded = []
        for encoding in input_ids:
            dec = self._enc.decode_tokens_bytes(encoding)
            if convertByte:
                dec = list(map(lambda x: x.decode('utf-8'), dec))
            decoded.append(dec)
        return decoded

    def convert_ids_to_tokens(self, ids):
        if type(ids) != list:
            ids = [ids]
        return self.decode(ids)

    def convert_tokens_to_ids(self, tokens):
        return self.encode(tokens)
