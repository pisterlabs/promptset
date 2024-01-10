# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# from pytorch_pretrained_bert import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer
import numpy as np
from LAMA.lama.modules.base_connector import *


class GPT(Base_Connector):

    def __init__(self, args, tokenizer, model, _is_self_debias=False, bias_type=None):
        '''args is not used'''
        super().__init__()

        # if args.gpt_model_dir is not None:
        #     # load bert model from file
        #     gpt_model_name = str(args.gpt_model_dir) + "/"
        #     dict_file = gpt_model_name
        #     print("loading Open AI GPT model from {}".format(gpt_model_name))
        # else:
        #     # load GPT model from huggingface cache
        #     gpt_model_name = args.gpt_model_name
        #     dict_file = gpt_model_name

        # Load pre-trained model tokenizer (vocabulary)
        # self.tokenizer = OpenAIGPTTokenizer.from_pretrained(dict_file)
        self.tokenizer = tokenizer # modification-1

        # GPT uses different way to represent BPE then BERT. Namely, the
        # final suffixes are indicated with </w> suffix, while pieces that must
        # be followed are written as is. In BERT the prefixes are written as is
        # while the parts that must follow (not be followed!) have '##' prefix.
        # There is no one-to-one coversion. But at least we may make pieces that
        # may form a full word look the same.
        # Note that we should be very careful now,
        # tokenizer.convert_tokens_to_ids won't work with our vocabulary.
        def convert_word_for_openai_gpt(word):
            if word == OPENAI_UNK:
                return word
            if word == '\n</w>':
                # Redefine symbol EOS to improve visualization.
                return OPENAI_EOS
            return word[:-4] if word.endswith('</w>') else f'{word}##'

        def convert_word(word):
            if word == OPENAI_UNK:
                return word
            if word == '\n</w>':
                # Redefine symbol EOS to improve visualization.
                return OPENAI_EOS
            return word[1:] if word.startswith(OPENAI_SPACE) else f'##{word}'

        _, gpt_vocab = zip(*sorted(self.tokenizer.decoder.items())) 
        # tokenizer.decoder is a dict{int:str}; many tokens start with ord(decoder[50255][0])=288,
        # which should represent a space
        # self.vocab = [convert_word(word) for word in gpt_vocab]
        self.vocab = gpt_vocab
        self._init_inverse_vocab()

        # Get UNK symbol as it's written in the origin GPT vocab.
        unk_index = self.inverse_vocab[OPENAI_UNK]
        self.unk_symbol = self.tokenizer.decoder[unk_index]

        # Load pre-trained model (weights)
        # self.gpt_model = OpenAIGPTLMHeadModel.from_pretrained(gpt_model_name)
        self.gpt_model = model # modification-2
        # self.gpt_model.eval()
        if _is_self_debias:
            self.gpt_model._model.config
        else:
            print(self.gpt_model.config)

        # Sanity check.
        assert len(self.vocab) == (self.gpt_model.config.vocab_size if hasattr(self.gpt_model,'config') else self.gpt_model._model.config.vocab_size)
        # assert 0 == (self.gpt_model.config.n_special if self.gpt_model.config is not None else self.gpt_model._model.config.n_special)

        self.eos_id = self.inverse_vocab[OPENAI_EOS]
        self.model_vocab = self.vocab

        self._is_self_debias = _is_self_debias
        self._bias_type = bias_type

    def _cuda(self):
        if not self._is_self_debias:
            self.gpt_model.cuda()
        else:
            self.gpt_model._model.cuda()

    def get_id(self, string):
        tokenized_text = self.tokenizer.tokenize(' '+string)
        indexed_string = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        # indexed_string = self.convert_ids(indexed_string)
        return indexed_string

    def __get_input_tensors(self, sentence_list):
        """Concatenates, tokenize and converts a sentences to model inputs.

        Args:
            sentence_list: A list of strings. The string may contain a special
            [MASK] token.

        Returns:
            A tuple (src_tensor, dst_tensor, masked_indices, tokenized_text).
                src_tensor: torch.LongTensor with shape (seq_len), the input to
                    the new without the last symbol and with EOS prepended.
                dst_tensor: torch.LongTensor with shape (seq_len).
                masked_indices: A list of indices of [MASK] in dst_tensor.
                tokenized_text: A list of token string.
            """
        # Split the sentence by [MASK] and tokenize the chunks independently.
        tokenized_text = []
        masked_indices = []
        for sentence_idx, sentence in enumerate(sentence_list):
            if sentence_idx > 0:
                tokenized_text.append(OPENAI_EOS)
            for chunk_idx, chunk in enumerate(sentence.split('[MASK]')):
                if chunk_idx > 0:
                    masked_indices.append(len(tokenized_text)) # calculate the masked position
                    tokenized_text.append(self.unk_symbol) # use unk_symbol as the mask token
                chunk = chunk.strip()
                if chunk:
                    tokenized_text.extend(self.tokenizer.tokenize(chunk))

        full_indexed_tokens = [
            self.eos_id
        ] + self.tokenizer.convert_tokens_to_ids(tokenized_text)
        # full_indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        full_tokens_tensor = torch.tensor(full_indexed_tokens)
        src_tensor = full_tokens_tensor[:-1]
        dst_tensor = full_tokens_tensor[1:]

        # masked_indices = [idx-1 for idx in masked_indices]
        return src_tensor, dst_tensor, masked_indices, tokenized_text

    def get_batch_generation(self, sentences_list, logger=None, try_cuda=True):
        if try_cuda:
            self.try_cuda()
        src_tensor_list, dst_tensor_list, masked_indices_list, _ = zip(*[
            self.__get_input_tensors(sentences) for sentences in sentences_list
        ])

        src_tensor_batch = torch.nn.utils.rnn.pad_sequence(
            src_tensor_list, batch_first=True)

        # The model uses shared embedding space for tokens and positions. More
        # precisely, the first len(vocab) indidices are reseved for words, the
        # last n_special symbols are reserved for special symbols and the rest
        # is used for positions. Softmax and embedding matrices are shared and
        # as result some of output "symbols" correspond to positions. To fix
        # that we have to manually remove logits for positions.
        with torch.no_grad():
            if self._is_self_debias:
                debiasing_prefixes = [DEBIASING_PREFIXES[self._bias_type]]
                (logits, input_ids,) = self.gpt_model.compute_loss_self_debiasing(
                    src_tensor_batch.to(self._model_device), debiasing_prefixes=debiasing_prefixes
                )

                # Lengths of prompts:
                # 13 for gender
                # 15 for race
                # 13 for religion
                bias_type_to_position = {"gender": 13, "race-color": 15, "religion": 13}

                # Don't include the prompt.
                logits = logits[:, bias_type_to_position[self._bias_type] :, :]
            else:
                logits = self.gpt_model(src_tensor_batch.to(self._model_device))[0] # (bsz,seq_len-1,vocab_size)
                # logits = logits[..., :self.gpt_model.config.vocab_size] # (bsz,seq_len,vocab_size)

            log_probs = torch.nn.functional.log_softmax(logits, dim=-1).cpu() # (bsz,seq_len-1,vocab_size)

        token_ids_list = [
            np.array(dst_tensor.numpy()) for dst_tensor in dst_tensor_list
        ]

        return log_probs, token_ids_list, masked_indices_list

    def get_contextual_embeddings(self, sentences_list, try_cuda=True):

        if try_cuda:
            self.try_cuda()

        src_tensor_list, dst_tensor_list, masked_indices_list, _ = zip(*[
            self.__get_input_tensors(sentences) for sentences in sentences_list
        ])

        src_tensor_batch = torch.nn.utils.rnn.pad_sequence(
            src_tensor_list, batch_first=True)

        with torch.no_grad():
            output = self.gpt_model.transformer(src_tensor_batch.to(self._model_device))

        # TODO
        sentence_lengths = None
        tokenized_text_list = None

        # As we only return the last layer, [] to have the same format as other models
        return [output], sentence_lengths, tokenized_text_list
