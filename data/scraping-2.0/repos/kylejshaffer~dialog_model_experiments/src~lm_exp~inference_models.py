import argparse
import json
import os
import sys

sys.path.append('/home/kyle.shaffer/cakechat')
from cakechat.utils.dataset_loader import load_conditioned_dataset
from cakechat.dialog_model.model_utils import get_training_batch

import numpy as np

import keras.backend as K
from keras.models import Model, load_model
from keras.layers import Add, Lambda
from keras.preprocessing.sequence import pad_sequences


class FuseModel(object):
    def __init__(self, s2s_path:str, lm_path:str):
        self.s2s_model = self._load_model(s2s_path)
        self.lm_model = self._load_model(lm_path)
        # self.fuse_model = self._build_fuse_model(s2s_model=self.s2s_model, lm_model=self.lm_model)

    def _train_loss(self, y_true, y_pred, from_logits=True):
        return K.sparse_categorical_crossentropy(y_true, y_pred, from_logits)

    def _load_model(self, model_path:str):
        if 'lm' in model_path:
            loss_name = '_loss'
        else:
            loss_name = '_train_loss'

        model = load_model(model_path, custom_objects={loss_name: self._train_loss})
        print('Model loaded from {}...'.format(model_path))
        return model

    def _build_fuse_model(self, s2s_model, lm_model, lm_coef:float=1.0):
        new_lm_output = lm_model(s2s_model.inputs[1])
        lm_weighted_logits = Lambda(lambda x: lm_coef * x)(new_lm_output)
        add_logits_layer = Add()([s2s_model.output, lm_weighted_logits])
        fused_graph = Model(inputs=s2s_model.inputs, outputs=add_logits_layer)
        fused_graph.summary()
        
        return fused_graph

    def _get_batch_generator(self, input_data, batch_size):
        RANDOM_SEED = 7
        DECODER_DEPTH = 2
        UTT_HIDDEN_DIM = 600
        epoch_id = 0

        while True:  # inifinite batches generator
            epoch_id += 1

            for train_batch in get_training_batch(
                    input_data,
                    batch_size,
                    random_permute=False,
                    random_seed=RANDOM_SEED * epoch_id):
                
                context_tokens_ids, response_tokens_ids = train_batch
                
                # response tokens are wraped with _start_ and _end_ tokens
                # output shape == (batch_size, seq_len)

                # get input response ids by removing last sequence token (_end_)
                input_response_tokens_ids = response_tokens_ids[:, :-1]
                # output shape == (batch_size, seq_len - 1)

                # get target response ids by removing the first (_start_) token of the sequence
                target_response_tokens_ids = response_tokens_ids[:, 1:]
                # output shape == (batch_size, seq_len - 1)

                # workaround for using sparse_categorical_crossentropy loss
                # see https://github.com/tensorflow/tensorflow/issues/17150#issuecomment-399776510
                target_response_tokens_ids = np.expand_dims(target_response_tokens_ids, axis=-1)
                # output shape == (batch_size, seq_len - 1, 1)

                init_dec_hs = np.zeros(
                    shape=(context_tokens_ids.shape[0], DECODER_DEPTH, UTT_HIDDEN_DIM),
                    dtype=K.floatx())

                yield [context_tokens_ids, input_response_tokens_ids, init_dec_hs], target_response_tokens_ids

    def eval_loss(self, valid_data, batch_size=256, mode='s2s'):
        # S2S Loss = 4.036
        # LM Loss = 4.79

        total_loss = 0
        steps = 10

        datagen = self._get_batch_generator(input_data=(valid_data.x, valid_data.y), batch_size=batch_size)

        if mode == 's2s':
            print("EVALUATING S2S")
            for i in range(steps):
                x, y = next(datagen)
                logits = self.s2s_model.predict_on_batch(x)
                loss = K.eval(K.sparse_categorical_crossentropy(K.variable(y), K.variable(logits), from_logits=True))
                batch_loss = 0

                for row in loss:
                    batch_loss += np.mean(row.squeeze())

                total_loss += (batch_loss / row.shape[0])
                
            print('Loss:', total_loss / steps)
        else:
            print("EVALUATING LM")
            for i in range(steps):
                sys.stdout.write('\r {}...'.format(i))
                x, y = next(datagen)
                x_in = x[1]
                # logits = self.lm_model.predict_on_batch(x_in)
                # print(logits.shape)
                # loss = K.eval(K.sparse_categorical_crossentropy(K.variable(y), K.variable(logits), from_logits=True))
                # batch_loss = 0

                # for row in loss:
                #     batch_loss += np.mean(row.squeeze())

                # total_loss += (batch_loss / row.shape[0])
                batch_loss = self.lm_model.test_on_batch(x_in, y)
                total_loss += batch_loss

            print('\nLoss:', total_loss / steps)

    def combo_eval(self, lm_coef:float, valid_data, batch_size:int):
        combo_graph = self._build_fuse_model(s2s_model=self.s2s_model, lm_model=self.lm_model, lm_coef=lm_coef)
        combo_graph.compile(loss=self._train_loss, optimizer='sgd')
        n_valid_iters = (valid_data.x.shape[0] // batch_size) + 1
        print('No. validation iters:', n_valid_iters)

        datagen = self._get_batch_generator(input_data=(valid_data.x, valid_data.y), batch_size=batch_size)

        total_loss_combined = 0
        total_loss_s2s = 0
        for i in range(n_valid_iters):
            sys.stdout.write('\rEvaluating batch {}...'.format(i))
            x, y = next(datagen)
            batch_loss_combined = combo_graph.test_on_batch(x, y)
            batch_loss_s2s = self.s2s_model.test_on_batch(x, y)
            # Update loss trackers
            total_loss_combined += batch_loss_combined
            total_loss_s2s += batch_loss_s2s

        print()
        return {'lm_coef': lm_coef, 'combined_loss': (total_loss_combined / n_valid_iters),
                's2s_loss': (total_loss_s2s / n_valid_iters)}

    def debug_lm(self, valid_data, batch_size=256):
        total_loss = 0
        return


def run_combo_eval(fuse_model, valid_data:tuple, batch_size:int, result_path:str, lm_coefs:list):
    print('Running LM coef eval....')
    result_container = []
    for coef in lm_coefs:
        print('Evaluating with coefficient:', coef)
        result = fuse_model.combo_eval(lm_coef=coef, valid_data=valid_data, batch_size=batch_size)
        result_container.append(result)

    with open(result_path, mode='w') as outfile:
        json.dump(result_container, outfile)

    print('Results saved!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, required=False, default=256)
    parser.add_argument('--gpu', type=int, required=False, default=0)
    args = parser.parse_args()
    
    # from lm import load_cakechat_data_with_tok
    # from pytorch_pretrained_bert import OpenAIGPTTokenizer

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    max_len = 45

    model_dir = '/data/users/kyle.shaffer/chat_models'
    # Hack code for loading necessary data for 3rd party functions...
    vocab_path = '/data/users/kyle.shaffer/dialog_data/cornell_movie/cakechat_model/tokens_index/t_idx_processed_dialogs.json'
    conditions_path = '/data/users/kyle.shaffer/dialog_data/cornell_movie/cakechat_model/conditions_index/conditions_index.json'

    with open(vocab_path, mode='r') as infile:
        token_to_index = json.load(infile)
        index_to_token = {int(v): k for k, v in token_to_index.items()}

    with open(conditions_path, mode='r') as infile:
        index_to_condition = json.load(infile)
        index_to_condition = {int(k): v for k, v in index_to_condition.items()}
        print(index_to_condition)
        
    condition_to_index = {v: k for k, v in index_to_condition.items()}

    valid_name = 'valid_no_tok'
    valid_data = load_conditioned_dataset(valid_name, token_to_index, condition_to_index, use_gpt_tok=True)
    # model_name = 'openai-gpt'
    # special_tokens = ['_start_', '_delimiter_', '_classify_']
    # gpt_tok = OpenAIGPTTokenizer.from_pretrained(model_name, special_tokens=special_tokens)
    # print('GPT tokenizer initialized...')

    # valid_path = '/data/users/kyle.shaffer/dialog_data/cornell_movie/cakechat_model/corpora_processed/valid_no_tok.txt'
    # valid_lines = load_cakechat_data_with_tok(valid_path, gpt_tok, max_len)
    # x_valid_lines = [i[:-1] for i in valid_lines]
    # y_valid_lines = [i[1:] for i in valid_lines]
    # del valid_lines
    # x_valid, y_valid = pad_sequences(x_valid_lines, padding='post', maxlen=max_len), pad_sequences(y_valid_lines, padding='post', maxlen=max_len)

    s2s_path = os.path.join(model_dir, 'hierarch_cakechat_50_4.00.h5')
    lm_path = os.path.join(model_dir, 'movie_lm_07_4.13.h5')

    fuse_model = FuseModel(s2s_path, lm_path)
    # fuse_model.eval_loss(valid_data, batch_size=args.batch_size, mode='lm')
    # print('VALIDATION METRICS:')
    # print('Valid loss:', valid_loss)
    # print('Valid perplexity:', valid_ppl)

    # loss = fuse_model.lm_model.evaluate(x_valid, y_valid, batch_size=128, verbose=1)
    # print(loss)

    # RUN COEF SEARCH EVAL
    lm_coefs = [np.round(i, 2) for i in np.arange(0.1, 1.1, 0.1)]
    run_combo_eval(fuse_model, valid_data=valid_data, batch_size=256, result_path='results/combo_eval.json', lm_coefs=lm_coefs)
