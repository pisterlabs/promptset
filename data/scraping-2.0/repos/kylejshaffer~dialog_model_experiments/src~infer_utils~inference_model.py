import argparse
import os

import numpy as np
import tensorflow as tf

import keras.backend as K
from keras.models import Model, load_model
from keras.layers import Add, Lambda
from keras.preprocessing.sequence import pad_sequences

from keras_transformer.transformer import LayerNormalization
from keras_transformer.attention import MultiHeadAttention
from keras_transformer.extras import ReusableEmbedding, TiedOutputEmbedding
from keras_transformer.position import TransformerCoordinateEmbedding

from pytorch_pretrained_bert import OpenAIGPTTokenizer


def sparse_loss(y_true, y_pred, from_logits=True):
    """ Custom loss function - needed for loading and compiling model
    
    Arguments:
        y_true {numpy.ndarray} -- Numpy array containing ground-truth word-labels
        y_pred {numpy.ndarray} -- Numpy array containing un-normalized logit weights for words
    
    Keyword Arguments:
        from_logits {bool} -- [description] (default: {True})
    
    Returns:
        crossentropy loss -- Float corresponding to the loss on either training or test set, lower is better
    """
    return K.sparse_categorical_crossentropy(y_true, y_pred, from_logits=from_logits)

def perplexity(y_true, y_pred):
    """ Custom metric - needed for loading and compiling model
    
    Arguments:
        y_true {numpy.ndarray} -- Numpy array containing ground-truth word-labels
        y_pred {numpy.ndarray} -- Numpy array containing un-normalized logit weights for words
    
    Returns:
        perpelxity score -- Float corresponding to perplexity score for model, lower is better (e.g. can
        be interpreted as answering the question "how many words is the model confused between at any
        given timestpe in a sequence?")
    """
    cross_entropy = K.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)
    return K.mean(K.exp(K.mean(cross_entropy, axis=-1)))

def load_gpt_tokenizer():
    """ Helper function for loading sub-word tokenizer
    
    Returns:
         Instance of pytorch_pretrained_bert.OpenAIGPTTokenizer tokenizer
    """
    model_name = 'openai-gpt'
    special_tokens = ['_start_', '_end_', '_pad_']
    tok = OpenAIGPTTokenizer.from_pretrained(model_name, special_tokens=special_tokens)
    # Explicitly set padding token to be word_id = 0
    tok.special_tokens['_pad_'] = 0
    tok.encoder['<unk>'] = len(tok)
    print('GPT tokenizer initialized...')

    return tok

class InferenceModel(object):
    def __init__(self, tokenizer, model_path:str):
        """ Inference wrapper class for getting predictions from model
        and decoding into text to be preented to end-user
        
        Arguments:
            tokenizer -- Instance of pytorch_pretrained_bert.OpenAIGPTTokenizer
            model_path {str} -- Path to trained model
        """
        self.tokenizer = tokenizer
        self.model_path = model_path
        self.model = self._load_model()
        self.start_tok_id = self.tokenizer.special_tokens['_start_']
        self.stop_tok_id = self.tokenizer.special_tokens['_end_']
        self.max_len = 45
        self.s2s_only = True

    def _load_model(self):
        """ Wrapper method for loading model
        
        Returns:
            keras.models.Model -- Keras model ready for inference
        """
        s2s_only = False
        model = load_model(self.model_path, custom_objects={'tf': tf,
                                                                  'sparse_loss': sparse_loss,
                                                                  'perplexity': perplexity})
        model.summary()
        
        return model

    def _encode_from_text(self, input_text:str):
        """ Internal method for prepping input text to be fed to model
        
        Arguments:
            input_text {str} -- String of utterance that will have a response generated
            for it
        
        Returns:
            numpy.ndarray -- Numpy array containing word ID's needed for model prediction
        """
        tok_ids = self.tokenizer.encode(input_text)
        tok_ids.insert(0, self.start_tok_id)

        if len(tok_ids) > self.max_len:
            tok_ids = tok_ids[:(self.max_len - 1)]
        tok_ids.append(self.stop_tok_id)
        x_input = np.asarray(tok_ids)
        x_input = np.expand_dims(x_input, 0)

        return x_input
    
    def greedy_decode(self, input_seq:str):
        """ Greedy decoding method for generating sequence of outputs
        
        Arguments:
            input_seq {str} -- Input utterance that we'd like to return predictions for
        
        Returns:
            decoded_string {str} -- String of response utterance generated from model
        """
        x_input = self._encode_from_text(input_seq)

        # Set up decoder input data
        decoded_tokens = []
        target_seq = np.zeros((1, self.max_len), dtype='int32')
        target_seq[0, 0] = self.start_tok_id

        # Loop through and generate decoder tokens
        print('Generating output...')
        for i in range(self.max_len - 1):
            print('=', end='', flush=True)
            output = self.model.predict_on_batch([x_input, target_seq]).argmax(axis=2)
            # sampled_index = np.argmax(output[0, i, :])
            sampled_index = int(output[:, i])
            if sampled_index == self.stop_tok_id:
                break
            decoded_tokens.append(sampled_index)
            target_seq[0, i+1] = sampled_index
        
        decoded_string = self.tokenizer.decode(decoded_tokens)
        print()
        print(decoded_string)

        return decoded_string

    def beam_search_decode(self, input_seq:str):
        # TBD for this method
        raise NotImplementedError()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=False, default='/data/users/kyle.shaffer/chat_models/joint_model_gpt_lm_flat04_3.88.h5')
    parser.add_argument('--gpu', type=int, required=False, default=-1)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    bpe_tokenizer = load_gpt_tokenizer()
    print('Tokenizer loaded...')

    inference_model = InferenceModel(tokenizer=bpe_tokenizer, model_path=args.model_path)

    example_sent = "Hey, how are you doing?"
    response = inference_model.greedy_decode(input_seq=example_sent)

    print("SUCCESSFULLY EXECUTED!")
