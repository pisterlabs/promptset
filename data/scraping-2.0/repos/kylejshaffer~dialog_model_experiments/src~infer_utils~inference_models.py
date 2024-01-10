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
from search_utils import softmax, Beam


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

class DialogModel(object):
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
        self.min_length = 5
        self.s2s_only = True

    def _load_model(self):
        """ Wrapper method for loading model

        Returns:
            keras.models.Model -- Keras model ready for inference
        """
        s2s_only = False
        model = load_model(self.model_path, custom_objects={'tf': tf, 'sparse_loss': sparse_loss, 'perplexity': perplexity})
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

    def _get_next_words(self, x_input, context):
        """ Internal helper method for beam search - takes in input
        and generates actual probabilities from trained model.
        
        Arguments:
            x_input {[type]} -- [description]
            context {[type]} -- [description]
        
        Returns:
            [type] -- [description]
        """
        # Process word context
        if len(context) < self.max_len:
            diff = self.max_len - len(context)
            padding = [0] * diff
            new_context = context + padding
        else:
            new_context = context[:]
        y_input = np.asarray(new_context)
        y_input = np.expand_dims(y_input, 0)
        # print(y_input.shape)
        # print(y_input)
        logits = self.model.predict_on_batch([x_input, y_input])
        
        # Return list of score-word tuples sorted by score
        row = len(context) - 1
        # print('row:', row)
        # print('logits shape:', logits.squeeze().shape)
        next_word_log_probs = logits.squeeze()[row, :]
        # print(next_word_log_probs.shape)
        next_word_probs = softmax(next_word_log_probs)
        # print(next_word_probs.shape)
        score_word_result = list(zip(next_word_probs, list(range(len(next_word_probs)))))
        
        return score_word_result

    def beam_search_decode(self, input_seq:str, beam_width:int=10, return_beams:bool=False):
        """ Beam search decoding method - optionally returns beams and scores 
        for each beam when traversing the logits matrix
        
        Arguments:
            input_seq {str} -- [Input sentence as string]
        
        Keyword Arguments:
            beam_width {int} -- [Number of beams/sequences to use in search (higher can get better responses, but takes longer)] (default: {10})
            return_beams {bool} -- [Whether to return multiple responses or only the best response] (default: {False})
        
        Returns:
            [tuple] -- [Tuple of (float, list) and optionally Beam object if returning multiple responses]
            
        """
        x_input = self._encode_from_text(input_seq)
        
        prev_beam = Beam(beam_width)
        prev_beam.add(1.0, False, [self.start_tok_id])
        while True:
            curr_beam = Beam(beam_width)
            
            #Add complete sentences that do not yet have the best probability to the current beam, the rest prepare to add more words to them.
            for (prefix_prob, complete, prefix) in prev_beam:
                if complete == True:
                    curr_beam.add(prefix_prob, True, prefix)
                else:
                    #Get probability of each possible next word for the incomplete prefix.
                    for (next_prob, next_word) in self._get_next_words(context=prefix, x_input=x_input):
                        if next_word == self.stop_tok_id: #if next word is the end token then mark prefix as complete and leave out the end token
                            curr_beam.add(prefix_prob*next_prob, True, prefix)
                        else: #if next word is a non-end token then mark prefix as incomplete
                            curr_beam.add(prefix_prob*next_prob, False, prefix+[next_word])
            
            (best_prob, best_complete, best_prefix) = max(curr_beam)
            if best_complete == True or len(best_prefix)-1 == self.max_len: #if most probable prefix is a complete sentence or has a length that exceeds the clip length (ignoring the start token) then return it
                return (best_prefix[1:], best_prob, curr_beam) #return best sentence without the start token and together with its probability
                
            prev_beam = curr_beam


# Run it
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=False, default='')
    parser.add_argument('--beam_width', type=int, required=False, default=1)
    parser.add_argument('--gpu', type=int, required=False, default=-1)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    bpe_tokenizer = load_gpt_tokenizer()
    print('Tokenizer loaded...')

    inference_model = DialogModel(tokenizer=bpe_tokenizer, model_path=args.model_path)

    example_sent = "your account may be compromised." # "Hey, how are you doing?"
    # response = inference_model.greedy_decode(input_seq=example_sent)
    # print("got response: {}".format(response))

    response = inference_model.beam_search_decode(input_seq=example_sent, beam_width=args.beam_width, return_beams=True)
    beams = response[-1]
    print('RANKED RESPONSES:')
    for b in sorted(beams):
        print(inference_model.tokenizer.decode(b[-1]), ':', b[0])
