"""Defines classes for each encoder: GloVe, BERT, InferSent, 
Vanilla and GPT language model along with related code"""

import torch
import torch.nn as nn
import numpy as np
import math

import config

from pymagnitude import Magnitude
try:
    from pytorch_transformers import BertTokenizer, BertModel
    from pytorch_transformers import OpenAIGPTTokenizer, OpenAIGPTLMHeadModel
    from pytorch_transformers import GPT2Tokenizer, GPT2LMHeadModel
    
    from sentence_transformers import SentenceTransformer
except:
    print('Failed to import BERT, GPT, or GPT2')

try:
    from InferSentModels import InferSent
except:
    print('Failed to import InferSent')

# Set device and pretrained models list
DEVICE = config.DEVICE
GPU_ENABLED = config.GPU_ENABLED

pretrained_models_list = ['GloveEncoder', 'InferSentEncoder', 'BERTEncoder', 'InitializedEncoder']

class EncoderRNN(nn.Module):
    """Encoder class which trains embeddings from scratch and specifies GRU architecture"""
    def __init__(self, input_size, embedding_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.name = 'RandomEncoderRNN'
        self.trainable_model = True
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, self.embedding_size)
        self.gru = nn.GRU(self.embedding_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=DEVICE)

def create_vocab_tensors(input_vocab_index):
    """Creates a matrix of the glove embeddings for terms contained in the model for improve runtime
        Also used in ESIM"""
    print('Creating vocabulary tensors...')
    # Define GloVe model from Magnitude package
    model = Magnitude(config.glove_magnitude_path)
    
    np.random.seed(config.SEED)
    # Randomly initialize matrix
    vocab_tensors = np.random.normal(0, 1, (input_vocab_index.n_words, model.dim)).astype('float32')
    
    vocab_words = list(input_vocab_index.word2index.keys())
    unk_words = []
    
    # Get vector for each word in vocabulary if in model
    for idx, word in enumerate(vocab_words):
        if word in model:
            vocab_tensors[idx] = model.query(word)
        else:
            unk_words.append(word)
            
    # Override special tokens
    special_tokens = ['SOS', 'EOS', 'UNK']
    
    # Override special tokens
    vocab_tensors[:len(special_tokens), :] = np.random.uniform(
            -0.1, 0.1, (len(special_tokens), model.dim)).astype('float32')
    
    print('Tensor vocabulary complete.')
    print('    Total vocabulary size {}, {} UNK words ({:.2}%)'.format(len(vocab_words), len(unk_words),
    (len(unk_words) / len(vocab_words)) *100))
    return torch.tensor(vocab_tensors, dtype=torch.float64), unk_words

class InitializedEncoderRNN(nn.Module):
    """Encoder class which trains embeddings from scratch and specifies GRU architecture"""
    def __init__(self, input_size, embedding_size, hidden_size, caption_vocab_index, freeze_weights):
        super(InitializedEncoderRNN, self).__init__()
        self.name = 'InitializedEncoderRNN'
        self.trainable_model = True
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.freeze_weights = freeze_weights
        
        self.embedding = nn.Embedding(input_size, self.embedding_size)
        self.embedding.weight = nn.Parameter(create_vocab_tensors(caption_vocab_index)[0])
        if freeze_weights == True:
            self.embedding.weight.requires_grad = False
        self.gru = nn.GRU(self.embedding_size, self.hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden
    
    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=DEVICE)

class GloveEncoder():
    """Encodes an input sentence as a mean or max pooled sentence embedding given the individual word embeddings"""
    def __init__(self, pooling='mean'):
        self.name = 'GloveEncoder'
        self.trainable_model = False
        self.pooling = pooling
        self.model = Magnitude(config.glove_magnitude_path)
        self.hidden_size = self.model.dim

    def sentence_embedding(self, input_text):
        words_in_model = [word for word in input_text.split() if word in self.model]
        sentence_embedding = np.zeros((len(words_in_model), self.model.dim))
        sentence_embedding.fill(np.nan)
        
        for idx, token in enumerate(words_in_model):
            sentence_embedding[idx] = self.model.query(token)
             
        if self.pooling == 'max':
            sentence_embedding = np.max(sentence_embedding, axis=0)
            
        else:
            sentence_embedding = np.mean(sentence_embedding, axis=0)
        
        return torch.tensor(sentence_embedding.reshape(1, 1, -1), device=DEVICE)
    
def load_InferSent_model(vocab_size=250000, enable_GPU=True):
    """   Loads the pretrained InferSent model
    Based on https://github.com/facebookresearch/InferSent, 
    note: needed to download file manually from: https://dl.fbaipublicfiles.com/senteval/infersent/infersent1.pkl
    and also make changes to data and model files per: https://github.com/facebookresearch/InferSent/issues/98 """
    
    model_version = 1 # Uses Glove Embeddings
    MODEL_PATH = config.infersent_model_path
    params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                    'pool_type': 'max', 'dpout_model': 0.0, 'version': model_version}
    model = InferSent(params_model)
    model.load_state_dict(torch.load(MODEL_PATH))
    
    # Put model on GPU   
    if enable_GPU:
        model = model.cuda()
    else:
        model

    model.set_w2v_path(config.glove_txt_path)

    # Load embeddings of K most frequent words
    model.build_vocab_k_words(K=vocab_size)
    return model

class InferSentEncoder():
    """Class designed for converting an input sentence to an embedding using InferSent"""
    def __init__(self):
        self.name = 'InferSentEncoder'
        self.trainable_model = False
        self.model = load_InferSent_model(vocab_size=250000, enable_GPU=GPU_ENABLED)
        self.hidden_size = 4096
    
    def sentence_embedding(self, input_text):
        embedded_sentence = self.model.encode([input_text], bsize=128, tokenize=False, verbose=False)
        return torch.tensor(embedded_sentence, device=DEVICE).view(1, 1, -1)

class BERTEncoder():
    """Class designed for converting an input sentence to an embedding using fine-tuned BERT"""
    def __init__(self):
        self.name = 'BERTEncoder'
        self.trainable_model = False
        self.model = SentenceTransformer('bert-base-nli-mean-tokens')
        self.hidden_size = 768

    def sentence_embedding(self, input_text):
        #https://huggingface.co/pytorch-transformers/model_doc/bert.html#bertmodel
        with torch.no_grad():
            encoded_sentence = self.model.encode([input_text], batch_size=1, show_progress_bar=False)[0]
            return torch.tensor(encoded_sentence, device=DEVICE).view(1,1,-1)

class GPTLanguageModel():
    """Class designed for returning the fluency score using GPT for an input sentence"""
    def __init__(self):
        self.name = 'GPTLanguageModel'
        self.trainable_model = False
        self.GPT_tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
        self.model = OpenAIGPTLMHeadModel.from_pretrained('openai-gpt').eval()
    
    def fluency_score(self, input_sentence, max_value=500):
        try:
            with torch.no_grad():
                tokenize_input = self.GPT_tokenizer.encode(input_sentence)
                input_ids = torch.tensor(tokenize_input).unsqueeze(0)
                loss=self.model(input_ids, labels=input_ids)[0]
            return 1-min((math.exp(loss)/max_value), 0.99)
        except:
            print('GPT rejected sentence: ', input_sentence)
            return 0.01

class GPT2LanguageModel():
    """Class designed for returning the fluency score using GPT2 for an input sentence"""
    def __init__(self):
        self.name = 'GPT2LanguageModel'
        self.trainable_model = False
        self.GPT2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.model = GPT2LMHeadModel.from_pretrained('gpt2').eval()   
    
    def fluency_score(self, input_sentence, max_value=500):
        try:
            with torch.no_grad():
                tokenize_input = self.GPT2_tokenizer.encode(input_sentence)
                input_ids = torch.tensor(tokenize_input).unsqueeze(0)
                loss=self.model(input_ids, labels=input_ids)[0]
            return 1-min((math.exp(loss)/max_value), 0.99)
        except:
            print('GPT rejected sentence: ', input_sentence)
            return 0.01

#%% ---------------------- ARCHIVE ---------------------

#class BERTEncoder():
#    """$$$ general BERT without finetuning"""
#    def __init__(self):
#        self.name = 'BERTEncoder'
#        self.trainable_model = False
#        self.BERT_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#        self.model = BertModel.from_pretrained('bert-base-uncased')
#        self.hidden_size = 768
#
#    def sentence_embedding(self, input_text):
#        #https://huggingface.co/pytorch-transformers/model_doc/bert.html#bertmodel
#        with torch.no_grad():
#            tokenized_text = self.BERT_tokenizer.encode(input_text)
#            input_ids = torch.tensor(tokenized_text).unsqueeze(0)  # Batch size 1
#            
#            outputs = self.model(input_ids)
#             
#            # Uses max pooling instead of classification token as apparently
#            # classification token does not contain meaningful semantic information
#            return torch.max(outputs[0], 1)[0]
        