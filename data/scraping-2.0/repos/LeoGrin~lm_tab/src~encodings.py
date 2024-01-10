# create a new model
import torch.nn as nn
import torch
from tabpfn import TabPFNClassifier
from src.utils import preprocess_input
from transformers import BertModel, BertTokenizer, AutoTokenizer
from sentence_transformers import SentenceTransformer
import numpy as np
from skrub import MinHashEncoder, TableVectorizer
import pandas as pd
import tiktoken
from src.models import BertAndTabPFN
import os
import openai
from openai.embeddings_utils import get_embedding
from dotenv import load_dotenv
from tqdm import tqdm
from ast import literal_eval
from tenacity import retry, stop_after_attempt, wait_random_exponential
from torch.utils.data import DataLoader
from sklearn.feature_extraction.text import HashingVectorizer
from scipy.sparse import hstack, vstack
from fasttext import load_model
import fasttext.util
from sklearn.base import BaseEstimator, TransformerMixin



#taken from https://github.com/pcerda/string_categorical_encoders/blob/master/column_encoder.py
class PretrainedFastText(BaseEstimator, TransformerMixin):
    """
    Category embedding using a fastText pretrained model.
    """

    def __init__(self, n_components, language='english'):
        self.n_components = n_components
        self.language = language

    def fit(self, X, y=None):

        path_dict = dict(
            #english='crawl-300d-2M-subword.bin',
            english="cc.en.300.bin",
            french='cc.fr.300.bin',
            hungarian='cc.hu.300.bin')
        


        if self.language not in path_dict.keys():
            raise AttributeError(
                'language %s has not been downloaded yet' % self.language)


        self.ft_model = load_model(path_dict[self.language])
        # reduce dimension if necessary
        if self.n_components < 300:
            fasttext.util.reduce_model(self.ft_model, self.n_components)

        return self

    def transform(self, X):
        X = X.ravel()
        unq_X, lookup = np.unique(X, return_inverse=True)
        X_dict = dict()
        for i, x in enumerate(unq_X):
            if x.find('\n') != -1:
                unq_X[i] = ' '.join(x.split('\n'))

        for x in unq_X:
            X_dict[x] = self.ft_model.get_sentence_vector(x)

        X_out = np.empty((len(lookup), self.n_components))
        for x, x_out in zip(unq_X[lookup], X_out):
            x_out[:] = X_dict[x]
        return X_out

@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def get_batch_embeddings(texts: str, model="text-embedding-ada-002"):
    res = openai.Embedding.create(input=texts, model=model)["data"]
    return np.array([literal_eval(str(x["embedding"])) for x in res])


from transformers import AutoTokenizer, AutoModel
import torch


#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask



def encode_hf(sentences, model, batch_size=1):
    print("Encoding with HF")
    # Load AutoModel from huggingface model repository
    tokenizer = AutoTokenizer.from_pretrained(model)
    model = AutoModel.from_pretrained(model)

    # Set padding token if not set
    if tokenizer.pad_token is None:
        print("Setting padding token")
        tokenizer.pad_token = tokenizer.eos_token

    # Make sure model and tokenizer are on the same device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    # Create a DataLoader to handle batching of the sentences
    sentences_loader = DataLoader(sentences, batch_size=batch_size, shuffle=False)

    # List to store all embeddings
    all_embeddings = []

    for sentence_batch in sentences_loader:
        # Tokenize sentences
        encoded_input = tokenizer(sentence_batch, padding=True, truncation=True, max_length=128, return_tensors='pt')

        # Move tensors to the same device as the model
        encoded_input = encoded_input.to(device)

        # Compute token embeddings
        with torch.no_grad():
            model_output = model(**encoded_input)

        # Perform pooling. In this case, mean pooling
        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

        # Move embeddings to CPU, convert to numpy and store
        all_embeddings.extend(sentence_embeddings.cpu().numpy())
    return np.array(all_embeddings)


def encode(X, col, encoder_name, dataset_name=None, use_cache=True, override_cache=False, fail_if_not_cached=False):
    print("working dir", os.getcwd())
    if use_cache and dataset_name is not None and not override_cache:
        # check if the cache exists
        try:
            res = np.load(f"cache/{dataset_name}_{col}_{encoder_name.replace('/', '_')}.npy")
            print("Loaded from cache")
            return res
        except FileNotFoundError:
            if fail_if_not_cached:
                raise FileNotFoundError(f"Cache not found for {dataset_name}_{col}_{encoder_name.replace('/', '_')}.npy")
            print("Cache not found, computing")
            pass
    X_col = np.array(X[col])
    encoder_type, encoder_params = encoder_name.split("__", 1)
    print("Encoder type", encoder_type)
    print("Encoder params", encoder_params)
    if encoder_type == "lm":
        encoder = SentenceTransformer(encoder_params)
        X_col = X_col.reshape(-1)
        if "e5" in encoder_params:
            print("Seems like a e5 model, adding 'query: '")
            print(encoder_params)
            X_col = np.array(["query: " + elem for elem in X_col])
            print("Samples:")
            print(X_col[:10])
        res = encoder.encode(X_col)
    elif encoder_type == "hf":
        res = encode_hf(X_col.tolist(), encoder_params)
    elif encoder_type == "fasttext":
        res = PretrainedFastText(n_components=int(encoder_params)).fit_transform(X_col)
    elif encoder_type == "skrub":
        if encoder_params.startswith("minhash"):
            n_components = int(encoder_params.split("_")[1])
            print("n components", n_components)
            if len(encoder_params.split("_")) > 2:
                analyzer = encoder_params.split("_")[2]
                tokenizer = encoder_params.split("_")[3]
                if tokenizer == "none":
                    tokenizer = None
                print(f"Using {analyzer} analyser and {tokenizer} tokenizer, {n_components} components")
            else:
                analyzer = "char"
                tokenizer = None
            encoder = MinHashEncoder(n_components=n_components, analyzer=analyzer, tokenizer=tokenizer,
                                    ngram_range=(2, 4) if analyzer == "char" else (1, 3), hashing="fast" if analyzer == "char" else "murmur")
            # reshape to 2d array
            # if pandas dataframe, convert to numpy array
            res = X_col.reshape(-1, 1)
            res = encoder.fit_transform(res)
        else:
            raise ValueError(f"Unknown skrub encoder {encoder_params}")
    elif encoder_type == "openai":
        load_dotenv()  # take environment variables from .env.
        openai_api_key = os.getenv('OPENAI_API_KEY')
        if openai_api_key is None:
            raise ValueError("OPENAI_API_KEY is not set")
        else:
            openai.api_key = openai_api_key
        embedding_model = "text-embedding-ada-002"
        embedding_encoding = "cl100k_base"  # this the encoding for text-embedding-ada-002
        #max_tokens = 8000
        #encoding = tiktoken.get_encoding(embedding_encoding)
        #n_tokens = X_col.combined.apply(lambda x: len(encoding.encode(x)))
        ## check that the max number of tokens is not exceeded
        #if (n_tokens > max_tokens).any():
        #    raise ValueError("The maximum number of tokens is exceeded")
        #res = np.array([get_embedding(x, engine=embedding_model) for x in X_col.tolist()])
        #df = pd.DataFrame(X_col, columns=["name"])
        #res = df.name.apply(lambda x: get_embedding(x, engine=embedding_model))
        # embed in batch of 100
        for i in tqdm(range(0, len(X_col), 500)):
            batch = X_col[i:i+500].tolist()
            print(batch)
            res_batch = get_batch_embeddings(batch, model=embedding_model)
            if i == 0:
                res = res_batch
            else:
                res = np.concatenate([res, res_batch], axis=0)

    elif encoder_type == "bert_custom":
        #FIXME: results are not great with thisr
        transformer_name = encoder_params
        # I could instantiate just Bert but this is to check for bugs in BertAndTabPFN
        lm = BertAndTabPFN(preprocess_before_tabpfn=True, linear_translator=False, transformer_name=transformer_name,
                                dim_tabpfn=30, lora=False, disable_dropout=False).to('cuda')
        lm .eval()
        tokenizer = AutoTokenizer.from_pretrained(transformer_name)
        texts = X_col.tolist()
        all_encoding = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        # print the non-padded length median and quantiles
        non_padded_lengths = np.sum(all_encoding["attention_mask"].numpy(), axis=1)
        max_length = np.quantile(non_padded_lengths, 0.95)
        all_encoding = tokenizer(texts, padding="max_length", truncation=True, max_length=int(max_length), return_tensors="pt")
        # move to gpu
        all_encoding = {k: v.to('cuda') for k, v in all_encoding.items()}
        # generate random y
        with torch.no_grad():
            res = lm (**all_encoding, y=None, return_tabpfn_input=True).cpu().detach().numpy()
    elif encoder_type == "bert_custom_pooling":
        transformer_name = encoder_params
        # I could instantiate just Bert but this is to check for bugs in BertAndTabPFN
        lm = BertAndTabPFN(preprocess_before_tabpfn=True, linear_translator=False, transformer_name=transformer_name,
                                dim_tabpfn=30, lora=False, disable_dropout=False, embedding_stragegy="mean_pooling").to('cuda')
        lm .eval()
        tokenizer = AutoTokenizer.from_pretrained(transformer_name)
        texts = X_col.tolist()
        all_encoding = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        # # print the non-padded length median and quantiles
        # non_padded_lengths = np.sum(all_encoding["attention_mask"].numpy(), axis=1)
        # max_length = np.quantile(non_padded_lengths, 0.95)
        # all_encoding = tokenizer(texts, padding="max_length", truncation=True, max_length=int(max_length), return_tensors="pt")
        # move to gpu
        all_encoding = {k: v.to('cuda') for k, v in all_encoding.items()}
        # generate random y
        with torch.no_grad():
            res = lm (**all_encoding, y=None, return_tabpfn_input=True).cpu().detach().numpy()
    
    if use_cache and dataset_name is not None:
        print("Saving to cache")
        # save the cache
        np.save(f"cache/{dataset_name}_{col}_{encoder_name.replace('/', '_')}.npy", res)
    
    return res

def encode_high_cardinality_features(X, encoder_name, dataset_name=None, use_cache=True, override_cache=False, cardinality_threshold=30, fail_if_not_cached=False):
    tb = TableVectorizer(cardinality_threshold=cardinality_threshold,
                        high_card_cat_transformer = "passthrough",
                        low_card_cat_transformer = "passthrough",
                        numerical_transformer = "passthrough",
                        datetime_transformer = "passthrough",
    ) #just to get the high cardinality columns
    tb.fit(X)
    # get high cardinality columns
    high_cardinality_columns = []
    for name, trans, cols in tb.transformers_:
        print(name, cols)
        if "high" in name:
            high_cardinality_columns.extend(cols)
            break
    print("High cardinality columns", high_cardinality_columns)
    # encode the high cardinality columns
    res = []
    lengths = []
    for col in high_cardinality_columns:
        new_enc = encode(X, col, encoder_name, dataset_name=dataset_name, use_cache=use_cache, override_cache=override_cache, fail_if_not_cached=fail_if_not_cached)
        res.append(new_enc)
        lengths.append(new_enc.shape[1])
    # create a dataframe with name original_col_name__index
    df = pd.DataFrame(np.concatenate(res, axis=1))

    #df = pd.DataFrame(np.concatenate(res, axis=1))
    # for i in range(len(res)):
    #     for j in range(lengths[i]):
    #         df.rename(columns={i*lengths[i] + j: high_cardinality_columns[i] + "__" + str(j)}, inplace=True)
    new_column_names = []
    for i in range(len(res)):
        for j in range(lengths[i]):
            new_column_names.append(high_cardinality_columns[i] + "__" + str(j))
    df.columns = new_column_names
    return df, X.drop(high_cardinality_columns, axis=1)

