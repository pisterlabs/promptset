
# # Copyright (c) 2019-present, HuggingFace Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import logging
import random
from argparse import ArgumentParser
from itertools import chain
from pprint import pformat
import warnings

import torch
import torch.nn.functional as F

import transformers

from transformers import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer, GPT2LMHeadModel, GPT2Tokenizer
from train import SPECIAL_TOKENS, build_input_from_segments, add_special_tokens_
from utils import get_dataset, download_pretrained_model


import tensorflow as tf

import tensorflow_hub as hub
import os
import numpy as np
import pandas as pd
import scipy as sp
from scipy.spatial import distance
from scipy.stats import chi2
# Set Up the Embedding Model - https://www.tensorflow.org/hub/tutorials/semantic_similarity_with_tf_hub_universal_encoder
module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
embed_model = hub.load(module_url)
print ("module %s loaded" % module_url)

# Set up the Perplexity Model
perplex_model = GPT2LMHeadModel.from_pretrained('gpt2')
perplex_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

""" Examples of good sentences created by the team. We made up 100 fictional text prompts we then had this model
 produce 20 alternative sentences to the original prompt and we hand-selected the examples from this process
 that we considered the best, the metrics of these sentences by relevancy, perplexity, and empathy. Relevancy
 is measured by the distance in vector space from the original intended sentence to the suggested alternative
 sentence. Perplexity is measured by the relative rarity of the sentence in GPT2's training data. Empathy is measured
using the data produced by an earlier study that labelled user comments on news articles on a scale of 1 to 7
for emapth yas determined by the psychologists retained to label the data. The idea was to see if there was
some fundamental similarity between the good sentences that could be measured across these three metrics. We would
then measure these metrics in vector space using Mahalanobis distance to determine which metrics or axes of the 
vector space were most deterministic for their classifcation as good sentences. We then intended to have the mdoel again 
produce 20 revisions of each sentence and then select the one most likely to be good by its closensss in vector space to
the center of the good sentence cluster in Mahalanobis distance. The key point of using Mahalanobis was that unlike Euclidean
distance which assumes equal weight to all axes in distance, MAhalanobis learns which axes are more important. 
In the end we found the more naive implementation found in interact_top20_formatted_V2.py in this folder
produced better results. Given more resources and time to produce more training data, it is likely that this 
more sophisticated implemenation would be superior."""

# some prompts had multiple good sentences produced. Converting what were orginially three separate columns in the excel sheet
# into one column for each metric.
good_sentence_df = pd.read_csv("AI_test_sentences_fixed sentences.csv")
temp_gs_df = good_sentence_df[['perplexity_1','embedding_1','empathy_1','perplexity_2','embedding_2','empathy_2','perplexity_3','embedding_3','empathy_3']]
temp_gs_perplex = (temp_gs_df[['perplexity_1','perplexity_2','perplexity_3']]).stack()
temp_gs_embedding = (temp_gs_df[['embedding_1','embedding_2','embedding_3']]).stack()
temp_gs_empathy = (temp_gs_df[['empathy_1','empathy_2','empathy_3']]).stack()

gs_metrics = []
temp_metric = []
for w in range(0,len(temp_gs_perplex)):
    temp_metric = []
    temp_metric.append(temp_gs_perplex.iloc[w])
    temp_metric.append(temp_gs_embedding.iloc[w])
    temp_metric.append(temp_gs_empathy.iloc[w])
    gs_metrics.append(temp_metric)

final_gs_metrics = pd.DataFrame(gs_metrics, columns = ['Perplexity','Embeddings','Empathy'])

# Mahalanobis function and class - both from here: https://www.machinelearningplus.com/statistics/mahalanobis-distance/
def mahalanobis(x=None, data=None, cov=None):
    """Compute the Mahalanobis Distance between each row of x and the data  
    x    : vector or matrix of data with, say, p columns.
    data : ndarray of the distribution from which Mahalanobis distance of each observation of x is to be computed.
    cov  : covariance matrix (p x p) of the distribution. If None, will be computed from data.
    """
    x_minus_mu = x - np.mean(data)
    if not cov:
        cov = np.cov(data.values.T)
    inv_covmat = sp.linalg.inv(cov)
    left_term = np.dot(x_minus_mu, inv_covmat)
    mahal = np.dot(left_term, x_minus_mu.T)
    return mahal.diagonal()

class MahalanobisOneclassClassifier():
    def __init__(self, xtrain, significance_level=0.01):
        self.xtrain = xtrain
        self.critical_value = chi2.ppf((1-significance_level), df=xtrain.shape[1]-1)
        print('Critical value is: ', self.critical_value)

    def predict_proba(self, xtest):
        mahalanobis_dist = mahalanobis(xtest, self.xtrain)
        self.pvalues = 1 - chi2.cdf(mahalanobis_dist, 2)
        return mahalanobis_dist

    def predict(self, xtest):
        return np.array([int(i) for i in self.predict_proba(xtest) > self.critical_value])


# Embedding Function... Pretty basic
def embed(input):
    temp_input = [input]
    return embed_model(temp_input)

# Perplexity Model not that complicated, from here: https://stackoverflow.com/questions/63543006/how-can-i-find-the-probability-of-a-sentence-using-gpt-2
def perplex_score(sentence, model, tokenizer):
    tokens_tensor = tokenizer.encode(sentence,add_special_tokens=False,return_tensors="pt")
    loss=model(tokens_tensor, labels =tokens_tensor)[0]
    return np.exp(loss.cpu().detach().numpy())

############################

""" this function is mostly unchanged from the original chatbot from hugging face that this was based on.
 This section mostly defines the top-k and top-p filtering functions but for the purposes of our 
 modifications to this code, most of these functions are vestigal.
 Source of original code: https://github.com/huggingface/transfer-learning-conv-ai """

def top_filtering(logits, top_k=0., top_p=0.9, threshold=-float('Inf'), filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k, top-p (nucleus) and/or threshold filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k: <=0: no filtering, >0: keep only top k tokens with highest probability.
            top_p: <=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
                whose total probability mass is greater than or equal to the threshold top_p.
                In practice, we select the highest probability tokens whose cumulative probability mass exceeds
                the threshold top_p.
            threshold: a minimal threshold to keep logits
    """
    assert logits.dim() == 1  # Only work for batch size 1 for now - could update but it would obfuscate a bit the code
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        # Remove all tokens with a probability less than the last token in the top-k tokens
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # Compute cumulative probabilities of sorted tokens
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probabilities > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Back to unsorted indices and set them to -infinity
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    indices_to_remove = logits < threshold
    logits[indices_to_remove] = filter_value

   # print("Logits: ",logits)
    return logits


"""
This function is where most of the changes were made to the logic of the chatbot to accomodate our 
changes to the selection process of the tokens. The main logic here is that the previous message from 
the conversation partner is taken as context for GPT. Then the empathy for the intended response is evaluated.
If the mean empathy of the tokens in the sentence is below 3.5, then we find the lowest empathy token. In this
version of the code, we have GPT look at the top 20 tokens to follow the sentence segment up to the least 
empathy token in the sentence. We then make 20 different versions of the sentence, one with each of the 20
tokens. We don't alter GPT's logic for the tokens following this swapped token, we let it complete the
sentences how it pleases.

We then evaluate each of the 20 sentences for each of the three metrics mentioned earlier. We then try and find
which of these 20 sentences is closest to the center of our examples of good sentencesin vector space by Mahalanobis
distance. Mahalanobis is used because the three dimensions by which we are looking at these sentences do not 
equally predict which of the sentences is a "good" sentence.

We choose the sentence from the 20 which scores as the smallest Mahalanobis distance from the center of the
good sentence examples we imported earlier. We compared the sentences that this script chose with the sentences
chosen by the simpler, more naive system in interact_top20_v2_formatted in the same folder. We simply evaluated
by human opinion which of the scripts produced more natural sounding and more empathetic sounding responses.
The simpler script produced better results and was used in the final demo, but this script is included in the 
github because we spent so much time on it.
"""

def sample_sequence(history, tokenizer, model, args, current_output=None):
    special_tokens_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
    if current_output is None:
        current_output = []

# define our later outputs
    embedding_dict = {}
    empathy_dict ={}
    perplex_dict = {}

# User's initial input, note conversation partner's input is under the run section of this script at the bottom

"""
 this key phrase was intended for the live demo of the AI for our final presentation. We originally wanted
 to have more history taken into consideration by GPT and therefore needed a way to manually reset the history 
 that GPT was considering when determining the most likely tokens to follow the sequence. In the end we used 
 just single exchanges for the demo, and cleared history after every rewrite making this function effectively
 vestigal. """

    history_erase = 0
    while history_erase ==0:
        raw_text = input(">>> ")
        if raw_text not in ["for_a_great_nose_indicates_a_great_man"]:
            history_erase +=1
        else:
            history = []
            print("History is now erased!")
    #######################################
# Get baseline embedding and perplexity of this initial intended sentence. Note I don't put int in the 
# output dictionaries because our final evaluation function will be comparing these metrics for our suggested 
# sentences against the original sentence
    baseline_embed = embed(raw_text)
    baseline_perplex = perplex_score(model=perplex_model,tokenizer=perplex_tokenizer,sentence = raw_text)

# Tokenize our initial intended sentence, note that we do this in case we need to keep this version of the 
# sentence and put it in the history

    tokenized_input = tokenizer.encode(raw_text)
    for i in tokenized_input:
        current_output.append(i)

    special_tokens_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)

# New to this script, we're adding these tokens to our raw text just making all text consistent so they can be read into history
    bos, eos, speaker1, speaker2 = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[:-1])
    sequence = [[bos]+ list(tokenized_input) + [eos]]
    instance = {}
    instance["input_ids"] = list(chain(*sequence))
    instance["token_type_ids"] = [speaker2 if i % 2 else speaker1 for i, s in enumerate(sequence) for _ in s]
    instance["mc_token_ids"] = len(instance["input_ids"]) - 1

    # First we convert the tokens that were just generated by GPT into words we can read

    import pandas as pd

    #Now we import our empathy lexicon data
    data = pd.read_csv('empathy_lexicon.txt', sep=",", header=None)
    data.columns = ["word","empathy_score"]
    # as it is the first column is useless as it is a header that won't import, so we erase first row
    new_data = data.iloc[1:]

    # because first row is a header originally, we need to convert scores back to numbers
    new_data['empathy_score'] = new_data['empathy_score'].astype(float)
    # if you recall from my empathy evaluator on github, I used a different tokenizer, the NLTK version
    # this means to find the GPT-generated tokens in the empathy lexicon I have to convert them back to words
    # then break them into separate words using this package
    import nltk
    import statistics
    from nltk.tokenize import sent_tokenize,word_tokenize
    # defining variables before first call to a void errors because I am just not that smart 
    score = 3.5
    score_list = []
    
    x = 0
    # break the original output into tokens, note this is not numbers but separating words into parts
    temp_sent = word_tokenize(raw_text)
    lower_temp_sent =[]
    for t in temp_sent:
        # all of empathy lexicon is lowercase, so make all words lwoercase to allow merges
        lower_temp_sent.append(t.lower())

    for word in lower_temp_sent:
        # this bit basically finds the word in the empathy lexicon and appends its assigned empathy value to the list for calculation of the mean
        # if the word is not in the list we go with the mean (3.5)
        if word in new_data['word'].values:
#            print(word)
            x = new_data.loc[new_data['word']==word,'empathy_score'].iloc[0]
            score_list.append(x)
        else:
            score_list.append(3.5)
            pass
 #   print("Original Score_List: ",score_list)
    result = statistics.mean(score_list)
 #   print("Original Result: ",result)

    orig_score_list = score_list.copy()
    orig_sentence = current_output.copy()

# Add the empathy of the newly generated sentences to the dictionary so we can parse them
    empathy_dict[raw_text] = result
    minIndex = 0
    temp_list = []
# Added cycle count to make sure we only run it through 5 times
    cycle_count = 0

    while cycle_count <20:
# find the position in the score list with the lowest empathy
        minIndex = orig_score_list.index(min(orig_score_list))
# break the current output prior to the lowest empathy token
        current_output = orig_sentence[0:(minIndex)]
# z is a cycle count, I only want to force it to pick the higher empathy token once, and then 
# let it finish the rest of the sentence as it pleases so if z ==1 we just leave the rest of the sentence alone
        z = 0
# most of this bit is from the original chatbot function
        for i in range(args.max_length):
# this part is calling the function which uses persona, history, GPT, etc
            instance = build_input_from_segments(history, current_output, tokenizer, with_eos=False)
# these two lines basically break all of this information, the hisroy, persona, et al, and breaks 
# them into tokens with numeric assignment
            input_ids = torch.tensor(instance["input_ids"], device=args.device).unsqueeze(0)
            token_type_ids = torch.tensor(instance["token_type_ids"], device=args.device).unsqueeze(0)
# the logits are rough probabilities of all tokens in GPT for following this sequence
            logits = model(input_ids, token_type_ids=token_type_ids)
            if isinstance(logits, tuple):  # for gpt2 and maybe others
                logits = logits[0] 
            logits = logits[0, -1, :] / args.temperature
# break logits into the top p or top k results depending on how it's defined 
# (this is defined below iirc in the run function of this script)
            logits = top_filtering(logits, top_k=args.top_k, top_p=args.top_p)
# convert logits into raw probabilities
            probs = F.softmax(logits, dim=-1)

# prev is the top prediction and what this script originally ends up using
            prev = torch.topk(probs, 1)[1] if args.no_sample else torch.multinomial(probs, 1)
# we instead force the top 5 results here, notice the 5 argument is in the else conditional, which I don't get but it
# almost always calls this instead of the if portion

            test = torch.topk(probs, 1)[1] if args.no_sample else torch.multinomial(probs, 20)
# this part is mine, again we only want to change the lowest empathy token so it will replace 
# the token if z ==0, but otherwise generate tokens as normal.
            if z ==0:
# we now decode the top k tokens back to readable words like before
                temp_test_output = tokenizer.decode(test,skip_special_tokens=True)
                temp_list =[]
# we now split these words into a list of word tokens, again actual words not numbers
                temp_sent_second = word_tokenize(temp_test_output)
# we once again find these words in the emapthy lexicon and get their emapthy values.

                for word in temp_sent_second:

                    if word in new_data['word'].values:
                        x = new_data.loc[new_data['word']==word,'empathy_score'].iloc[0]
                        temp_list.append(x)
                    else:
                        temp_list.append(3.5)
                prev = test[cycle_count]
                z +=1
# this part here allows the rest of the sentence to be generated as normal.
            if i < args.min_length and prev.item() in special_tokens_ids:
                while prev.item() in special_tokens_ids:
                    if probs.max().item() == 1:
                        warnings.warn("Warning: model generating special token with probability 1.")
                        break  # avoid infinitely looping over special token
                    prev = torch.multinomial(probs, num_samples=1)
# this part is key to some weird errors I sometimes see. If I type something that the chatbot has 
# no idea how to respond to it sometimes defers to what I call second_prev, it seems to happen if the chatbot 
# messes up and thinks some token has 100% probability but usually this results in it breaking and just 
# printing something like "?" as sort of a default response for when it doesn't understand
                    print("Let's see what the second prev is: ", prev)
            if prev.item() in special_tokens_ids:
                break
            current_output.append(prev.item())
#################################################
        score_list = []
        temp_output = tokenizer.decode(current_output,skip_special_tokens=True)
# Evaluate Perplexity and Embedding for each of these sentences to parse over later
        perplex_dict[temp_output] = perplex_score(sentence = temp_output, model=perplex_model, tokenizer = perplex_tokenizer)
        embedding_dict[temp_output] = embed(temp_output)

        x = 0
        temp_sent = word_tokenize(temp_output)
        lower_temp_sent =[]
        for t in temp_sent:
            lower_temp_sent.append(t.lower())

        for word in lower_temp_sent:

            if word in new_data['word'].values:
                x = new_data.loc[new_data['word']==word,'empathy_score'].iloc[0]
                score_list.append(x)
            else:
                pass
# note that the empathy score as I currently define it is just mean empathy.
        empathy_dict[temp_output] = statistics.mean(score_list)
        result = statistics.mean(score_list)
    # we increase cycle_count to keep generating alternate versions of sentences 
        cycle_count +=1

    temp_dot_products =[]
# find dot products of all the embeddings of the generated sentences against the embedding of the original sentence.
    final_embeds = {}
    for i, embeds in embedding_dict.items():
        temp_dot_products.append(np.dot((np.squeeze(np.asarray(baseline_embed))),(np.squeeze(np.asarray(embeds)))))
        final_embeds[i] = np.dot((np.squeeze(np.asarray(baseline_embed))),(np.squeeze(np.asarray(embeds))))
# Print all of my messy new metrics so we can compare them for our sentences.

    print("Empathy Dict: ",empathy_dict)
    print("Embedding Dictionary: ",final_embeds)
    print("Baseline Perplexity: ",baseline_perplex)
    print("Perplexity Dictionary: ",perplex_dict)

    temp_list = []
    final_results = {}

    for i in range(1,20):
        temp_list =[]
        temp_perplex = list(perplex_dict.values())[i]
        temp_empathy = list(empathy_dict.values())[i]
        temp_list.append(temp_perplex)
        temp_list.append(list(final_embeds.values())[i])
        temp_list.append(temp_empathy)
    
        temp_key = list(empathy_dict.keys())[i]
        final_results[temp_key] = temp_list

    final_test = pd.DataFrame.from_dict(final_results, orient = 'index', columns= ["Perplexity","Embeddings","Empathy"])
    clf = MahalanobisOneclassClassifier(final_gs_metrics,significance_level = 0.1)

    final_similarity = clf.predict_proba(final_test)

    for q in range(len(final_results)):
        print("Sentence: ", list(final_results.keys())[q])
        print("Similarity: ",final_similarity[q])

    max_similarity_index = np.argmin(final_similarity)
    final_sentence = list(final_results.keys())[max_similarity_index]
    tokenized_output = tokenizer.encode(final_sentence)

    current_output = []
    for t in tokenized_output:
        current_output.append(t)
    return current_output

def run():
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="", help="Path or url of the dataset. If empty download from S3.")
    parser.add_argument("--dataset_cache", type=str, default='./dataset_cache', help="Path or url of the dataset cache")
    parser.add_argument("--model", type=str, default="openai-gpt", help="Model type (openai-gpt or gpt2)", choices=['openai-gpt', 'gpt2'])  # anything besides gpt2 will load openai-gpt
    parser.add_argument("--model_checkpoint", type=str, default="", help="Path, url or short name of the model")
    parser.add_argument("--max_history", type=int, default=2, help="Number of previous utterances to keep in history")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")

    parser.add_argument("--no_sample", action='store_true', help="Set to use greedy decoding instead of sampling")
    parser.add_argument("--max_length", type=int, default=20, help="Maximum length of the output utterances")
    parser.add_argument("--min_length", type=int, default=1, help="Minimum length of the output utterances")
    parser.add_argument("--seed", type=int, default=0, help="Seed")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling softmax temperature")
    parser.add_argument("--top_k", type=int, default=0, help="Filter top-k tokens before sampling (<=0: no filtering)")
    parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus filtering (top-p) before sampling (<=0.0: no filtering)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__file__)
    logger.info(pformat(args))

    if args.model_checkpoint == "":
        if args.model == 'gpt2':
            raise ValueError("Interacting with GPT2 requires passing a finetuned model_checkpoint")
        else:
            args.model_checkpoint = download_pretrained_model()
	
	
    if args.seed != 0:
    	random.seed(args.seed)
    	torch.random.manual_seed(args.seed)
    	torch.cuda.manual_seed(args.seed)


    logger.info("Get pretrained model and tokenizer")
    tokenizer_class, model_class = (GPT2Tokenizer, GPT2LMHeadModel) if args.model == 'gpt2' else (OpenAIGPTTokenizer, OpenAIGPTLMHeadModel)
    tokenizer = tokenizer_class.from_pretrained(args.model_checkpoint)
    model = model_class.from_pretrained(args.model_checkpoint)
    model.to(args.device)
    add_special_tokens_(model, tokenizer)

   # logger.info("Sample a personality")
    dataset = get_dataset(tokenizer, args.dataset_path, args.dataset_cache)
   # personalities = [dialog["personality"] for dataset in dataset.values() for dialog in dataset]
   # personality = random.choice(personalities)
   # logger.info("Selected personality: %s", tokenizer.decode(chain(*personality)))

    history = []
    while True:
#        history_erase = 0
#        while history_erase ==0:
        
        raw_text = input(">>> ")
        #    if raw_text not in ["for_a_great_nose_indicates_a_great_man"]:
        #        history_erase +=1
        #    else:
        #        history = []
        #        print("History is now erased!")
        while not raw_text:
            print('Prompt should not be empty!')
            raw_text = input(">>> ")
        
        history.append(tokenizer.encode(raw_text))
        with torch.no_grad():
            out_ids = sample_sequence(history, tokenizer, model, args)
        history.append(out_ids)
        history = history[-(2*args.max_history+1):]

#        print("out_ids: ",out_ids)
#        print(type(out_ids))
        out_text = tokenizer.decode(out_ids, skip_special_tokens=True)
        
        print("out_text: ",out_text)
#        print(type(out_text))

#######################################

if __name__ == "__main__":
    run()
