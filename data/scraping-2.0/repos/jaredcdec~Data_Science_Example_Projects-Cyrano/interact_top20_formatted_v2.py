
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
warnings.filterwarnings('ignore')

import torch
import operator
import torch.nn.functional as F

import transformers

from transformers import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer, GPT2LMHeadModel, GPT2Tokenizer, BasicTokenizer
from train import SPECIAL_TOKENS, build_input_from_segments, add_special_tokens_
from utils import get_dataset, download_pretrained_model

###############################

from tabulate import tabulate
import tensorflow as tf

import tensorflow_hub as hub
import os
import numpy as np
import pandas as pd

# Set Up the Embedding Model - from https://www.tensorflow.org/hub/tutorials/semantic_similarity_with_tf_hub_universal_encoder
module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
embed_model = hub.load(module_url)
print ("module %s loaded" % module_url)

# Set up the Perplexity Model
perplex_model = GPT2LMHeadModel.from_pretrained('gpt2')
perplex_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

example_tokenizer = BasicTokenizer()

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

# This functrion hasn't changed much from the original chatbot this was based on which can be found 
# at this URL: https://github.com/huggingface/transfer-learning-conv-ai

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
Most of the major changes to the original code happen here in this function. The idea was to interfere with how GPT
originally selected which tokens were likely to follow a sequence of text. In this code, the conversation partner 
writes an original message and the user writes their intended response. The mean empathy of the tokens in the user's 
intended response is then evaluated. If it is below 3.5 (or neutral empathy), this code finds the lowest empathy token
among the ones in the original response and cuts the sentence segment up to the point prior to that token. It then
looks at the 20 most likely tokens that GPT predicts to follow the sequence of the conversation partner's original message
and the sentence segment of the user's intended response up to that lowest-empathy token. In this version of the code 
it simply chooses the most empathetic of those 20 tokens based on the unigram empathy data. GPT then completes the
rest of the sentence as it pleases. Despite this being more naive than the approach implement in interact_experiment.py 
in this same folder, this version of the code produced better results in the tests we ran and was the version used in our 
final live demonstration.
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
The original purpose of this function was for the live demonstration. GPT predicts the next token to follow the sequence
based on the tokens saved in a list of the previous tokens which is called "history". In our test runs, exchanges with 
too many responses or too long of a sequence of tokens tended to produce less relevant results. We decided to implement
this phrase so that we could control when the history was erased. In the end, we used exchanges for the live demo that 
were only an initial message and then a response. Consequently we hardcoded the code to erase the hsitory after each
cycle, making this function effectively vestigal.
"""

    history_erase = 0
    while history_erase ==0:
        raw_text = input(">>> ")
        if raw_text not in ["for_a_great_nose_indicates_a_great_man"]:
            history_erase +=1
        else:
            history = []
            print("History is now erased!")

    #######################################
# Get baseline embedding and perplexity of this initial intended sentence. Note I don't put it in the output dictionaries because 
# our final evaluation function will be comparing these metrics for our suggested sentences against the original sentence
    baseline_embed = embed(raw_text)
    baseline_perplex = perplex_score(model=perplex_model,tokenizer=perplex_tokenizer,sentence = raw_text)

# Tokenize our initial intended sentence, note that we do this in case we need to keep this version of the sentencve and put it in the history

    tokenized_input = tokenizer.encode(raw_text)
    draft_tokens = tokenizer.convert_ids_to_tokens(tokenized_input, skip_special_tokens=True)

# Joe Mirza's additions
    draft_tokens = np.array(draft_tokens).astype(str)

# Jared's additions resume
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

# Now we import our empathy lexicon data
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
    temp_sent = example_tokenizer.tokenize(raw_text)
    lower_temp_sent =[]
    for t in temp_sent:
# all of empathy lexicon is lowercase, so make all words lwoercase to allow merges
        lower_temp_sent.append(t.lower())


    for word in lower_temp_sent:
# this bit basically finds the word in the empathy lexicon and appends its assigned empathy value to the list 
# for calculation of the mean empathy of the sentence. If the word is not in the list we assign it a neutral empathy score (3.5)
        if word in new_data['word'].values:
            x = new_data.loc[new_data['word']==word,'empathy_score'].iloc[0]
            score_list.append(x)
        else:
            score_list.append(3.5)
            pass

  ##################################### Additional Code from Joe Mirza

    print("\n\nTokenized Draft Response and Empathy Value of Tokens")

    tokens_a = [' '] + list(draft_tokens)
    emp_a = list(np.round(score_list,2))

    orig_result = statistics.mean(score_list)

    alist =  draft_tokens
    alist = np.char.replace(alist, '</w>', '')


    alist = list(alist) + ['Mean Empathy']
    len_list = len(alist)


    emp_avg = orig_result
    emp_a = emp_a + [round(emp_avg,2)]

    tot_len = 0
    for a in alist:
        len_a = max(len(a), 4)
        tot_len += (len_a + 5)
    tot_len += 11
    print("-"*tot_len)

    print(" "*10 + "|", end="")
    #tot_len = 11
    for a in alist:
        len_a = max(len(a), 4)
    #    tot_len += len_a + 5
        print(f'{a:>{len_a+3}} |', end="")
    print("\n", tot_len*"-")

    print("Empathy   |", end="")
    for a, e in zip(alist, emp_a):
        len_a = max(len(a), 4)
        print(f'{e:>{len_a+3}} |', end="")
    print("\n", tot_len*"-")

    print("\nMean empathy of draft response: ", round(orig_result,2))
##########################

# Jared's code resumes

    orig_score_list = score_list.copy()
    orig_sentence = current_output.copy()

# Add the empathy of the newly generated sentences to the dictionary so we can parse them
    empathy_dict[raw_text] = orig_result
    minIndex = 0
    temp_list = []
# Added cycle count to make sure we only run it through 5 times
    cycle_count = 0
# if empathy in proposed response is below 3.5, the below loop is triggered.
    if orig_result < 3.5:
# find the position in the score list with the lowest empathy
        minIndex = orig_score_list.index(min(orig_score_list))

        least_empathetic_token = orig_sentence[minIndex]
        final_least_empathetic_token = tokenizer.decode(least_empathetic_token)
        print("The least empathetic token in original sentence is: ", final_least_empathetic_token, "\n")

# break the current output prior to the lowest empathy token
        current_output = orig_sentence[0:(minIndex)]
# z is a cycle count, I only want to force it to pick the higher empathy token once, and then let it finish 
# the rest fo the sentence as it pleases. So if z ==1 we just leave the rest of the sentence alone
        z = 0
# most of this bit is from the original chatbot function
        for i in range(args.max_length):
# this part is calling the function which uses persona, history, GPT, etc
            instance = build_input_from_segments(history, current_output, tokenizer, with_eos=False)
# these two lines basically break all of this information, the hisroy, persona, et al, and breaks them into 
# tokens with numeric assignment
            input_ids = torch.tensor(instance["input_ids"], device=args.device).unsqueeze(0)
            token_type_ids = torch.tensor(instance["token_type_ids"], device=args.device).unsqueeze(0)
# the logits are rough probabilities of all tokens in GPT for following this sequence
            logits = model(input_ids, token_type_ids=token_type_ids)
            if isinstance(logits, tuple):  # for gpt2 and maybe others
                logits = logits[0]
# I honestly don't understand this line or the if condition above, what si temperature?
            logits = logits[0, -1, :] / args.temperature
# break logits into the top p or top k results depending on how it's defined (this is defined below in the run 
# function of this script)
            logits = top_filtering(logits, top_k=args.top_k, top_p=args.top_p)
# convert logits into raw probabilities
            probs = F.softmax(logits, dim=-1)

# prev is the top prediction and what this script ends up using
            prev = torch.topk(probs, 1)[1] if args.no_sample else torch.multinomial(probs, 1)

# we instead force the code to produce the top 20 results among which we choose the highest empathy token

            test = torch.topk(probs, 1)[1] if args.no_sample else torch.multinomial(probs, 20)
# this part is mine, again we only want to change the lowest empathy token so it will replace the token if z ==0, 
# but otherwise generate tokens as normal.

            if z ==0:
# we now decode the top k tokens back to readable words like before
                temp_test_output = tokenizer.decode(test,skip_special_tokens=True)
                temp_list =[]
# we now split these words into a list of word tokens, again actual words not numbers

                temp_sent_second = example_tokenizer.tokenize(temp_test_output)
# we once again find these words in the emapthy lexicon and get their emapthy values.

                for word in temp_sent_second:

                    if word in new_data['word'].values:
                        x = new_data.loc[new_data['word']==word,'empathy_score'].iloc[0]
                        temp_list.append(x)
                    else:
                        temp_list.append(3.5)
                token_analysis_dict = {}
                token_analysis_dict = {temp_sent_second[i]: temp_list[i] for i in range(len(temp_sent_second))}
                token_analysis_dict_s = sorted(token_analysis_dict,reverse=True)
# We now find the word with the highest empathy among these 20 and we stick that in the sentence.

                maxIndex = temp_list.index(max(temp_list))
                prev = test[maxIndex]

  ##################################### Joe Mirza Code Additions

                sorted_token_dict = dict( sorted(token_analysis_dict.items(), key=operator.itemgetter(1),reverse=True))

                print("5 Most Empathetic Candidate Tokens")
                print("-"*34)
                print("| Rank |        Token |  Empathy |")

                print("-"*34)
                for i, j in enumerate(list(sorted_token_dict.keys())[:5]):
                    word = j
                    empathy = round(sorted_token_dict[j],2)
                    print(f'|{i+1:>{5}} | {word:>{12}} | {empathy:>8.2f} |')
                    print("-"*34)

                print("Highest empathy token among these is: ", list(sorted_token_dict.keys())[0], "with an empathy of: " , round(sorted_token_dict[list(sorted_token_dict.keys())[0]],2))

  ##################################### 

# Jared's code resumes
                z +=1
# this part here allows the rest of the sentence to be generated as normal.
            if i < args.min_length and prev.item() in special_tokens_ids:
                while prev.item() in special_tokens_ids:
                    if probs.max().item() == 1:
                        warnings.warn("Warning: model generating special token with probability 1.")
                        break  # avoid infinitely looping over special token
                    prev = torch.multinomial(probs, num_samples=1)
"""This part is key to some weird errors I sometimes see. If I type something that the chatbot legit has no idea how to 
respond to it sometimes defers to what I call second_prev, it seems to happen if the chatbot messes up and thinks some 
token has 100% probability but usually this results in it breaking and just printing something like "?" as sort of a 
default response for when it doesn't understand
"""
                    print("Let's see what the second prev is: ", prev)
            if prev.item() in special_tokens_ids:
                break
            current_output.append(prev.item())
#################################################
        score_list = []
        temp_output = tokenizer.decode(current_output,skip_special_tokens=True)
""" Evaluate Perplexity and Embedding for each of these sentences to parse over later
 Note that this version of the code does not consider the embedding or perplexity  of the sentences, as it
 performed better. The version of the code that uses these metrics is in this folder as interact_experiment.py
 but was not used in the final demonstration. """
        perplex_dict[temp_output] = perplex_score(sentence = temp_output, model=perplex_model, tokenizer = perplex_tokenizer)
        embedding_dict[temp_output] = embed(temp_output)

        x = 0
        temp_sent = example_tokenizer.tokenize(temp_output)
        lower_temp_sent =[]
        for t in temp_sent:
            lower_temp_sent.append(t.lower())

        for word in lower_temp_sent:

            if word in new_data['word'].values:
                x = new_data.loc[new_data['word']==word,'empathy_score'].iloc[0]
                score_list.append(x)
            else:
                pass

        result = statistics.mean(score_list)


  ##################################### Joe Mirza's Code Additions

        print("\nCyrano's Tokenized Draft Response and Empathy Value of Tokens")

        tokens_b = [' '] + list(lower_temp_sent)
        emp_b = list(np.round(score_list,2))

        orig_result = statistics.mean(score_list)

        blist =  lower_temp_sent
        blist = np.char.replace(blist, '</w>', '')

        blist = list(blist) + ['Mean Empathy']
        len_list = len(blist)

        emp_avg = orig_result
        emp_b = emp_b + [round(emp_avg,2)]

        tot_len = 0
        for b in blist:
            len_b = max(len(b), 4)
            tot_len += (len_b + 5)
        tot_len += 11
        print("-"*tot_len)

        print(" "*10 + "|", end="")
        #tot_len = 11
        for b in blist:
            len_b = max(len(b), 4)
        #    tot_len += len_a + 5
            print(f'{b:>{len_b+3}} |', end="")
        print("\n", tot_len*"-")

        print("Empathy   |", end="")
        for b, e in zip(blist, emp_b):
            len_b = max(len(b), 4)
            print(f'{e:>{len_b+3}} |', end="")
        print("\n", tot_len*"-")

        print("\nMean empathy of Cyrano's Suggestion: ", round(orig_result,2))
    ########################## 

# Jared's Code resumes

    # note that the empathy score as I currently define it is just mean empathy.
        empathy_dict[temp_output] = statistics.mean(score_list)

        result = statistics.mean(score_list)

        cycle_count +=1

    else:
# this section only triggers if the original intended response has a mean empathy above 3.5.
        print("Good job! your sentence was already quite empathetic!")
        print("Original Sentence's empathy was: ",orig_result)
        return current_output

    temp_dot_products =[]
# find dot products of all the embeddings of the generated sentences against the embedding of the original sentence.
# Note that again these metrics are not used as part of the logic here. Only in interact_experiment.py also in this 
# folder but was not used in the final demonstration.
    final_embeds = {}
    for i, embeds in embedding_dict.items():
        temp_dot_products.append(np.dot((np.squeeze(np.asarray(baseline_embed))),(np.squeeze(np.asarray(embeds)))))
        final_embeds[i] = np.dot((np.squeeze(np.asarray(baseline_embed))),(np.squeeze(np.asarray(embeds))))

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

        temp_y = tokenizer.encode(raw_text)
        #print("\nLine 1 - Tokenized Partner's Original Sentence: ",temp_y)



        history.append(tokenizer.encode(raw_text))
        with torch.no_grad():
            out_ids = sample_sequence(history, tokenizer, model, args)
        history.append(out_ids)
        history = history[-(2*args.max_history+1):]

#        print("out_ids: ",out_ids)
#        print(type(out_ids))
        out_text = tokenizer.decode(out_ids, skip_special_tokens=True)

        history = []
   #     print("History is now erased!")

        print("\nFinal sentence: ",out_text, "\n")
#        print(type(out_text))

#######################################

if __name__ == "__main__":
    run()
