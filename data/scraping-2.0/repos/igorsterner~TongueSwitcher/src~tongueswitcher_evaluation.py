import itertools
import json
import os
import pickle
import random
import re
import string
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import _jsonnet
import emoji
import openai
import spacy
from easydict import EasyDict
from lingua import Language, LanguageDetectorBuilder
from nltk import ngrams
from sklearn.metrics import precision_recall_fscore_support
from spacy.tokens import Doc
from tokenizations import get_alignments
from tqdm import tqdm
from transformers import pipeline

import utils.classifier_feature_util as clfutil
from tongueswitcher import *
from utils.corpus import Corpus

nlp_big = spacy.load('de_dep_news_trf')

from flair.data import Sentence
from flair.models import SequenceTagger
from seqeval.metrics import classification_report
from tokenizations import get_alignments

cost = 0

tongueswitcher_file = Path("/results/rules-results.pkl")
denglisch_file = Path("/results/denglisch-results.pkl")
tsbert_file = Path("/results/tsbert-results.pkl")

gold_file = Path("/results/gold.pkl")

tongueswitcher_testset_dir = Path("../tongueswitcher-corpus/tongueswitcher_testset.jsonl")
tsbert_model = "igorsterner/german-english-code-switching-identification"

languages = [Language.ENGLISH, Language.GERMAN]
detector = LanguageDetectorBuilder.from_languages(*languages).build()

random.seed(10)

def open_token_gold(file):
    data = {}
    with open(file, 'r') as f:
        for line in tqdm(f):
            json_line = json.loads(line)
            if json_line["answer"] != "accept":
                    continue
            token_idxs = {i["start"]: i["end"] for i in json_line["tokens"]}
            span_idxs = {i["start"]: i["label"] for i in json_line["spans"]}
            labels = [span_idxs[token["start"]] if token["start"] in span_idxs else 'D' for token in json_line["tokens"]]
            labels = [label if (label != 'UNSURE' and label != 'UNKNOWN') else 'D' for label in labels]
            labels = ['E' if label == 'ENGLISH' else label for label in labels]
            labels = ['M' if label == 'MIXED' else label for label in labels]
            punct = find_punct(json_line["text"])
            # labels = [label if i not in punct else "P" for i, label in enumerate(labels)]
            tokens = [i["text"] for i in json_line["tokens"]]
            data[json_line["meta"]["URL"][35:]] = {
                "labels": [label if i not in punct else "P" for i, label in enumerate(labels)],
                "text": json_line["text"],
                "punct": punct,
                "tokens": tokens,
                "token_idx": {i["text"]: {"start": i["start"], "end": i["end"]} for i in json_line["tokens"]},
                "annotation": [{"token": token, "lan": label} for i, (token, label) in enumerate(zip(tokens, labels))]
            }

            for idx in punct:
                data[json_line["meta"]["URL"][35:]]["annotation"][idx]["punct"] = True

    return data

def find_punct(tweet):

    all_punct = []
    doc = nlp_big(tweet)
    for i, token in enumerate(doc):
        if token.pos_ == "PUNCT" or token.text == 'URL':
            all_punct.append(i)

    return all_punct

def open_entity_gold(file):
    data = {}
    with open(file, 'r') as f:
        for line in f:
            json_line = json.loads(line)
            if json_line["answer"] != "accept":
                    continue
            span_idxs = [(i["start"], i["end"]) for i in json_line["spans"]]
            data[json_line["meta"]["URL"][35:]] = {
                 "text": json_line["text"],
                 "entity_idx": span_idxs,
                 "tokens": [i["text"] for i in json_line["tokens"]],
            }
    return data

def remove_islands(tokens, labels):
    new_list = []
    consecutive_en = 0

    for i in range(len(tokens)):
        if labels[i] != 'D':
            consecutive_en += 1
        else:
            if consecutive_en > 4:
                for _ in range(consecutive_en): new_list.append('D')
            else:
                new_list = new_list + labels[i-consecutive_en:i]
            new_list.append('D')
            consecutive_en = 0

    if consecutive_en > 4:
        for _ in range(consecutive_en): new_list.append('D')
    elif consecutive_en > 0:
        new_list = new_list + labels[-consecutive_en:]

    return new_list

def identify_with_lingua(text, tokens_idx):
    char_output = ['D']*len(text)

    for result in detector.detect_multiple_languages_of(text):
        for char in range(result.start_index,result.end_index):
            if result.language.name == 'ENGLISH':
                char_output[char] = 'E'

    for word, indexes in tokens_idx.items():
        if len(set(char_output[indexes['start']:indexes['end']])) > 1:
            for i in range(indexes['start'], indexes['end']):
                char_output[i] = 'M'

    return char_output


def char_baseline(all_data):

    output = {}

    for id in tqdm(all_data):
        text, tokens, tokens_idx = all_data[id]["text"], all_data[id]["tokens"], all_data[id]["token_idx"]
        char_output = identify_with_lingua(text, tokens_idx)
        labels = []
        for token in tokens:

            token_char_outputs = char_output[tokens_idx[token]["start"]:tokens_idx[token]["end"]]

            if len(set(token_char_outputs)) > 1:
                raise Exception("Somehow this mixed token wasn't picked up on...")
            elif token_char_outputs[0] == 'E':
                labels.append('E')
            elif token_char_outputs[0] == 'M':
                labels.append('M')
            else:
                labels.append('D')

        labels = [label if i not in all_data[id]["punct"] else "P" for i, label in enumerate(labels)]

        output[id] = labels

    return output

def process_tweet(words_processed, TS, id):
    full_anno = TS.tongueswitcher_detect(words_processed)
    return full_anno, id

def rules_based(all_data, punct=True, flair_cache_file=""):

    test_data = {id: {"text": all_data[id]["text"], "date": "2023-03"} for id in all_data}


    with open('../data/cache/dictionaries.pkl', 'rb') as f:
        dictionaries = pickle.load(f)

    with open('../data/cache/affixes.pkl', 'rb') as f:
        affixes = pickle.load(f)

    data_loader = EasyDict({"data": {"dictionaries": dictionaries, "zenodo_tweets": test_data, "affixes": affixes}})

    with open('../configs/tongueswitcher_detection.jsonnet') as f:
        jsonnet_str = f.read()

    json_str = _jsonnet.evaluate_snippet('', jsonnet_str)
    config = json.loads(json_str)

    TS = TongueSwitcher(EasyDict(config), data_loader)
    data_dict = {id: {} for id in test_data}

    flair_tagger = SequenceTagger.load("flair/upos-multi")

    data_dict = {}

    count = 0

    with tqdm(total=len(all_data)) as pbar:
        with ThreadPoolExecutor() as executor:

            if os.path.isfile(flair_cache_file):
                with open(flair_cache_file, 'rb') as f:
                    input_rule_data = pickle.load(f)
            else:
                sentences = [Sentence(all_data[id]["text"]) for id in all_data.keys()]
                assert len(sentences) == len(all_data), f"length before Sentences: {len(all_data)}, length after: {len(sentences)}"

                flair_tagger.predict(sentences, mini_batch_size=16, verbose=True, return_probabilities_for_all_classes=True)
                assert len(sentences) == len(all_data), f"length before predict: {len(all_data)}, length after: {len(sentences)}"
                
                for sentence, id in tqdm(zip(sentences, all_data)):
                    original_sentence = all_data[id]["text"].strip()
                    if sentence.text.strip() != original_sentence:
                        continue

                input_rule_data = {}
                for i, id in tqdm(enumerate(all_data), desc="Checking token alignments"):

                    original_tokens = all_data[id]["tokens"]
                    flair_tokens = [token.text for token in sentences[i]]

                    

                    if original_tokens != flair_tokens:
                        flair_labels = [token.get_label("upos").value for token in sentences[i]]
                        new_pos_labels = get_new_tokenization_labels(original_tokens, flair_tokens, flair_labels)

                        flair_dists = [{token.tags_proba_dist["upos"][i].value: token.tags_proba_dist["upos"][i].score for i in range(len(token.tags_proba_dist["upos"]))} for token in sentences[i]]
                        new_flair_dists = get_new_tokenization_labels(original_tokens, flair_tokens, flair_dists)

                        assert len(new_pos_labels) == len(original_tokens), f"original_tokens:\n{original_tokens}\nflair_tokens:\n{flair_tokens}\nflair_labels:\n{flair_labels}"

                        words_processed = [{"token": token, "lan": "U", "pos": pos_label, "pos_dist": pos_dist} for token, pos_label, pos_dist in zip(original_tokens, new_pos_labels, new_flair_dists)]
                    else:
                        words_processed = [{"token": token.text, "lan": "U", "pos": token.get_label("upos").value, "pos_dist": {token.tags_proba_dist["upos"][i].value: token.tags_proba_dist["upos"][i].score for i in range(len(token.tags_proba_dist["upos"]))}} for token in sentences[i]]

                    input_rule_data[id] = words_processed

                with open(flair_cache_file, 'wb') as f:
                    pickle.dump(input_rule_data, f)
                
            futures = {executor.submit(process_tweet, input_rule_data[id], TS, id) for id in all_data}

            for future in as_completed(futures):
                id = future.result()[1]
                data_dict[id] = {"anno": future.result()[0], "text": all_data[id]["text"], "tokens": all_data[id]["tokens"]}
                pbar.update(1)

    if punct:
        all_labels = {}
        for id in data_dict:
            labels = [token["lan"] if i not in all_data[id]["punct"] else "P" for i, token in enumerate(data_dict[id]["anno"])]
            all_labels[id] = labels
    else:
        all_labels = {}
        for id in data_dict:
            labels = [token["lan"] for token in data_dict[id]["anno"]]
            all_labels[id] = labels

    return all_labels

def get_new_tokenization_labels(original_tokens, subword_tokens, labels):
    a2b, b2a = get_alignments(original_tokens, subword_tokens)

    subword_labels = []
    for label_indices in a2b:
        aligned_subwords = [labels[j] for j in label_indices]

        if not aligned_subwords:
            try: 
                aligned_subwords = [subword_labels[-1]]
            except:
                aligned_subwords = ['D']

        most_common = aligned_subwords[0]

        subword_labels.append(most_common)

    return subword_labels

def get_ngrams(word_list, num_of_ngrams):
    ngrams_dict = dict()
    for word in word_list:
        ngram_list = [''.join(ngram) for ngram in list(ngrams(word, 2)) + list(ngrams(word, 3))]
        for ngram in ngram_list:
            if ngram in ngrams_dict.keys():
                ngrams_dict[ngram] += 1
            else:
                ngrams_dict[ngram] = 1
    sorted_list = sorted(ngrams_dict.items(), key=lambda item: item[1],reverse=True)

    res_lst = [strng for strng, value in sorted_list[:num_of_ngrams]]
    return res_lst

def word2features(sent, i, most_freq_ngrams=[]):
    """
    :param sent: the sentence
    :param i: the index of the token in sent
    :param tags: the tags of the given sentence (sent)
    :return: the features of the token at index i in sent
    """
    word = sent[i]

    lower_word = word.lower()
    list_of_ngrams = list(ngrams(lower_word, 2)) + list(ngrams(lower_word, 3))
    list_of_ngrams = [''.join(ngram) for ngram in list_of_ngrams]


    features = {
        'word.lower()': word.lower(),
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word_with_digit': any(char.isdigit() for char in word) and word.isnumeric() is False,
        'word_pure_digit': word.isnumeric(),
        'word_with_umlaut': any(char in "üöäÜÖÄß" for char in word),
        'word_with_punct': any(char in string.punctuation for char in word),
        'word_pure_punct': all(char in string.punctuation for char in word),
        'frequent_en_word': lower_word in clfutil.FreqLists.EN_WORD_LIST,
        'frequent_de_word': lower_word in clfutil.FreqLists.DE_WORD_LIST,
        'frequent_ngrams_de': any(ngram in clfutil.MOST_COMMON_NGRAMS_DE for ngram in list_of_ngrams),
        'frequent_ngrams_en': any(ngram in clfutil.MOST_COMMON_NGRAMS_EN for ngram in list_of_ngrams),
        'is_in_emoticonlist': lower_word in clfutil.OtherLists.EMOTICON_LIST,
        'is_emoji': any(char in emoji.EMOJI_DATA for char in word),

        #derivation and flextion
        'D_Der_A_suff': any(lower_word.endswith(silbe) for silbe in list(itertools.chain.from_iterable(clfutil.FlexDeri.D_DER_A_suf_dict.values()))),
        'D_Der_N_suff': any(lower_word.endswith(silbe) for silbe in list(itertools.chain.from_iterable(clfutil.FlexDeri.D_DER_N_suf_dict.values()))),
        'D_Der_V_pref': any(lower_word.startswith(silbe) for silbe in clfutil.FlexDeri.D_DER_V_pref_list),
        'E_Der_A_suff': any(lower_word.endswith(silbe) for silbe in clfutil.FlexDeri.E_DER_A_suf_list),
        'E_Der_N_suff': any(lower_word.endswith(silbe) for silbe in list(itertools.chain.from_iterable(clfutil.FlexDeri.E_DER_N_suf_dict.values()))),
        'E_Der_V_pref': any(lower_word.startswith(silbe) for silbe in clfutil.FlexDeri.E_DER_V_pref_list),
        'D_Der_V_suff': any(lower_word.endswith(silbe) for silbe in list(itertools.chain.from_iterable(clfutil.FlexDeri.D_DER_V_suf_dict.values()))),
        'E_Der_V_suff': any(lower_word.endswith(silbe) for silbe in list(itertools.chain.from_iterable(clfutil.FlexDeri.E_DER_V_suf_dict.values()))),
        'D_Flex_A_suff': any(lower_word.endswith(silbe) for silbe in list(itertools.chain.from_iterable(clfutil.FlexDeri.D_FLEX_A_suf_dict.values()))),
        'D_Flex_N_suff': any(lower_word.endswith(silbe) for silbe in clfutil.FlexDeri.D_FLEX_N_suf_list),
        'D_Flex_V_suff': any(lower_word.endswith(silbe) for silbe in clfutil.FlexDeri.D_FLEX_V_suf_list),
        'E_Flex_A_suff': any(lower_word.endswith(silbe) for silbe in clfutil.FlexDeri.E_FLEX_A_suf_list),
        'E_Flex_N_suff': any(lower_word.endswith(silbe) for silbe in clfutil.FlexDeri.E_FLEX_N_suf_list),
        'E_Flex_V_suff': any(lower_word.endswith(silbe) for silbe in clfutil.FlexDeri.E_FLEX_V_suf_list),
        'D_Flex_V_circ': lower_word.startswith("ge") and (lower_word.endswith("en") or lower_word.endswith("t")),

        #NE:
        'D_NE_Demo_suff': any(lower_word.endswith(silbe) for silbe in clfutil.NELexMorph.D_NE_Demo_suff),
        'D_NE_Morph_suff': any(lower_word.endswith(silbe) for silbe in clfutil.NELexMorph.D_NE_Morph_suff),
        'E_NE_Demo_suff': any(lower_word.endswith(silbe) for silbe in clfutil.NELexMorph.E_NE_Demo_suff),
        'E_NE_Morph_suff': any(lower_word.endswith(silbe) for silbe in clfutil.NELexMorph.E_NE_Morph_suff),
        'O_NE_Morph_suff': any(lower_word.endswith(silbe) for silbe in clfutil.NELexMorph.O_NE_Morph_suff),
        'D_NE_parts': any(silbe in lower_word for silbe in clfutil.NELexMorph.D_NE_parts),
        'E_NE_parts': any(silbe in lower_word for silbe in clfutil.NELexMorph.E_NE_parts),
        'O_NE_parts': any(lower_word.endswith(silbe) for silbe in clfutil.NELexMorph.O_NE_suff),

        #entity lists
        'D_NE_REGs': any(w in lower_word for w in clfutil.NELists.D_NE_REGs)
                     or lower_word in clfutil.NELists.D_NE_REGs_abbr,
        'E_NE_REGs': any(w in lower_word for w in clfutil.NELists.E_NE_REGs)
                     or lower_word in clfutil.NELists.E_NE_REGs_abbr,
        'O_NE_REGs': any(w in lower_word for w in clfutil.NELists.O_NE_REGs)
                     or lower_word in clfutil.NELists.O_NE_REGs_abbr
                     or any(lower_word.startswith(w) for w in clfutil.NELists.O_REG_demonym_verisons),

        'D_NE_ORGs': lower_word in clfutil.NELists.D_NE_ORGs,
        'E_NE_ORGs': lower_word in clfutil.NELists.E_NE_ORGs,
        'O_NE_ORGs': lower_word in clfutil.NELists.O_NE_ORGs,

        'D_NE_VIPs': lower_word in clfutil.NELists.D_NE_VIPs,
        'E_NE_VIPs': lower_word in clfutil.NELists.E_NE_VIPs,
        'O_NE_VIPs': lower_word in clfutil.NELists.O_NE_VIPs,

        'D_NE_PRESS': lower_word in clfutil.NELists.D_NE_PRESS,
        'E_NE_PRESS': lower_word in clfutil.NELists.E_NE_PRESS,
        'O_NE_PRESS': lower_word in clfutil.NELists.O_NE_PRESS,

        'D_NE_COMPs': lower_word in clfutil.NELists.D_NE_COMPs,
        'E_NE_COMPs': lower_word in clfutil.NELists.E_NE_COMPs,
        'O_NE_COMPs': lower_word in clfutil.NELists.O_NE_COMPs,

        'NE_MEASURE': any(w in lower_word for w in clfutil.NELists.NE_MEASURE),

        'D_CULT': any(w in lower_word for w in clfutil.CultureTerms.D_CULT),
        'E_CULT': any(w in lower_word for w in clfutil.CultureTerms.E_CULT),
        'O_CULT': any(w in lower_word for w in clfutil.CultureTerms.O_CULT),

        'D_FuncWords': lower_word in clfutil.FunctionWords.deu_function_words,
        'E_FuncWords': lower_word in clfutil.FunctionWords.eng_function_words,

        'Interj_Word': lower_word in clfutil.OtherLists.Interj_Words,

        'URL': any(lower_word.startswith(affix) for affix in clfutil.OtherLists.URL_PREF) or any(lower_word.endswith(affix) for affix in clfutil.OtherLists.URL_SUFF) or any(affix in lower_word for affix in clfutil.OtherLists.URL_INFIX)
    }

    for ngram in most_freq_ngrams:
        features[ngram] = ngram in list_of_ngrams

    if i > 0:
        pass
    else:
        features['BOS'] = True

    if i == len(sent) - 1:
        features['EOS'] = True

    return features

def sent2features(sent, most_freq_ngrams=[]):
    """
    This function returns a list of features of each token in the given sentence (and using the corresponding tags)
    """
    return [word2features(sent, i, most_freq_ngrams) for i in range(len(sent))]

def denglisch_crf(all_data, train_file="../data/denglisch/Manu_corpus_collapsed.csv"):

    train_corpus = Corpus(train_file)

    # Find most frequent N-grams in training data.
    word_list, _ = train_corpus.get_tokens()
    most_freq_ngrams = get_ngrams(word_list, 200)

    model_file = "../data/denglisch/model_collapsed.pkl"

    if os.path.isfile(model_file):
        with open(model_file, "rb") as f:
            crf = pickle.load(f)
    else:
        raise Exception("No CRF model file found :(")

    # Predict tags for new data. We extract indices along with the tokens so we can update the tags later.
    # print("start predict")

    test_data = {id: all_data[id]["tokens"] for id in all_data}

    ids = list(test_data.keys())
    tweets = list(test_data.values())
    
    X_new = [sent2features(t, most_freq_ngrams) for t in tweets]
    y_new = crf.predict(X_new)

    output = {}

    for i, (id, y) in enumerate(zip(ids, y_new)):
        labels = []
        for j, t in enumerate(y):
            if t == 'E' or t == 'SE':
                labels.append('E')
            elif t == 'M':
                labels.append('M')
                # print(tweets[i][j])
            else:
                labels.append('D')

        labels = [label if i not in all_data[id]["punct"] else "P" for i, label in enumerate(labels)]

        output[id] = labels

    return output

def denglisch_rules(file_name, out_file, flair_cache_file=""):
    
    corpus = Corpus(file_name)
    idxs, toks, tags = corpus.get_sentences(index=True)

    for i in range(len(toks)):
        toks[i] = ['!' if t == '' else t.replace("’", "'").replace("”", "'").replace("“", "'").replace("„", "'").replace("―", "-").replace("–", "-").replace("…", "...").replace("`", "'").replace("‘", "'").replace("—", "-").replace("´", "'") for t in toks[i]]

    tokenization = {' '.join(t): t for t in toks}

    max_length = 100
    long_sequences = {}


    input_data = {str(id): {'text': ' '.join(t), 'tokens': t, 'date': '2023-03'} for id, t in enumerate(toks)}

    all_tags = rules_based(input_data, punct=False, flair_cache_file=flair_cache_file) 

    for id, i in enumerate(idxs):
        assert len(all_tags[str(id)]) == len(toks[id]), toks[id]
        new_tags = all_tags[str(id)]
        corpus.set_tags(i, new_tags)


    corpus.to_csv(out_file)

def replace_emojis_with_X(tokens):
    emoj = re.compile("["
        u"\U0001F600-\U0001F64F"  
        u"\U0001F300-\U0001F5FF"  
        u"\U0001F680-\U0001F6FF"  
        u"\U0001F1E0-\U0001F1FF"  
        u"\U00002500-\U00002BEF"  
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  
        u"\u3030"
                      "]+", re.UNICODE)
    return ['X' if re.match(emoj, token) else token for token in tokens]

def find_longest_list(list_of_lists):
    longest_list = None
    max_length = 0

    for sublist in list_of_lists:
        if len(sublist) > max_length:
            longest_list = sublist
            max_length = len(sublist)

    return longest_list

def prompt_based(tweet, tokens, model="gpt-4"):

    empty_list = str([(t, '') for t in tokens])

    prompt = f"""Sentence: {tweet}
Task: Fill in the following list of words and their labels by identifying each of the words in the sentence as English ('E'), Mixed ('M') or German ('G') . Punctuation should be the same language as its surrounding associated words. Mixed words switch between English and German within the word. Only use the tags 'E', 'M' or 'G'.
Fill in: {empty_list}
"""

    if model == "text-davinci-003":
        response = openai.Completion.create(
            model=model,
            prompt=prompt,
            temperature=0.6,
            max_tokens=len(empty_list)*2
            )

        labels = response["choices"][0]["text"].replace("\n", "").replace("JSON object: ", "").replace("JSON object", "").replace("Answer: ", "").replace("Answer:", "").replace("Tags: ", "")
    else:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6,
            max_tokens=len(empty_list)*2
            )

        labels = response["choices"][0]["message"]["content"]

    cost = 0.03*response["usage"]["prompt_tokens"]/1000 + 0.06*response["usage"]["completion_tokens"]/1000

    tags = eval(labels)

    tags = [x[1] for x in tags]

    tags = list(tags)

    assert len(tags) == len(tokens)

    return tags, cost

def prompt_gpt(all_data, prompt_file):
    if prompt_file.exists():
        with open(prompt_file, "rb") as f:
            prompt_results = pickle.load(f)
    else:
        prompt_results = {}

    replace_dict = {'G': 'D', 'ENGLISH': 'E', 'MIXED': 'M', '': 'D'}
    prompt_results = {k: [replace_dict.get(i, i) for i in v] for k,v in prompt_results.items()}

    prediction_labels = {}

    for id in prompt_results:
        labels = [label if i not in all_data[id]["punct"] else "P" for i, label in enumerate(prompt_results[id])]
        prediction_labels[id] = labels


    missing_ids = [id for id in all_data if id not in prompt_results]

    with tqdm(total=len(missing_ids)) as pbar:
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_id = {executor.submit(prompt_based, all_data[id]["text"], all_data[id]["tokens"],model="gpt-4"): id for id in missing_ids}
            for future in as_completed(future_to_id):
                id = future_to_id[future]
                test_labels, cost = future.result()
                total_cost += cost
                print(total_cost)
                prompt_results[id] = test_labels
                with open(prompt_file, "wb") as f:
                    pickle.dump(prompt_results, f)
                pbar.update(1)
    return prediction_labels


def mbert_label(all_data, model_path, punct=True):

    print(f"Labelling with mBERT with {model_path}")

    output_labels = {}
    task = "token-classification"
    mbert_token_classification = pipeline(task, model=model_path, tokenizer=model_path)
    denglisch_mapping = {'SO': 'D', 'SD': 'D', 'SE': 'E'}
    def process_tweet(id):
        input_text = all_data[id]["text"]
        classification_output = mbert_token_classification(input_text)
        mbert_subword_tokens = [token["word"] for token in classification_output]
        mbert_subword_labels = [token["entity"] for token in classification_output]
        original_tokens = all_data[id]["tokens"]

        mbert_word_labels = get_subword_labels(mbert_subword_tokens, original_tokens, mbert_subword_labels)
        mbert_word_labels = [denglisch_mapping[label] if label in denglisch_mapping else label for label in mbert_word_labels]
        if punct:
            mbert_word_labels = [label if i not in all_data[id]["punct"] else "P" for i, label in enumerate(mbert_word_labels)]

        assert len(mbert_word_labels) == len(original_tokens)

        return id, mbert_word_labels

    with ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_tweet, all_data.keys()), total=len(all_data)))

    for id, mbert_word_labels in results:
        output_labels[id] = mbert_word_labels
            
    return output_labels

def get_subword_labels(a, b, a_labels):
    a2b, b2a = get_alignments(a, b)

    # Assign labels to subwords
    b_labels = []
    most_common = 'D'

    for i, label_indices in enumerate(b2a):

        aligned_subwords = []

        if label_indices:
            for j in label_indices:
                if j < len(a_labels):
                    aligned_subwords.append(a_labels[j])

        if not aligned_subwords:
            aligned_subwords = [most_common]

        # if len(set(aligned_subwords)) > 1:
        #     most_common = 'M'
        # else:
        #     most_common = aligned_subwords[0]
        most_common = max(set(aligned_subwords), key=aligned_subwords.count)

        b_labels.append(most_common)
    
    return b_labels

def identification(data, method, cache_file="", reset=False, save=False, **kwargs):

    cache_file = Path(cache_file)
    if cache_file.exists() and not reset:
        with open(cache_file, "rb") as f:
            prediction_labels = pickle.load(f)
    else:
        prediction_labels = globals()[method](data, **kwargs)
        if save:
            with open(cache_file, "wb") as f:
                pickle.dump(prediction_labels, f)

    return prediction_labels

def map_to_bio(labels):
    last_lan = None
    bio_labels = []
    for label in labels:
        if label != 'E':
            bio_labels.append('O')
        elif label == last_lan:
            bio_labels.append('I-' + label)
        else:
            bio_labels.append('B-' + label)
        last_lan = label
    
    return bio_labels

def short_map_to_bio(labels):
    last_lan = None
    bio_labels = []
    entity = []
    for label in labels:
        if label != 'E':
            if entity:
                if len(entity) >=2 and len(entity) <= 4:
                    bio_labels += entity
                else:
                    bio_labels += ['O']*len(entity)
            entity = []
            bio_labels.append('O')
        else:
            if entity:
                entity.append('I-' + label)
            else:
                entity.append('B-' + label)

    if len(entity) >=2 and len(entity) <= 4:
        bio_labels += entity
    else:
        bio_labels += ['O']*len(entity)
        
    return bio_labels

def cs_token_f1_score(all_data, file_name, baseline = False, tagger=False, rules=False, denglisch=False, prompt=False, mbert=False, model_path = "", islands=False, reset=False):

    if baseline:
        prediction_labels = identification(all_data, "char_baseline", file_name)
    elif rules:
        prediction_labels = identification(all_data, "rules_based", file_name, flair_cache_file="../data/cache/rules_flair_cache.pkl")
    elif tagger:
        prediction_labels = identification(all_data, "spacy_tagger", file_name)
    elif denglisch:
        prediction_labels = identification(all_data, "denglisch_crf", file_name)
    elif mbert:
        prediction_labels = identification(all_data, "mbert_label", file_name, model_path=model_path)
    elif prompt:
        prediction_labels = prompt_gpt(all_data, prompt_file)
            


    true_labels = [all_data[id]["labels"] for id in all_data]
    prediction_labels = [prediction_labels[id] for id in all_data]

    for true, pred in zip(true_labels, prediction_labels):
        assert len(true) == len(pred), f"\n{true}\n{pred}"

    true_labels = [label for tweet_labels in true_labels for label in tweet_labels]
    prediction_labels = [label for tweet_labels in prediction_labels for label in tweet_labels]

    true_labels = [label for label in true_labels if label != "P"]
    prediction_labels = [label for label in prediction_labels if label != "P"]

    results = {}

    precision, recall, f1, support = precision_recall_fscore_support(true_labels, prediction_labels, labels=['D', 'E', 'M'], zero_division=0.0)

    for l, p, r, f, s in zip(['D', 'E', 'M'], precision, recall, f1, support):
        results[l] = {"P": p, "R": r, "F1": f, "support": s}

    p, r, f, s = precision_recall_fscore_support(true_labels, prediction_labels, average='micro', zero_division=0.0) 
    results["F1"] = {"P": p, "R": r, "F1": f, "support": s}
    return results

def cs_entity_f1_score(all_data, file_name, baseline = False, tagger=False, rules=False, denglisch=False, prompt=False, mbert=False, model_path = "", island=False):


    if baseline:
        prediction_labels = identification(all_data, "char_baseline", file_name)
    elif rules:
        prediction_labels = identification(all_data, "rules_based", file_name)
    elif tagger:
        prediction_labels = identification(all_data, "spacy_tagger", file_name)
    elif denglisch:
        prediction_labels = identification(all_data, "denglisch_crf", file_name)
    elif mbert:
        prediction_labels = identification(all_data, "mbert_label", file_name, model_path=model_path)
    elif prompt:
        if prompt_file.exists():
            with open(prompt_file, "rb") as f:
                prompt_results = pickle.load(f)
        else:
            prompt_results = {}

        replace_dict = {'G': 'D', 'ENGLISH': 'E', 'MIXED': 'M'}
        prompt_results = {k: [replace_dict.get(i, i) for i in v] for k,v in prompt_results.items()}
        prediction_labels = {}

        for id in prompt_results:
            labels = [label if i not in all_data[id]["punct"] else "P" for i, label in enumerate(prompt_results[id])]
            prediction_labels[id] = labels
            
    true_labels = [all_data[id]["labels"] for id in all_data]
    prediction_labels = [prediction_labels[id] for id in all_data]
    
    for true, pred in zip(true_labels, prediction_labels):
        assert len(true) == len(pred), f"{true}\n{pred}"

    true_labels = [[element for element in sublist if element != "P"] for sublist in true_labels]
    prediction_labels = [[element for element in sublist if element != "P"] for sublist in prediction_labels]

    entity_true_labels = []
    entity_prediction_labels = []
    for true, pred in zip(true_labels, prediction_labels):
        if island:
            entity_true_labels.append(short_map_to_bio(true))
            entity_prediction_labels.append(short_map_to_bio(pred))          
        else:
            entity_true_labels.append(map_to_bio(true))
            entity_prediction_labels.append(map_to_bio(pred))

    return classification_report(entity_true_labels, entity_prediction_labels, output_dict=True, mode="strict")["E"]

def print_latex_table(all_results, header_map, labels=['D', 'E', 'M', 'F1'], metrics=['P', 'R', 'F1'], entity=False):

    print("\\begin{table*}[]")
    print(''.join(["\\begin{tabular}{l"] + ["ccc"]*len(labels)) + "}")
    print("\\toprule")
    print(''.join([" & \multicolumn{3}{c}{\\textbf{" + header_map[label] + "}} " for label in labels]) + "\\\\")
    
    if entity:
        print(''.join([" & P     & R     & F-e   "]*len(labels)) +   "\\\\ \\midrule")
    else:
        print(''.join([" & P     & R     & F-t   "]*len(labels)) +   "\\\\ \\midrule")

    for key, result in all_results.items():
        print(f"\\textit{{{key}}} ", end="")

        for label in labels:
            print(f"& {100*result[label][metrics[0]]:.2f}  & {100*result[label][metrics[1]]:.2f}  & {100*result[label][metrics[2]]:.2f}  ", end="")

        print("\\\\")

        if key == "denglisch CRF":
            print("\\midrule")
        
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table*}")

def split_dict_randomly(input_dict, test_ratio):
    keys = list(input_dict.keys())
    random.shuffle(keys)
    split_index = int(len(keys) * test_ratio)

    train_keys = keys[split_index:]
    test_keys = keys[:split_index]

    train_set = {key: input_dict[key] for key in train_keys}
    test_set = {key: input_dict[key] for key in test_keys}

    return train_set, test_set

def main():

    all_data = {}

    with open(tongueswitcher_testset_dir, 'r') as f:
        for i, line in enumerate(f):
            json_line = json.loads(line)
            punct = [idx for idx, token in enumerate(json_line["annotation"]) if "punct" in token]
            labels = [token["lan"] if i not in punct else "P" for i, token in enumerate(json_line["annotation"])]
            all_data[str(i)] = {
                "labels": labels,
                "text": json_line["text"],
                "punct": punct,
                "tokens": [token["token"] for token in json_line["annotation"]]    
            }

    all_results = {
        # "Lingua": cs_token_f1_score(all_data, baseline_file, baseline=True),
        # "GPT-4": cs_token_f1_score(all_data, prompt_file, prompt=True),
        # "denglisch CRF": cs_token_f1_score(all_data, denglisch_file, denglisch=True),
        # "BERT": cs_token_f1_score(all_data, bert_file, mbert=True, model_path = bert_model),
        # "mBERT": cs_token_f1_score(all_data, mbert_file, mbert=True, model_path = mbert_model),
        # "gBERT": cs_token_f1_score(all_data, gbert_file, mbert=True, model_path = gbert_model),
        # "tsBERT": cs_token_f1_score(all_data, tsbert_file, mbert=True, model_path = tsbert_model),
        "TongueSwitcher": cs_token_f1_score(all_data, tongueswitcher_file, rules=True),
    }

    print_latex_table(all_results,header_map={"D": "German", "E": "English", "M": "Mixed", "F1": "Overall"})

    all_results = {
        # "Lingua": {},
        # "GPT-4": {},
        "denglisch CRF": {},
        # "BERT": {},
        # "mBERT": {},
        # "gBERT": {},
        "tsBERT": {},
        "TongueSwitcher": {},
    }

    # all_results["Lingua"]["island"] = cs_entity_f1_score(all_data, baseline_file, baseline=True)
    # all_results["GPT-4"]["island"] = cs_entity_f1_score(all_data, prompt_file, prompt=True)
    all_results["denglisch CRF"]["island"] = cs_entity_f1_score(all_data, denglisch_file, denglisch=True)
    # all_results["BERT"]["island"] = cs_entity_f1_score(all_data, bert_file, mbert=True, model_path = bert_model)
    # all_results["mBERT"]["island"] = cs_entity_f1_score(all_data, mbert_file, mbert=True, model_path = mbert_model)
    # all_results["gBERT"]["island"] = cs_entity_f1_score(all_data, gbert_file, mbert=True, model_path = gbert_model)
    all_results["tsBERT"]["island"] = cs_entity_f1_score(all_data, tsbert_file, mbert=True, model_path = tsbert_model)
    all_results["TongueSwitcher"]["island"] = cs_entity_f1_score(all_data, tongueswitcher_file, rules=True)

    # all_results["Lingua"]["short island"] = cs_entity_f1_score(all_data, baseline_file, baseline=True, island=True)
    # all_results["GPT-4"]["short island"] = cs_entity_f1_score(all_data, prompt_file, prompt=True, island=True)
    all_results["denglisch CRF"]["short island"] = cs_entity_f1_score(all_data, denglisch_file, denglisch=True, island=True)
    # all_results["BERT"]["short island"] = cs_entity_f1_score(all_data, bert_file, mbert=True, model_path = bert_model, island=True)
    # all_results["mBERT"]["short island"] = cs_entity_f1_score(all_data, mbert_file, mbert=True, model_path = mbert_model, island=True)
    # all_results["gBERT"]["short island"] = cs_entity_f1_score(all_data, gbert_file, mbert=True, model_path = gbert_model, island=True)
    all_results["tsBERT"]["short island"] = cs_entity_f1_score(all_data, tsbert_file, mbert=True, model_path = tsbert_model, island=True)
    all_results["TongueSwitcher"]["short island"] = cs_entity_f1_score(all_data, tongueswitcher_file, rules=True, island=True)

    print_latex_table(all_results, header_map = {"island": "Island", "short island": "Short Island"}, labels = ["island", "short island"], metrics = ["precision", "recall", "f1-score"], entity=True) 

    denglisch_rules("../data/denglisch/Manu_corpus_collapsed.csv", out_file = "./data/resources/denglisch_labelled_with_tongueswitcher.csv")

if __name__ == '__main__':
    main()