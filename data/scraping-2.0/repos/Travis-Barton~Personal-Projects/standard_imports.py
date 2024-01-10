import platform
import pandas as pd
import numpy as np
import spacy
from matplotlib.pyplot import plot
import matplotlib.pyplot as plt
import os
import time
import sys
from itertools import islice
import sklearn.metrics as metrics
import datetime
# symphony imports
import os
import secret as sec
from typing import List
import json
import transformers as tr

os.environ['AYASDI_APISERVER'] = 'https://platform.ayasdi.com/workbench/'
import ayasdi
import ayasdi.core as ac
import ayasdi.core.models as acm
from ayasdi.core.api import Api
from ayasdi.core.unsupervised_analysis import NameFilterSpec, AutoNetworkSpec, AutoAnalysis
from ayasdi.core.source_subset import SourceSubset
import urllib3
import random as rd
import openai
import praw

urllib3.disable_warnings()
openai.api_key = sec.open_ai_key
reddit = praw.Reddit(user_agent='Comment Extraction (by /u/symphonyai_ds)',
                     client_id=sec.reddit_client_id, client_secret=sec.reddit_client_secret,
                     username=sec.reddit_username, password=sec.reddit_password)
nlp = spacy.load('en_core_web_lg')
openai_engine_dict = {'curie': "text-similarity-curie-001",
                      'davinci': 'text-similarity-davinci-001',
                      'babbage': 'text-similarity-babbage-001',
                      'ada': 'text-similarity-ada-001'
                      }

R = "\033[1;31m"  # RED
G = "\033[1;32m"  # GREEN
Y = "\033[1;33m"  # Yellow
B = "\033[1;34m"  # Blue
N = "\033[0m"  # Reset
BLACK = '\033[0;30m'
RED = '\033[0;31m'
GREEN = '\033[0;32m'
BROWN = '\033[0;33m'
BLUE = '\033[0;34m'
PURPLE = '\033[0;35m'
CYAN = '\033[0;36m'
LIGHT_GREY = '\033[0;37m'
DARK_GREY = '\033[1;30m'
LIGHT_RED = '\033[1;31m'
LIGHT_GREEN = '\033[1;32m'
YELLOW = '\033[1;33m'
LIGHT_BLUE = '\033[1;34m'
LIGHT_PURPLE = '\033[1;35m'
LIGHT_CYAN = '\033[1;36m'
WHITE = '\033[1;37m'
DEFAULT = '\033[0m'
REVERSE = '\033[7m'
RESET = '\033[0m'
UNDERLINE = '\033[4m'
DIM = '\033[2m'
BLINK = '\033[5m'
# Set the terminal's background ANSI color to black.
ON_BLACK = '\033[40m'
# Set the terminal's background ANSI color to red.
ON_RED = '\033[41m'
# Set the terminal's background ANSI color to green.
ON_GREEN = '\033[42m'
# Set the terminal's background ANSI color to yellow.
ON_YELLOW = '\033[43m'
# Set the terminal's background ANSI color to blue.
ON_BLUE = '\033[44m'
# Set the terminal's background ANSI color to magenta.
ON_MAGENTA = '\033[45m'
# Set the terminal's background ANSI color to cyan.
ON_CYAN = '\033[46m'
# Set the terminal's background ANSI color to white.
ON_WHITE = '\033[47m'
ON_BLUE = '\033[44m'

ON_GRAY232 = '\033[48;5;232m'
ON_GRAY233 = '\033[48;5;233m'
ON_GRAY234 = '\033[48;5;234m'
ON_GRAY235 = '\033[48;5;235m'
ON_GRAY236 = '\033[48;5;236m'


# if you want 256 colors in bash use params 0 - 255
# function EXT_COLOR () { echo -ne "\033[38;5;$1m"; }

class Pcolors:
    """
    Example:
        print(Pcolors.BLUE + 'this is blue text' + Pcolors.N)  # the N is used to reset the color
    """
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    MAGENTA = '\u001b[35m'
    R = "\033[1;31m"  # RED
    G = "\033[1;32m"  # GREEN
    Y = "\033[1;33m"  # Yellow
    B = "\033[1;34m"  # Blue
    N = "\033[0m"  # Reset
    M = '\u001b[35;1m'  # Bright magenta
    YELLOW = "\033[33;1m"
    BLACK = '\033[0;30m'
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    BROWN = '\033[0;33m'
    BLUE = '\033[0;34m'
    PURPLE = '\033[0;35m'
    CYAN = '\033[0;36m'
    LIGHT_GREY = '\033[0;37m'
    DARK_GREY = '\033[1;30m'
    LIGHT_RED = '\033[1;31m'
    LIGHT_GREEN = '\033[1;32m'
    YELLOW = '\033[1;33m'
    LIGHT_BLUE = '\033[1;34m'
    LIGHT_PURPLE = '\033[1;35m'
    LIGHT_CYAN = '\033[1;36m'
    WHITE = '\033[1;37m'
    DEFAULT = '\033[0m'
    REVERSE = '\033[7m'
    RESET = '\033[0m'
    UNDERLINE = '\033[4m'
    DIM = '\033[2m'
    BLINK = '\033[5m'
    # Set the terminal's background ANSI color to black.
    ON_BLACK = '\033[40m'
    # Set the terminal's background ANSI color to red.
    ON_RED = '\033[41m'
    # Set the terminal's background ANSI color to green.
    ON_GREEN = '\033[42m'
    # Set the terminal's background ANSI color to yellow.
    ON_YELLOW = '\033[43m'
    # Set the terminal's background ANSI color to blue.
    ON_BLUE = '\033[44m'
    # Set the terminal's background ANSI color to magenta.
    ON_MAGENTA = '\033[45m'
    # Set the terminal's background ANSI color to cyan.
    ON_CYAN = '\033[46m'
    # Set the terminal's background ANSI color to white.
    ON_WHITE = '\033[47m'
    ON_BLUE = '\033[44m'
    ON_GRAY232 = '\033[48;5;232m'
    ON_GRAY233 = '\033[48;5;233m'
    ON_GRAY234 = '\033[48;5;234m'
    ON_GRAY235 = '\033[48;5;235m'
    ON_GRAY236 = '\033[48;5;236m'


def get_subreddit_data(subreddit, kind='all', drop_duplicates=None, limit=1000, quarantine: bool = False):
    if quarantine:
        reddit.subreddit(subreddit).quaran.opt_in()
        subred = reddit.subreddit(subreddit)
    else:
        subred = reddit.subreddit(subreddit)
    command_table = {'new': subred.new, 'top': subred.top, 'rising': subred.rising,
                     'controversial': subred.controversial}
    invert_command_table = {v: k for k, v in command_table.items()}
    data = pd.DataFrame(columns=['post_id', 'title', 'body', 'tag', 'created_on', 'author', 'score', 'up_ratio', 'url',
                                 'source'])
    if kind.lower() == 'all':
        kind = [subred.rising, subred.top, subred.new, subred.controversial]
    else:
        if isinstance(kind, list):
            kind = [command_table[k] for k in kind]
        else:
            kind = [command_table[kind]]
    for k in kind:
        for sub in k(limit=limit):
            ids = sub.id
            if sub.author is None:
                author = 'DELETED'
            else:
                author = sub.author.name
            title = sub.title
            body = sub.selftext
            if sub.link_flair_text is None:
                tag = None
            else:
                tag = sub.link_flair_text
            score = sub.score
            up_ratio = sub.upvote_ratio
            url = sub.url
            created_on = datetime.datetime.fromtimestamp(sub.created_utc)
            new_data = pd.DataFrame([ids, title, body, tag, created_on, author, score, up_ratio, url,
                                     invert_command_table[k]]).T
            new_data.columns = data.columns
            data = pd.concat([data, new_data], axis=0)

    if drop_duplicates is not None:
        if drop_duplicates not in data.columns:
            raise Exception(f'must specify duplicated by column in data\n{data.columns}')
        data = data.drop_duplicates(subset=[drop_duplicates])
    data['subreddit'] = subreddit
    return data.reset_index(drop=True)


def get_embedding(text, engine='curie'):
    if engine.lower() == 'spacy':
        return nlp(text).vector
    text = text.replace("\n", " ")
    return openai.Embedding.create(input=[text], engine=openai_engine_dict[engine])['data'][0]['embedding']


def query_edit(instructions: str, text: str, heading: str = '', n: int = 1, engine: str = 'code',
               temperature: float = 0, top_p: float = 1):
    """
    Query openai for edits. The engine can be either 'code' or 'text'.
    :param instructions: The instructions for the edit.
    :param text: The text to be edited.
    :param heading: The heading of the text.
    :param n: The number of edits to be generated.
    :param engine: The engine to be used.
    :param temperature: The temperature for the edit.
    :param top_p: The top_p for the edit.
    :return: A list of edits.
    """
    if engine == 'code':
        engine = "code-davinci-edit-001"
    elif engine == 'text':
        engine = "text-davinci-edit-001"

    response = openai.Edit.create(
        engine=engine,
        input=heading + text,
        instruction=instructions,
        temperature=temperature,
        top_p=top_p
    )
    edits: List[str] = []
    for i in range(n):
        edits.append(response["choices"][0]["text"])
    return edits


def query_completion(heading: str, text: str, max_tokens: int = 50, temperature: float = 0.7, top_p: float = 0.9,
                     n: int = 1, engine: str = 'code', stop_token: list = None):
    """
    Query openai for completions.
    :param engine: what engine to use for the completion. if code, then it will use "code-davinci-002", if text, then it
                   will use "text-davinci-002", otherwise it will use the engine supplied.
    :param heading: The heading of the text.
    :param text: The text to be completed.
    :param max_tokens: The maximum number of tokens to be generated.
    :param temperature: The temperature of the sampling.
    :param top_p: The top p of the sampling.
    :param n: The number of completions to be generated.
    :return: A list of completions.
    """
    if engine == 'code':
        engine = "code-davinci-002"
    elif engine == 'text':
        engine = "text-davinci-002"

    completions = []
    for i in range(n):
        completion = openai.Completion.create(
            engine=engine,
            prompt=heading + text,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            n=1,
            stream=False,
            logprobs=None,
            stop=stop_token,
        )
        completions.append(completion["choices"][0]["text"].replace(heading, ""))
    return completions


def check_token_count(prompt: str):
    """
    check_token_count: takes a string and returns the number of tokens based on the tokenizer gpt-3-encoder
    :param prompt: a string
    :return: the number of tokens in the prompt
    """
    tokenizer = tr.GPT2Tokenizer.from_pretrained('gpt2')
    return len(tokenizer.encode(prompt))


def make_examples(prompts, labels, meta_data=None, save=False, filename='examples'):
    """
    make_examples: takes a list of prompts and labels and returns a list of examples
    """
    examples = []
    if meta_data is None:
        meta_data = [{} for i in range(len(labels))]
    for i in range(len(prompts)):
        examples.append({"text": prompts[i], "label": labels[i], "meta": meta_data[i]})
    if save:
        with open(f"{filename}.jsonl", "w") as f:
            for example in examples:
                f.write(json.dumps(example) + "\n")
    return examples


def make_file(filename):
    openai.File.create(file=open(filename), purpose='classifications')
    return [i for i in openai.File.list()['data'] if i['filename'] == filename][0]['id']


def classify(examples: list, file: str, collect_pandas: bool = True, save: bool = False, filename: str = 'examples',
             model: str = 'curie', search_model: str = 'ada') -> list:
    """
    classify: takes a list of examples and returns a list of predictions
    """
    predictions = []
    for i in range(len(examples)):
        predictions.append(openai.Classification.create(
            file=file,
            query=examples[i],
            search_model=search_model,
            model=model,
            max_examples=1
        ))
    if save:
        with open(f"{filename}.jsonl", "w") as f:
            for prediction in predictions:
                f.write(json.dumps(prediction) + "\n")

    # if collect_pandas:
    #     return_pandas = pd.DataFrame(columns=['text', 'label', 'score', 'document', ])
    #
    return predictions

"""
Tests for openai functions above:
prompts = ['http://www.amazon.com', 'SUPPLY CHIMP', 'http://www.Alliedelec.com', 'MSC INDUSTRIAL SUPPLY CO, INC',
           'W.W.GRAINGER, INC.', 'http://www.Grainger.com', 'COMPUTER SYKES (DBA SOFTWAREMORE)', 
           'Allied Electronics, Inc.', 'WWT', 'http://www.wwt.com', 'GovConnection']
labels = ['Amazon', 'Supply Chimp', 'Allied Electronics', 'MSC Industrial Supply', 'Grainger', 'Grainger', 
          'Computer Sykes', 'Allied Electronics', 'WWT', 'WWT', 'GovConnection']
res = make_examples(prompts, labels, save=True)
file_id = make_file('examples.jsonl')
print(openai.File.list())
print(file_id)
test = ['VALVE,GLOBE', 'RESISTOR,FIXED,COMPOSITION', 'nan', 'nan', 'nan', 'SHIELD ASSEMBLY,PROTECTIVE',
        'VALVE,LINEAR,DIRECTIONAL CONTROL', 'HOSE,NONMETALLIC', 'HOSE,NONMETALLIC',
        'COUPLING HALF,QUICK DISCONNECT']
predictions = classify(test, file_id, save=True, filename='predictions')
print(predictions)
"""

def time_it(f):
    """
    use as decorator above functions to print their timing every time they are called.

    """
    time_it.active = 0

    def tt(*args, **kwargs):
        time_it.active += 1
        t0 = time.time()
        tabs = '\t' * (time_it.active - 1)
        name = f.__name__
        print(Pcolors.HEADER + '{tabs}Executing <{name}>'.format(tabs=tabs, name=name) + Pcolors.N)
        res = f(*args, **kwargs)
        print(Pcolors.CYAN + '{tabs}Function <{name}> execution time: {time:.3f} seconds'.format(
            tabs=tabs, name=name, time=time.time() - t0) + Pcolors.N)
        time_it.active -= 1
        return res

    return tt


def codes_done(title='code complete', msg='', voice=False, speaker='Daniel'):
    """
    Example:
        codes_done('this is main text', 'this is subtext', True)
    Args:
        title: main text
        msg: sub test
        voice: (bool) should it speak
        speaker: use list of osx speaker voices

    Returns:

    """
    os.system("osascript -e 'display notification \"{}\" with title \"{}\"'".format(msg, title))
    if voice and (platform.system() in ['Linux', 'Darwin']):
        os.system(f"say -v {speaker} {title + ',' + msg}")


#
# def get_pa(wf):
#     e_wf_fund, e_wf_fund_pa = pq.get_frequency_content(wf, frequency=60, sample_rate=7812.5)
#     e_wf_total = pq.get_total_frequency_content(wf, frequency=60, sample_rate=7812.5)[0]
#     e_wf_total_fund = e_wf_total / e_wf_fund
#     e_wf_ref_pa = [2 * np.pi * xx * 60 / 7812.5 for xx in range(len(e_wf_fund_pa))]
#     e_wf_fund_pa_ref = [norm_angle if norm_angle < 180 else norm_angle - 360
#                         for norm_angle in ((e_wf_fund_pa - e_wf_ref_pa - np.pi) % (2 * np.pi)) * 180 / np.pi]
#     return e_wf_fund_pa_ref
#


def delay_print(s, sleep=.1):
    for c in s:
        sys.stdout.write(c)
        sys.stdout.flush()
        time.sleep(sleep)


def plot_keras_history(history, show=True):
    """

    Args:
        history: The common returbn from keras's fit method
        show: (bool)

    Returns:

    """
    plt.figure(figsize=[12, 8])
    plt.subplot(2, 1, 1)
    plot(history.history['acc'])
    plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    # Plot training & validation loss values
    plt.subplot(2, 1, 2)
    plot(history.history['loss'])
    plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.tight_layout()
    if show:
        plt.show()


def window(seq, n=2):
    """
    for the record, this is from stack, not mine -> https://stackoverflow.com/questions/6822725/rolling-or-sliding-window-iterator

    Returns a sliding window (of width n) over data from the iterable
       s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...
    """
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result


def island_info(y, trigger_val, stop_ind_inclusive=True):
    '''
    returns tuple (index's of islands) (length of islands)
    See https://stackoverflow.com/questions/50151417/numpy-find-indices-of-groups-with-same-value
    '''
    if not isinstance(y, np.ndarray):
        y = np.array(y)
    # NOTE THIS AGAIN IS NOT MY CODE, BUT SOMETHING I FOUND HERE
    # https://stackoverflow.com/questions/50151417/numpy-find-indices-of-groups-with-same-value

    # Setup "sentients" on either sides to make sure we have setup
    # "ramps" to catch the start and stop for the edge islands
    # (left-most and right-most islands) respectively
    y_ext = np.r_[False, y == trigger_val, False]

    # Get indices of shifts, which represent the start and stop indices
    idx = np.flatnonzero(y_ext[:-1] != y_ext[1:])

    # Lengths of islands if needed
    lens = idx[1::2] - idx[:-1:2]

    # Using a stepsize of 2 would get us start and stop indices for each island
    return list(zip(idx[:-1:2], idx[1::2] - int(stop_ind_inclusive))), lens


def get_confusion(true, pred):
    """
    creates confusion  matrix

    """
    if len(true) != len(pred):
        print(f'these values are not the same length \n len pred {len(pred)} ---- len true {len(true)}')
    temp = pd.DataFrame({'true': true, 'pred': pred}).groupby(['true', 'pred']).size().unstack()
    return temp


def regression_results(y_true, y_pred):
    """
    Creates regrtession evaluation metrics
    """
    # Found here: https://stackoverflow.com/questions/26319259/how-to-get-a-regression-summary-in-python-scikit-like-r-does
    # Regression metrics
    explained_variance = metrics.explained_variance_score(y_true, y_pred)
    mean_absolute_error = metrics.mean_absolute_error(y_true, y_pred)
    mse = metrics.mean_squared_error(y_true, y_pred)
    mean_squared_log_error = metrics.mean_squared_log_error(y_true, y_pred)
    median_absolute_error = metrics.median_absolute_error(y_true, y_pred)
    r2 = metrics.r2_score(y_true, y_pred)

    print('explained_variance: ', round(explained_variance, 4))
    print('mean_squared_log_error: ', round(mean_squared_log_error, 4))
    print('r2: ', round(r2, 4))
    print('MAE: ', round(mean_absolute_error, 4))
    print('MSE: ', round(mse, 4))
    print('RMSE: ', round(np.sqrt(mse), 4))


# def reload_function():  # this is not functional yet
#     pass


def mean_normalization(vector):
    return (vector - np.nanmean(vector)) / (np.nanmax(vector) - np.nanmin(vector))


# def geometric_median(vector):  # this is non functional yet
#     dist = []
#     for i in range(vector.shape[0]-1):
#         for j in range(i, vector.shape[0]):
#             dist.append(np.linalg.norm(vector[i, :], vector[j, :]))
#     min_dist = dist.argmax()
#     # which is longest


def most_common(lst):  # from stack: https://stackoverflow.com/questions/1518522/find-the-most-common-element-in-a-list
    return max(set(lst), key=lst.count)


def gaussian_mle(data):
    # Code was found at:
    # https://stackoverflow.com/questions/51342724/how-to-estimate-gaussian-distribution-parameters-using-mle-in-python
    # used for finding parameters of 2d gaussian dist data
    mu = data.mean(axis=0)
    var = (data - mu).T @ (data - mu) / data.shape[0]  # this is slightly suboptimal, but instructive

    return mu, var


def stopword_maker(docs, n):
    pass


def select(lst, indices):
    return (lst[i] for i in indices)


def get_normalized_template_match(wf, template_wf, return_array=False):
    """Compute normalized cross correlation of two signals(wf and template)
       and return max of the time series of cross correlation with index.
       :arguments
       wf: unknown waveform
       template_wf: waveform pattern to cross-correlate with the unknown waveform

       :returns
       max of normalized cross-correlations between wf and template
       max index of normalized cross-correlations between wf and template
       """
    # length of original wf must be equal to or longer than the length of template_wf
    len_wf = len(wf)
    len_template_wf = len(template_wf)
    if len_wf < len_template_wf:
        print("Original wf is too short to cross-correlate with template wf")
        return None, None

    # template signal energy
    euclidean_norm_template_wf = np.linalg.norm(template_wf)

    # normalized cross-correlation
    normalized_xcor = []
    for i in range(len_wf - len_template_wf + 1):
        windowed_wf = wf[i:i + len_template_wf]
        euclidean_norm_windowed_wf = np.linalg.norm(windowed_wf)
        normalized_xcor.append(np.dot(template_wf, windowed_wf) / euclidean_norm_windowed_wf)

    # use vectorized division
    if return_array:
        return normalized_xcor / euclidean_norm_template_wf
    else:
        normalized_xcor = normalized_xcor / euclidean_norm_template_wf
        try:
            return np.nanmax(np.abs(normalized_xcor)), np.nanargmax(np.abs(normalized_xcor))
        except ValueError:
            return None, None
