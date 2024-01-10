# the core of this is get_multiple_essence and get_essence. ive uploaded the results in this repo, in "essence_result.txt"

import tkinter as tk
import requests, json, threading, time, os
import numpy as np
import tkinter.font as tkfont

from colorama import init
init()

from secret_things import openai_key

def col(ft, s):
    """For printing text with colors.
    
    Uses ansi escape sequences. (ft is "first two", s is "string")"""
    # black-30, red-31, green-32, yellow-33, blue-34, magenta-35, cyan-36, white-37
    u = '\u001b'
    numbers = dict([(string,30+n) for n, string in enumerate(('bl','re','gr','ye','blu','ma','cy','wh'))])
    n = numbers[ft]
    return f'{u}[{n}m{s}{u}[0m'

class Helper:
    def __init__(self):
        self.d = {}
    def get_remaining(self, strings):
        existing = self.d.keys()
        return [s for s in strings if s not in existing]
    def get_vectors(self, strings):
        return [self.d[s] for s in strings]
    def update(self, strings, vectors):
        assert len(strings) == len(vectors)
        for s, v in zip(strings, vectors):
            self.d[s] = v
helper_obj_embedder_api = Helper()

def embedder_api(strings):
    global helper_obj_embedder_api
    # can use a helper_obj_embedder_api that exists as a global object, not sure if this is a wise "design pattern" lmao
    try:
        helper_obj_embedder_api.get_remaining
        helper_obj_embedder_api.get_vectors
        helper_obj_embedder_api.update
    except:
        print(col('ye', '''note: embedder_api wants helper_obj_embedder_api to have .get_remaining, .get_vectors, .update'''))
        helper_obj_embedder_api = None
        to_embed = strings
    else:
        to_embed = helper_obj_embedder_api.get_remaining(strings)

    if to_embed == []:
        return helper_obj_embedder_api.get_vectors(strings)
    else:
        headers = {
            "Authorization": f"Bearer {openai_key}",
            "Content-Type": "application/json"
        }
        data = {
            "input": to_embed,
            "model": "text-embedding-ada-002"
        }
        response = requests.post("https://api.openai.com/v1/embeddings", headers=headers, json=data)

        if response.status_code != 200:
            print(col('re', to_embed))
            print(vars(response))
            raise Exception
        else:
            print(f'successfully embedded {len(to_embed)} strings')
        data = response.json()['data']
        vectors = [d['embedding'] for d in data]

        if helper_obj_embedder_api != None:
            helper_obj_embedder_api.update(to_embed, vectors)
            return helper_obj_embedder_api.get_vectors(strings)
        else:
            assert len(vectors) == len(strings)
            return vectors

def embsort_multiquery(positive, negative, options, rescount):
    # 3 lists of strings --> list of `rescount` strings

    """
    - sort strings based on similarity to multiple strings' embeddings
    - calculate similarity of each option to each string in positive and negative, sort by sum
    """

    # assert list of strings
    for arg in (positive, negative, options,):
        assert type(arg) == list
        for i in arg:
            assert type(i) == str
            assert i != ''

    vectors = embedder_api(positive+negative+options)
    # assuming embedder_api preserves order, lol.
    positive_emb = vectors[:len(positive)]
    negative_emb = vectors[len(positive):len(positive)+len(negative)]
    options_emb = vectors[len(positive)+len(negative):]

    # for sanity
    assert all([
        len(positive) == len(positive_emb),
        len(negative) == len(negative_emb),
        len(options) == len(options_emb),
    ])

    triplets = sorted(
        [(
            n,
            options[n],
            sum([np.dot(v,options_emb[n]) for v in positive_emb]) - sum([np.dot(v,options_emb[n]) for v in negative_emb]),
        ) for n in range(len(options))],
        key=lambda triplet: triplet[2],
        reverse=True
    )

    top = []
    for t in triplets[:rescount]:
        idx = t[0]
        top.append(options[idx])
    return top

def embsort(query, options, rescount):
    # (string, list of strings) --> list of `rescount` strings

    return embsort_multiquery(
        [query],
        [],
        options,
        rescount
    ) # since this is a special case of embsort_multiquery, might as well just call that one

def get_essence(string, chunksize, rescount):
    def get_chunks(string, chunksize):
        # get overlapping multi-word chunks of the string
        chunks = []
        words = string.split(' ')
        for i in range(0, len(words)-chunksize+1):
            chunks.append(' '.join(words[i:i+chunksize]))
        return chunks
    return embsort(
        string,
        get_chunks(string, chunksize),
        rescount,
    )

def get_multiple_essence(string, chunksize, layercount):
    # the idea is, whichever group of words is most similar to the full string, that's the "essence" of the text.
    # then each iteration, you discount for previous picks, so that you get multiple (conceptually) different candidates for the "essence"

    # (embedder_api has some hideous hacky stuff to allow embedding storage and prevent multiple api calls)

    def get_chunks(string, chunksize):
        # get overlapping multi-word chunks of the string
        chunks = []
        words = string.split(' ')
        for i in range(0, len(words)-chunksize+1):
            chunks.append(' '.join(words[i:i+chunksize]))
        return chunks

    chunks = get_chunks(string, chunksize)
    picks = []
    for i in range(layercount):
        picks.append(embsort_multiquery(
            [string],  # positive query set
            picks,  # negative query set
            chunks,
            1,
        )[0])
    return picks


# to compare the 2 "essence grabber" algorithms
def test():
    inpath = 'paul_graham_alien_truth.txt'  # http://www.paulgraham.com/alien.html
    outpath = 'essences_result.txt'

    # the more "essences" you get and the longer the text, the more relevant this technique becomes

    def readfile(path):
        with open(path, 'r', encoding='utf-8') as f:
            if path.endswith('.json'):
                content = json.load(f)
            else:
                content = f.read()
        return content
    def writefile(path, content):
        with open(path, 'w', encoding='utf-8') as f:
            if isinstance(content, (dict, list)):
                json.dump(content, f, indent=2)
            else:
                f.write(content)

    # get paragraphs and full article
    teststrings = [p for p in readfile(inpath).split('\n\n') if len(p)>20]
    teststrings.append(readfile(inpath))

    # pre store embeddings
    def get_chunks(string, chunksize):
        # get overlapping multi-word chunks of the string
        chunks = []
        words = string.split(' ')
        for i in range(0, len(words)-chunksize+1):
            chunks.append(' '.join(words[i:i+chunksize]))
        return chunks
    chunks = []
    for s in teststrings[:-1]:
        chunks += get_chunks(s, 3)
    chunks += get_chunks(teststrings[-1], 10)
    chunks += teststrings
    embedder_api(chunks)  # im surprised the api can handle 1305 embeddings at a time lol

    lines = []
    for s in teststrings:
        last = s == teststrings[-1]
        chunksize = 10 if last else 3
        count = 10 if last else 5

        if last:
            lines.append('{full article here}')
        else:
            lines.append(s)

        res = get_essence(
            s,
            chunksize,
            count,
        )
        lines.append('unlayered:')
        lines.append('\n'.join([f'{n} - {r}' for n, r in enumerate(res)]))

        res = get_multiple_essence(
            s,
            chunksize,
            count
        )
        lines.append('layered:')
        lines.append('\n'.join([f'{n} - {r}' for n, r in enumerate(res)]))
        lines.append('')
        lines.append('='*30)
        lines.append('')
    writefile(outpath, '\n'.join(lines))
    print(f'used {inpath}, wrote results to {outpath}')
test()
