import requests, json, threading, time, os
import numpy as np

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

def embedder_api(strings):
    headers = {
        "Authorization": f"Bearer {openai_key}",
        "Content-Type": "application/json"
    }
    data = {
        "input": strings,
        "model": "text-embedding-ada-002"
    }
    response = requests.post("https://api.openai.com/v1/embeddings", headers=headers, json=data)

    if response.status_code != 200:
        print(col('re', strings))
        print(vars(response))
        raise Exception
    else:
        print(f'successfully embedded {len(strings)} strings')
    data = response.json()['data']
    return [d['embedding'] for d in data]

def input_multi():
    print("Enter/Paste your content. Ctrl-D or Ctrl-Z ( windows ) to save it.")
    contents = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        contents.append(line)
    return '\n'.join(contents)

class StaticDb:       
    def __init__(self, foldername):
        self.foldername = foldername
        self.nppath = f'{self.foldername}/array.npy'
        self.stringspath = f'{self.foldername}/idx_to_string.json'
    
    def grab_strings(self, cmd):
        if os.path.exists(cmd):
            return [p for p in readfile(cmd).split('\n\n') if p!='']
        elif cmd == 'input':
            return [p for p in input_multi().split('\n\n') if p!='']
        else:
            print(col('re', f'cant grab strings with "{cmd}"'))
            raise Exception
    
    # note to github: sry, this is only relevant for another program im using this with.
    def append_folderlist(self):
        l = readfile('folderlist.json')
        l.append(self.foldername)
        writefile('folderlist.json', list(set(l)))
        """writefile(
            'folderlist.json',
            list(set(readfile('folderlist.json')))+[self.foldername]
        )"""

    def create(self, strings):
        if type(strings) == str:
            cmd = strings
            strings = self.grab_strings(cmd)

        if not os.path.exists(self.foldername):
            os.mkdir(self.foldername)

        array = []
        
        to_embed = strings
        per_call = 50
        for i in range(0, len(to_embed), per_call):
            vectors = embedder_api(to_embed[i:i+per_call])
            array += vectors

        np.save(self.nppath, array)
        writefile(self.stringspath, strings)
    
    def apply_filter(self, func, ask_input=False):
        assert type(func('test')) == bool

        if ask_input:
            new = []
            auto = False
            for s in self.strings:
                if func(s):
                    new.append(s)
                else:
                    if auto:  # if func returns False you automatically go on without double checking with the user
                        continue
                    print(col('re', s))
                    print('happy with this removal? (y/n/stop asking)')
                    while True:
                        i = input('> ')
                        if i == 'y':
                            break
                        elif i == 'n':
                            new.append(s)
                            break
                        elif i == 'stop asking':
                            auto = True
                            break
                        else:
                            print(col('re', 'invalid input'))
        else:
            print(col('cy', 'filtering without asking for input'))
            new = list(filter(func, self.strings))

        if new == self.strings:
            print(col('re', 'filter not needed, result is the same lol'))
            return
        self.create(new)
        self.load()
        print(col('cy', f'loaded {self.foldername} after filtering'))
    
    def edit_strings(self, func, ask_input=False):
        assert type(func('test')) == str

        if ask_input:
            new = []
            auto = False
            for s in self.strings:
                changed = func(s)
                if auto:
                    new.append(changed)
                else:
                    print(col('re', s))
                    print(col('gr', changed))
                    print('happy with this change? (y/n/stop asking)')
                    while True:
                        i = input('> ')
                        if i == 'y':
                            new.append(changed)
                            break
                        elif i == 'n':
                            new.append(s)
                            break
                        elif i == 'stop asking':
                            new.append(changed)
                            auto = True
                            break
                        else:
                            print(col('re', 'invalid input'))
        else:
            print(col('cy', 'editing without asking for input'))
            new = list(filter(func, self.strings))

        if new == self.strings:
            print(col('re', 'edit not needed, result is the same lol'))
            return
        self.create(new)
        self.load()
        print(col('cy', f'loaded {self.foldername} after editing'))

    def load(self):
        self.array = np.load(self.nppath)
        self.strings = readfile(self.stringspath)

    def search(self, query, maxres):
        if type(query) == list and len(query) == 1536:
            query_emb = query
        else:
            query_emb = embedder_api([query])[0]

        #t0 = time.time()
        assert len(query_emb) == 1536

        scores = np.dot(self.array, np.array(query_emb))
        dtype = [
            ('score', float),
            ('text', 'O'),
        ]
        values = []
        for score, string in zip(scores, self.strings):
            # Store the string as bytes, not as a Unicode string.
            values.append((score, string.encode('utf-8')))

        special_array = np.array(values, dtype=dtype)
        sorted_by_dot = np.sort(special_array, order='score')
        #print(time.time()-t0)
        topn = list(reversed(sorted_by_dot))[:maxres]
        res = [tup[1].decode('utf-8') for tup in topn]
        return res

    def get_average(self, query):
        if type(query) == list and len(query) == 1536:
            query_emb = query
        else:
            query_emb = embedder_api([query])[0]
        scores = np.dot(self.array, np.array(query_emb))
        return np.average(scores)
    
    def __getitem__(self, i):
        return self.strings[i]

def example_capital_filter(s):
    try:
        int(s[0])
    except:
        pass
    else:
        return True
    if s.lower() == s:
        return False
    if s[0].lower() == s[0]:
        return False
    if s[0].upper() == s[0]:
        return True
    return True
