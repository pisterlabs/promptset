from openai.embeddings_utils import get_embedding
from openai.error import InvalidRequestError
from openai import api_key
#from transformers import GPT2Tokenizer
import json
import sys
import sqlite3
import pandas as pd
import argparse

defaultdbfile = 'vectors.db'
tabname = 'embeddings'
create_table_sql = f'CREATE TABLE {tabname} ( path TEXT, name TEXT, firstLine INTEGER, lastLine INTEGER, vectorid INTEGER PRIMARY KEY ); CREATE TABLE vectors ( elem DOUBLE, ord INT, id INTEGER, FOREIGN KEY(id) REFERENCES {tabname}(vectorid) );'

def main():
    global api_key
    parser = argparse.ArgumentParser()
    parser.add_argument('--database', '-D', type=str, default=defaultdbfile, help='SQLite3 database filename for storing vectors')
    parser.add_argument('--input-file', '-f', type=str, default='-', help='File for JSON input or - for stdin')
    parser.add_argument('--api-key', type=str, default=None, help='OpenAI API key (or use env var OPENAI_API_KEY)')
    args = parser.parse_args()
    if args.api_key is not None:
        api_key = args.api_key
    con = sqlite3.connect(args.database)

    if con.execute("SELECT count(name) FROM sqlite_master WHERE type='table' AND name=?", (tabname,)).fetchone()[0]==0:
        con.executescript(create_table_sql)

    with sys.stdin if args.input_file == '-' else open(args.input_file) as f:
        df = pd.read_json(f, lines=True)
    #tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    for ex in df.itertuples():
        print(f"file {ex.path} pu {ex.name} lines {ex.firstLine}-{ex.lastLine}")
        sys.stdout.flush()
        if con.execute("SELECT count(*) FROM embeddings WHERE path=? AND firstLine=?", (ex.path, ex.firstLine)).fetchone()[0] == 0:
            src = []
            with open(ex.path, errors='replace') as f:
                for line_num, line in enumerate(f):
                    if line_num >= ex.firstLine-1 and line_num <= ex.lastLine-1:
                        src.append(line)
            src=('').join(src)
            #print(len(tokenizer(ex['src'])['input_ids']))
            txt=src[:2048]
            emb=[]
            while len(emb) == 0 and len(txt) > 2:
                try:
                    emb=get_embedding(txt, engine='code-search-babbage-code-001')
                except Exception as err:
                    print(err)
                    txt=txt[:int(len(txt)/2)]
                    print(f'trying with len={len(txt)}')
            cur = con.execute("INSERT INTO embeddings (path, name, firstLine, lastLine) VALUES (?, ?, ?, ?)",
                              (ex.path, ex.name, ex.firstLine, ex.lastLine))
            vid = cur.lastrowid
            for i,x in enumerate(emb):
                con.execute("INSERT INTO vectors (id, ord, elem) VALUES (?, ?, ?)", (vid, i, x))
            con.commit()

if __name__=="__main__":
  main()
