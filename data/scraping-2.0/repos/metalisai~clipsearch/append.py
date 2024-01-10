import argparse
import os
import sys
from annoy import AnnoyIndex
import jsonlines
import json
import hashlib
import openai
import time

EMB_DIM = 1536 # OpenAI ada v2 has 1536 dimensions
EMB_MODEL = "text-embedding-ada-002"

parser = argparse.ArgumentParser(description="Clip videos")
parser.add_argument('database', type=str, help='database file name')
parser.add_argument('input', type=str, help='input json with metadata to be added to database')

args = parser.parse_args()

# load existing database if exists
t = AnnoyIndex(EMB_DIM, 'angular')
database_existed = False
database_json = os.path.splitext(args.database)[0] + '.jsonl'
database_meta_json = os.path.splitext(args.database)[0] + '.meta.json'

# check if database exists
if os.path.exists(args.database):
    database_existed = True
else:
    print(f"Datbase {args.database} doesn't exist, creating new...", file=sys.stderr)

# if vector database exists, the metadata should exist as well
if database_existed and not os.path.exists(database_json):
    raise FileNotFoundError(f'The database is missing "{database_json}". If you do not have it, the database is not usable.')
    exit()

# turns text into an embedding using an OpenAI model
def get_embedding(text):
    ret = openai.Embedding.create(input=text, model=EMB_MODEL)["data"][0]["embedding"]
    if len(ret) != EMB_DIM:
        raise ArgumentError(f'Returned embedding not valid')
    time.sleep(1.5)
    return ret
#def get_embedding(text):
#    return [0.0] * EMB_DIM

# database that contains all info about an embedding
db_json = []
if database_existed: 
    with jsonlines.open(database_json) as db_meta:
        db_json = [obj for obj in db_meta]

# json file that contains meta for key (shared between many embeddings)
meta_json = {}
if database_existed:
    with open(database_meta_json) as db_meta:
        meta_json = json.load(db_meta)

# load all embeddings
for embedding in db_json:
    t.add_item(embedding['idx'], embedding['emb'])

db_count = len(db_json)
old_db_count = db_count

# Open metadata json
with open(args.input, 'r') as in_meta:
    input_json = json.load(in_meta)

    # if have unseen metadata, get embeddings and store it
    meta_hash = None
    if not 'meta' in meta_json and input_json['meta'] != None:
        meta_hash = hashlib.sha256(str.encode(input_json['meta'], 'utf-8')).hexdigest()
        # this meta does not exist, create
        if not meta_hash in meta_json:
            emb = get_embedding(input_json['meta'])
            new_meta = {
                'meta': input_json['meta'], 
                'video': input_json['video'], 
                'source': args.input, 'emb': emb
            }
            meta_json[meta_hash] = new_meta
            print(f"add meta {meta_hash}")

    # iterate all subtitles and add to metadata database
    for utter in input_json['utters']:
        emb = get_embedding(utter['text'])
        print(f"Got embedding {db_count-old_db_count}/{len(input_json['utters'])}")
        # openai api only allows 60 requests per minute
        t.add_item(db_count, emb)
        out_meta = {
            'idx': db_count, 
            'meta': meta_hash, 
            'text': utter['text'], 
            'start': utter['start'], 
            'end': utter['end'],
            'emb': emb,
        }
        db_json.append(out_meta)
        db_count += 1

        with jsonlines.open(database_json, 'w') as f:
            f.write_all(db_json)

print(f"Added {db_count-old_db_count} entries to database. Now {db_count} items.")

t.build(10)
t.save(args.database)
# write utter database
with jsonlines.open(database_json, 'w') as f:
    f.write_all(db_json)
# write meta database
with open(database_meta_json, 'w') as db_meta:
    json.dump(meta_json, db_meta)
