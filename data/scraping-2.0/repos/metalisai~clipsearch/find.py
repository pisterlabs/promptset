from annoy import AnnoyIndex
import argparse
import openai
import os
import jsonlines
import json
import subprocess
import datetime

EMB_DIM = 1536
EMB_MODEL = "text-embedding-ada-002"

parser = argparse.ArgumentParser(description="Find clips")
parser.add_argument('database', type=str, help='database file name')
parser.add_argument('--query', nargs='?', const=None, type=str, help='what to search for')

args = parser.parse_args()

database_json = os.path.splitext(args.database)[0] + '.jsonl'
database_meta_json = os.path.splitext(args.database)[0] + '.meta.json'

# load all metadata
with jsonlines.open(database_json) as db_meta:
    db_json = [obj for obj in db_meta]

with open(database_meta_json) as db_meta:
    meta_json = json.load(db_meta)

while True:
    query = args.query
    if query == None:
        query = input("Insert query: ")

    query_emb = openai.Embedding.create(input=query, model=EMB_MODEL)["data"][0]["embedding"]
    if len(query_emb) != EMB_DIM:
        raise ArgumentError(f'Returned embedding is not valid')

    u = AnnoyIndex(EMB_DIM, 'angular')
    u.load(args.database)

    i = 0
    clips = []
    # query 10 best matches from vector database
    results = u.get_nns_by_vector(query_emb, 10, include_distances=True)
    for idx, dist in zip(results[0], results[1]):
        cur_clip = db_json[idx]
        meta_hash = cur_clip['meta']
        video_meta = meta_json[meta_hash]

        data = (db_json[idx]['text'], db_json[idx]['start'], db_json[idx]['end'], video_meta['video'])
        clips.append(db_json[idx])
        print(f"{i}: {data}")
        i += 1

    while True:
        action = input("Action: [n]ew, [p]lay <idx>, [q]uit:")
        if action[:1] == 'n': # user wants to query again
            break
        elif action[:1] == 'p': # user wants to play video clip from results
            clip_id = int(action.split(' ')[1])
            #print(f"play clip {clips[clip_id]}")
            cur_clip = clips[clip_id]
            meta_hash = cur_clip['meta']
            video_meta = meta_json[meta_hash]
            start_str = cur_clip['start'];
            # if you got a clip more than 24 hours long then you're in trouble!
            t = datetime.datetime.strptime(start_str, "%H:%M:%S.%f")
            td2 = datetime.timedelta(hours=t.hour, minutes=t.minute, seconds=t.second)
            #print(f"Play video {video_meta['video']} at {start_str}")
            subprocess.run(["vlc", f"--start-time={td2.total_seconds()}", f"{video_meta['video']}"])
        elif action[:1] == 'q':
            exit()

    # if we passed query into command, we're done (not interactive)
    if args.query != None:
        break
