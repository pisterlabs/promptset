import json
import numpy as np
import time

# Open the JSON file
with open('tg/result.json', 'r') as f:
  # Load the JSON data into a Python object
  data = json.load(f)


data['messages'][0]
msgs = data['messages']
#msgs = data['messages'][:100]
import openai
openai.api_key = "sk-xpr55FOwcFK8IurIltzukJJYurj0tdNw7jdjOtvx"
import numpy as np

# msgs[0]
# list(filter(lambda x: '^' in x["text"], msgs))[102]

msgs_by_id = {m['id']:m for m in msgs}

from collections import Counter
# msg_replies = Counter({m['reply_to_message_id']:[m] for m in filter(lambda x: 'reply_to_message_id' in x, msgs)})
msg_replies = Counter({m['reply_to_message_id']:[m["id"]] for m in filter(lambda x: 'reply_to_message_id' in x, msgs)})

MAX_CHARS = 8191*4

#%%


APP_KEY = 'xO9eRr4Eleg994pwfmEJTPdZA'
APP_SECRET = 'P4YITGt8qKBnMISXbydrfCapaA6ODxy2H4NUGYCQBXsFZZbgdb'

# from twython import Twython
# twitter = Twython(APP_KEY, APP_SECRET)
# len("868039496872407040")
# combined_messages[200]
# tweet_id = combined_messages[200]["text"][-1]["text"].split("/")[-1][:18]
# tweet = twitter.show_status(id=tweet_id)
# tweet['text']

import tweepy
auth = tweepy.OAuthHandler(APP_KEY, APP_SECRET)
api = tweepy.API(auth)

from urllib.parse import urlparse
import urllib.request

import pdfreader

from pdfreader import PDFDocument, SimplePDFViewer
from bs4 import BeautifulSoup

def get_text_from_pdf(my_raw_data):
    with open("my_pdf.pdf", 'wb') as my_data:
        my_data.write(my_raw_data)

    open_pdf_file = open("my_pdf.pdf", 'rb')
    read_pdf = SimplePDFViewer(open_pdf_file)
    return read_pdf.canvas.text_content

def get_link_text(text, include_non_twitter=True, include_pdfs=True):
    result = ""
    if type(text) == type([]):
        for t in text:
            if type(t) == type({}) and t["type"] == "link":
                url = t["text"]
                # print(url)
                try:
                    if urlparse(url).netloc.split(".")[-2:] == ["twitter", "com"]:
                        tweet_id = url.split("/")[-1].split('?')[0]
                        # print(tweet_id)
                        tweet = api.get_status(tweet_id, tweet_mode="extended")
                        time.sleep(3)
                        # print(tweet.full_text)
                        result += tweet.full_text + ". "
                    elif include_non_twitter:
                        #fp = urllib.request.urlopen(url)
                        req = urllib.request.Request(
                            url, 
                            data=None, 
                            headers={
                                        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.47 Safari/537.36'
                                    }
                        )
                        fp = urllib.request.urlopen(req)
                        if fp.url.split(".")[-1] == "pdf" and include_pdfs:
                            mybytes = fp.read()
                            result += get_text_from_pdf(mybytes)[:MAX_CHARS] + ". "
                        else:
                            if fp.url != url:
                                req = urllib.request.Request(
                                    fp.url, 
                                    data=None, 
                                    headers={
                                                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.47 Safari/537.36'
                                            }
                                )
                                fp = urllib.request.urlopen(req)
                            mybytes = fp.read()

                            mystr = mybytes.decode("utf8")
                            fp.close()
                            soup = BeautifulSoup(mystr)
                            if soup.body is not None:
                                result += soup.body.get_text(". ", strip=True)[:MAX_CHARS] + ". "
                            else:
                                result += soup.get_text(". ", strip=True)[:MAX_CHARS] + ". "
                except Exception as e:
                    print(e)
                    print(url)
                    # print(tweet_id)
                    continue
    return result

# get_link_text(combined_messages[201])
#
# combined_messages[301]
#

# # url = 'https://en.m.wikipedia.org/wiki/Ehud_Shapiro'
# url='https://t.co/HUUfZ7sveK'
# fp = urllib.request.urlopen(url)
# fp.url
# mybytes = fp.read()
#
# mystr = mybytes.decode("utf8")
# fp.close()
# mystr
#
# soup = BeautifulSoup(mystr)
# print(soup.body.get_text(". ", strip=True))
#

def nicefy(t, include_link_text=True, include_non_twitter=True, include_pdfs=True):
    output = ""
    for thingo in t:
        if type(thingo) == type(""):
            output += thingo + ". "
        elif "text" in thingo:
            #output += thingo["text"] + ". "
            output += thingo["text"]
        else:
            print("WHAT")

    output = output.replace("\n", " ").strip()
    if include_link_text:
        output += " . " + get_link_text(t, include_non_twitter=include_non_twitter, include_pdfs=include_pdfs)
    return output

def rec_get_text(id,include_link_text=True, include_non_twitter=True, include_pdfs=True):
    #textt = nicefy(msgs_by_id[id]["text"],include_link_text=include_link_text, include_non_twitter=include_non_twitter, include_pdfs=include_pdfs) + ". "
    textt = msgs_by_id["main_text"]
    ids = [id]
    if id in msg_replies:
        for id2 in msg_replies[id]:
            text,ids3 = rec_get_text(id2, include_link_text=include_link_text, include_non_twitter=include_non_twitter, include_pdfs=include_pdfs)
            textt += text + ". "
            ids.append(ids3)
    return textt, ids


#%%
num_msgs = len(msgs)
i=0
while i<num_msgs:
    if i%100 == 0:
        print(i)
    m=msgs[i]
    #m["main_text"] = nicefy(m["text"], include_link_text=True, include_non_twitter=False, include_pdfs=False)
    m["main_text"] = nicefy(m["text"], include_link_text=True, include_non_twitter=True, include_pdfs=True)
    msgs[i] = m
    i+=1

import json

with open('data.json', 'w') as outfile:
    # Write the data to the file using the json.dump() method
    json.dump(combined_messages, outfile)

#combine linked and nearby messages

combined_messages = []
message_nearby_threshold=120
i=0
included_ids = []
while i<num_msgs:
    if i%100 == 0:
        print(i)
    m=msgs[i]
    id = m["id"]
    date_unixtime = m["date_unixtime"]
    if id not in included_ids:
        new_m = m.copy()
        if id in msg_replies:
            new_m["replies"] = msg_replies[id]
        #new_m["main_text"] = nicefy(new_m["text"], include_non_twitter=False, include_pdfs=False)
        text,extra_ids = rec_get_text(id, include_non_twitter=False, include_pdfs=False)
        new_m["text_and_replies"] = text
        included_ids += extra_ids
        extended_text = new_m["text_and_replies"]
        i+=1
        #while i<num_msgs and np.abs(int(msgs[i]["date_unixtime"]) - int(date_unixtime)) <= 20:
        while i<num_msgs and np.abs(int(msgs[i]["date_unixtime"]) - int(date_unixtime)) <= message_nearby_threshold:
            m2 = msgs[i]
            date_unixtime = m2["date_unixtime"]
            id2 = m2["id"]
            if id2 not in included_ids:
                text,extra_ids = rec_get_text(id2, include_non_twitter=False, include_pdfs=False)
                extended_text += text + ". "
                included_ids += extra_ids
            i+=1

        new_m["full_text"] = extended_text

        combined_messages.append(new_m)
    else:
        i+=1


#combined_messages[100]
#
#combined_messages[result[4]-1]
#len(combined_messages["messages"])
#
#combined_messages = json.loads(open("data.json", "r").read())

#%%

with open('combined_messages_'+str(message_nearby_threshold)+'.json', 'w') as outfile:
    # Write the data to the file using the json.dump() method
    json.dump(combined_messages, outfile)

#msgs[3]["text"]
# cleaned_texts = list(map(lambda x: nicefy(x["text"]), msgs))
# cleaned_texts = list(filter(lambda x: x!="", map(lambda x: nicefy(x["text"]), msgs)))
# cleaned_texts = list(filter(lambda x: x!="", map(lambda x: nicefy(x["full_text"]), msgs)))
#cleaned_texts = list(filter(lambda x: x!="", map(lambda x: nicefy(x["full_text"]), combined_messages)))
cleaned_texts = list(filter(lambda x: x!="", map(lambda x: x["full_text"], combined_messages)))
#indices = list(filter(lambda i,x: x!="", map(lambda i,x: (i,x["full_text"]), enumerate(combined_messages))))
#%%

def get_embedding(texts, model="text-embedding-ada-002"):
    return openai.Embedding.create(input = texts, model=model)['data']

# len(" ".join(cleaned_texts).split(" "))

# cleaned_texts[:10]
# i=0
N=len(cleaned_texts)
# ws=10
ws=1
# bs=1000
bs=2000
# i=N//bs-1
# i=12
# window = lambda j: ". ".join(cleaned_texts[j:j+ws])[:int(8191/0.75)]
# eee = get_embedding([window(j) for j in range(i*bs,min((i+1)*bs,N-ws))])
embeddings = []
# for i in range(N//bs):
texts = []
for i in range(N//bs+1):
    print(i)
    # window = lambda j: ". ".join(cleaned_texts[j:j+ws])[:int(8191*0.75)]
    window = lambda j: ". ".join(cleaned_texts[j:j+ws])[:MAX_CHARS]
    windows = [window(j) for j in range(i*bs,min((i+1)*bs,N-ws+1))]
    eee = get_embedding(windows)
    embeddings += eee
    texts += windows

# embeddings = [get_embedding([". ".join(cleaned_texts[j:j+ws]) for j in range(i*bs,min((i+1)*bs,N-ws))]) for i in range(N//bs)]
# embeddingss = sum(embeddings,[])

# eee
# embs = np.stack(list(map(lambda x: np.array(x["embedding"]), eee)))

len(embeddings[0]["embedding"])

embs = np.stack(list(map(lambda x: np.array(x["embedding"]), embeddings)))

len(cleaned_texts)

embs.shape


# np.save("embeddings_ws10", embs)
np.save("embeddings_ws1_"+str(message_nearby_threshold), embs)

#%%

# embs = np.load("embeddings.py")
embs = np.load("embeddings_ws1.npy")
from semantic_search_sandbox import search, normalize_embeddings
normalized_embs = normalize_embeddings(embs)

#%%
# query_str = "^^^"
query_str = "^^^"
query_emb = get_embedding([query_str])[0]["embedding"]
query_emb = np.array(query_emb)
query_emb

import semantic_search_sandbox
import importlib; importlib.reload(semantic_search_sandbox)
from semantic_search_sandbox import search, normalize_embeddings

result = search(query_str, query_emb, normalized_sentence_embeddings=normalized_embs, texts=texts, fuzzy_weight=0.1, n=10)

texts[result[4]-1]

len(texts)
for i in range(10):
    # print(texts[result[i]])
    index = result[i]
    index_start = max(0,index-1)
    index_end = min(len(texts),index+1)
    first_chunk = ". ".join(texts[index_start:index])
    last_chunk = ". ".join(texts[index+1:index_end])
    max_len=500
    print(first_chunk[:max_len]+("..." if len(first_chunk)>max_len else "")+" **"+texts[index]+"** "+last_chunk[:max_len]+("..." if len(last_chunk)>max_len else ""))
