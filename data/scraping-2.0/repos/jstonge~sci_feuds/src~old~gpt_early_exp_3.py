
import json
from jsonlines import jsonlines
import openai
import tiktoken
import tqdm
from pathlib import Path
import re
from datetime import date

today = date.today().strftime("%Y-%m-%d")

ROOT_DIR = Path("..")
OUTPUT_DIR = ROOT_DIR / 'output'
DAILY_CONVERSATION = OUTPUT_DIR / today

if DAILY_CONVERSATION.exists() == False: 
    DAILY_CONVERSATION.mkdir()

openai.api_key = open("apikey.txt", "r").read().strip("\n")
enc = tiktoken.encoding_for_model("gpt-3.5-turbo")

# old
# instructions = """
# Instructions:
#  - You are an editors working for a journal. 
#  - Your task is to analyze articles that consist of many paragraphs.
#  - Your analysis should contain the following:
#      - Does each sentence contains mentions or references to other authors (recall that names often start with upper case) ?
#  - Provide a sentiment score for the paragraphs using the following scale: 1 to 9 (where 1 is most negative, 5 is neutral, and 9 is most positive). 
#  - Take into account opposing sentiments in the mentions or references to other authors, but also try to correctly identify descriptive statements. 
#  - Format your response in a json format where for each sentence, you provide the text, overall sentiment score, then if there are mentions or references, with their associated sentiment scores, and finally the equation, model, and if this is the author's view. Don't explain your results.
# """

# instructions="""
# Instructions:\n
# - Your task is to analyze articles that consist of many paragraphs.\n 
# - Your analysis should contain the following:\n   
# - Does each sentence contains mentions or references to other authors (recall that names often start with upper case)?\n 
# - Provide a sentiment score for the paragraphs from 1 to 9 (where 1 is most negative, 5 is neutral, and 9 is most positive).\n
# - Does each sentence contains mentions or references to theoretical constructs (theories that you could find on wikipedia such as natural selection, population genetics).  \n - Take into account opposing sentiments in the mentions or references to other authors (individuals that the author of the paragraph cite), but also be mindful of descriptive statements. \n NO NEED TO EXPLAIN OR COMMENT YOUR RESULTS. \n - Format your response in a json format where for each sentence,  provide the text, the overall sentiment score, any mentions or references with their associated sentiment scores.
# """

instructions = """
Extract the sentences in the paragraph from a scientific article below. First extract all sentiment scores (where 1=extremely negative; 5=neutral; 9=extremely positive.), then extract all reference names (other individuals that author cite) with their sentiment score (again from 1 to 9), then extract theoretical  constructs (theories that you could find on wikipedia such as natural selection, population genetics, or group selection) with discrete sentiment score (supportive, descriptive, against).

Desired format:

  { "Sentence 1": [
      "Text":  <text_sentence>,
      "Sentiment": <sentiment_score>,
      "References":  [
       { "Reference name": <reference_name>,  "Reference sentiment": <reference_sentiment> },
       ...
    ],
    "Theory": [
      {"Theory name": <theory>,  "Theory sentiment": <sentiment_theory>},
      ...
   ]
  },
  { "Sentence 2" : [
      "Text":  <text_sentence>,
     ...
  ]
},
  { "Sentence 10" : [
    "Text": <text_sentence>,
  ...
  ] 
}
]
"""

# with open("../output/group_selection_grobid/alexander_group_1978.json") as f:
#     dat = json.load(f)

target_article = 'braestrup_animal_1963'

with open(f"../output/group_selection_grobid/{target_article}.json") as f:
    dat = json.load(f)

texts = [_['text'] for _ in dat['pdf_parse']['body_text']]

print(f"Nb tokens: {len(enc.encode(' '.join(texts)))}")

# Define a function to interact with ChatGPT
#!TODO Maybe precalculate everything, then send it once a time to gpt
def chat_with_gpt(max_tokens=1000, temp=1):
    prompt = ""
    message_history = []
    global_message_history = [] 

    i = 0
    i_last_end = -1
    while i < len(texts):
        print(f"Doing batch {i}")
    
        toks_count = len(enc.encode(prompt)) + len(enc.encode(texts[i]))
        if i == (i_last_end + 1):   # if we are starting a new conversation, prepend the instructions        
            prompt += f"{instructions}##\n{texts[i]}\n"
        elif toks_count < max_tokens:
            prompt += f"##\n{texts[i]}\n"
        else: #if we have hit the limit
            message_history.append({"role": "user", "content": prompt}) #format the request properly for output

            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo", 
                messages=message_history,
                temperature=temp)          
            reply_content = completion.choices[0].message.content
            message_history.append({"role": "assistant", "content": reply_content})

            # update global history
            global_message_history += message_history
            
            # reinitialize everything
            message_history = []
            prompt = ""
            i_last_end = i
        
        i += 1

    return global_message_history

temp=0.2
out = chat_with_gpt(temp=temp)


fname_out = DAILY_CONVERSATION / f"{target_article}_gpt35turbo_formatted_{temp}temp.jsonl"
with jsonlines.open(fname_out, "w") as f:
    f.write(out)


# ------------------------------ comparing text ------------------------------ #

from difflib import SequenceMatcher

target_article = 'braestrup_animal_1963'

with jsonlines.open(DAILY_CONVERSATION / f"{target_article}_gpt35turbo.jsonl") as f:
    dat = f.read()

print(out[1]['content'])

# dat = out

def reconstruct_text_from_reply(text):
    return ' '.join([_['text'] for _ in text])

def flatten(l):
    return [item for sublist in l for item in sublist]

def parse_raw_conv(conv_raw):
    """
      - standardize raw conversation with gpt
    """
    all_conv = []
    for conv in conv_raw:  
        # conv = all_conv_raw[1]
        if isinstance(conv, dict) and conv.get('paragraphs'):
            all_conv.append(conv['paragraphs'])
        if isinstance(conv, dict) and conv.get('paragraphs'):
            all_conv.append(conv['paragraphs'])
        elif isinstance(conv, dict) and conv.get('sentences'):
            all_conv.append(conv['sentences'])
        elif isinstance(conv, dict) and conv.get('analysis'):
            all_conv.append(conv['analysis'])
        elif isinstance(conv, list) and conv[0].get('text'):
            all_conv.append(conv)
        else:
            print(conv)
    return all_conv

def similar(a, b):
        return SequenceMatcher(None, a, b).ratio()
        
def assess_qual_reconstruction(user, assistant):
    """
     - user: paragraphs in a list given by user
     - assitant: texts in a list given back by gpt
    """
    assert len(user) == len(assistant), "both list must be of the same len"
    for i in range(len(user)):
        raw_text = ' '.join(user[i])
        print(similar(raw_text, reconstruct_text_from_reply(assistant[i])))

instructions = dat[0]['content'].split("\n##\n")[0]
user_content = [_['content'].split("\n##\n")[1:] for _ in dat if _['role'] == 'user']

all_conv_raw = []
for conv in out:
     if conv['role'] == 'assistant':
        if len(conv['content'].split("\n##\n")) == 1:
            all_conv_raw.append(json.loads(conv['content']))
        elif len(conv['content'].split("\n##\n")) == 2:
            all_conv_raw.append(json.loads(conv['content'].split("\n##\n")[0]))
        else:
            print(conv)

all_conv_raw[0]

all_conv = parse_raw_conv(all_conv_raw)

BoWuser = set(' '.join(user_content[2]).split(" "))
BoWreconstructed = set(' '.join([_['text'] for _ in all_conv[2]]).split(" "))

BoWuser - BoWreconstructed


assess_qual_reconstruction(user_content, all_conv)

all_conv_flatten = flatten(all_conv)

lookup_authors = {'W.-E.': 'Wynne-Edwards',
                  'LACK': 'D. Lack',
                  'Lack': 'D. Lack',
                  'V. C. Wynne-Edwards': 'Wynne-Edwards',
                  'V. C. WYNNE-EDWARDS': 'Wynne-Edwards',
                  'P. L. ERRING-TON': 'P. L. Errington',
                  '0. KALELA': 'Kalela',
                  'WRIGHT': 'Wright',
                  'CARR-SAUNDERS': 'Carr-Saunders',
                  'BREDER and COATES, 1932': 'Breder and Coates',
                  'KARL VON FRISCH': 'Karl Von Frisch',
                  'PFEIFFER, 1962': 'Pfeiffer',
                  'ELLINOR BRO LARSEN': 'Ellinor Bro Larsen',
                  'WESENBERG-LUND 1943': 'Wesenberg-Lund',
                  'BRAESTRUP 1963': 'Braestrup',
                  'Gause': 'Gause',
                  'Fisher': 'R. A. Fisher',
                  'Fisher (24)': 'R. A. Fisher',
                  'Hamilton (39)': 'Hamilton'
                  }

def rename_key(x, new_name, old_name):
    x[new_name] = x[old_name]
    del x[old_name]

[rename_key(conv, 'mentions', 'references') for conv in all_conv_flatten if conv.get('references')];

author_mentions_raw = [conv.get('mentions') for conv in all_conv_flatten]

def clean_author_mentions():
    new_authors_mentions = []
    for i, authors in enumerate(author_mentions_raw):
        # authors = author_mentions_raw[0]
        if authors is not None:
            print(i)
            if isinstance(authors, list) and len(authors) == 1:
                author = authors[0]
                author['source'] = 'Alexander'
                if author.get('name'):
                    author['name'] = re.sub(' ?\(\d+\)', "", author['name'])
                    rename_key(author, 'target', 'name')
                    new_authors_mentions.append(author)
                elif author.get('text'):
                    author['text'] = re.sub(' ?\(\d+\)', "", author['text'])
                    rename_key(author, 'target', 'text')
                    new_authors_mentions.append(author)
            elif isinstance(authors, list) and len(authors) > 1:
                for author in authors:
                    # author = authors[0]
                    author['source'] = 'Alexander'
                    if author.get('name'):
                        author['name'] = re.sub(' ?\(\d+\)', "", author['name'])
                        rename_key(author, 'target', 'name')
                        new_authors_mentions.append(author)
                    elif author.get('text'):
                        author['text'] = re.sub(' ?\(\d+\)', "", author['text'])
                        rename_key(author, 'target', 'text')
                        new_authors_mentions.append(author)
            else:
                print(authors)

    for auth in new_authors_mentions:
        if auth['target'] and lookup_authors.get(auth['target']):
            auth['target'] = lookup_authors[auth['target']]

clean_author_mentions()

with open(f'{target_article}_gpt35turbo_clean.json', 'w') as f:
    f.write(json.dumps(all_conv_flatten, indent=4))
   
all_conv_flatten[0]['sentiment_score']