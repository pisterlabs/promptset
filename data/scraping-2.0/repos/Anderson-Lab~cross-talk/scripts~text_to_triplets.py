import os
import json
import sys
import time
import openai
import numpy as np

import string
printable = set(string.printable)

sys.path.insert(0,'/app/')

from pyknowledgegraph import pdf

max_tokens = 4097
min_num_words_in_sentences = 4
min_num_words_in_paragraph = min_num_words_in_sentences
min_len_sentence = min_num_words_in_sentences*4

infile = sys.argv[1]
outfile = sys.argv[2]
title = sys.argv[3]

openai.api_key = os.getenv("OPENAI_API_KEY")

import json
contents = json.loads(open(infile).read())

all_sentences_with_results = []
all_sentences_without_results = []

output = []
paragraphs = contents[title].split("\n\n")
for paragraph in paragraphs:
    output.append({"paragraph": {"text":paragraph}})
    num_words = len(paragraph.split(" "))
    if num_words < min_num_words_in_paragraph:
        continue
    #print(paragraph)
    response = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=[
          {"role": "system", "content": "You will be provided a paragraph, and your task is to extract all the sentences that describe scientific results. If there are no matching sentences, respond None. Your answer should be a numbered list."},
          {"role": "user", "content": paragraph }
      ],
      temperature=0,
      max_tokens=1024,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0
    )
    sentences = response['choices'][0]['message']['content'].split("\n")
    if sentences[0].strip() == "None":
        continue
    sentences_with_results = [" ".join(s.split(" ")[1:]) for s in sentences] # Remove the number
    sentences_without_results = []
    for sent in paragraph.strip().split(". "):
        if sent[-1] != ".":
            sent = f"{sent}."
        if f"{sent}." not in sentences:
            sentences_without_results.append(f"No results: {sent}")
    #all_sentences_with_results.extend(sentences_with_results)
    #all_sentences_without_results.extend(sentences_without_results)
    output[-1]["paragraph"]["sentences_with_results"] = [{"text":s, "clauses": []} for s in sentences_with_results]
    output[-1]["paragraph"]["sentences_without_results"] = [{"text":s, "clauses": []} for s in sentences_without_results]
    
    for j,sentence in enumerate(sentences_with_results):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You will be provided a sentence, and your task is break that sentence into independent clauses. Your answer should be a numbered list."},
                {"role": "user", "content": sentence }
            ],
            temperature=0,
            max_tokens=1024,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        #print("Sentence:",sentence)
        clauses = [" ".join(clause.split(" ")[1:]) for clause in response['choices'][0]['message']['content'].split("\n")]
        output[-1]["paragraph"]["sentences_with_results"][j]["clauses"] = [{"text": c} for c in clauses]
        for i, clause in enumerate(clauses):            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You will be provided a sentence, and your task is split it into a subject, verb, and object. Return your answer as JSON with keys subject, verb, and object."},
                    {"role": "user", "content": sentence }
                ],
                temperature=0,
                max_tokens=1024,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )   
            content = response['choices'][0]['message']['content']
            output[-1]["paragraph"]["sentences_with_results"][j]["clauses"][i]["triplet"] = json.loads(content)

open(outfile,"w").write(json.dumps(output))
exit(0)


all_sentences = contents[title].split(".")

"""
You will be provided a paragraph, and your task is to extract all the sentences that describe results of a study.

Can you extract the biomedical PICO elements from text?"""

i = 0
while i < len(all_sentences):
    sent = all_sentences[i]
    sent = ''.join(filter(lambda x: x in printable, sent)).strip()
    num_words = len(sent.split(" "))
    if num_words < min_num_words_in_sentences or len(sent) < min_len_sentence:
        continue

    response = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=[
          {"role": "system", "content": "You will be provided a sentence, and your task is to split it into independent clauses and then split into the form subject, verb, and object."},
          {"role": "user", "content": sent }
      ],
      temperature=0,
      max_tokens=1024,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0
    )

    response = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=[
          {"role": "system", "content": "You will be provided a sentence, and your task is to split it into independent clauses and then split into the form subject, verb, and object."},
          {"role": "user", "content": sent }
      ],
      temperature=0,
      max_tokens=1024,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0
    )
    #response = openai.Completion.create(
    #  engine="text-davinci-003",
    #  prompt=gpt_prompt,
    #  temperature=0,
    #  max_tokens=max_tokens-len(gpt_prompt),
    #  top_p=1.0,
    #  frequency_penalty=0.0,
    #  presence_penalty=0.0
    #)

    print(sent)
    for j,response in enumerate(response['choices']):
        print(response['message']['content'])
        text = response['text'].strip()
        exit(0)
        continue
        ixs = text.split("\n")[0].replace("Text","").replace(" and ",",").replace("and","").replace(".","").split(",")
        cleaned_ixs = []
        for ix in ixs:
            try:
                if "-" in ix:
                    start,end = ix.split("-")
                    for j in range(int(start)-1,int(end)):
                        cleaned_ixs.append(j)
                else:
                    cleaned_ixs.append(int(ix)-1)
            except:
                print("Error somewhere in",text)
    #print("\n".join(valid_sentences))
    #print(gpt_prompt+text)
    #if text.startswith("Yes"):
    #    valid_sentences.append((sent+".").strip())


    print(i+1,"out of",len(all_sentences),"sentences or",int(100*(i+1)/len(all_sentences)),"percent.")
    #if i % 50 == 0:
    #    print('Sleeping')
    #    time.sleep(60)

contents = "\n".join(valid_sentences)

sections = {}
sections['title'] = title
sections['contents'] = contents
open(outfile,"w").write(json.dumps(sections))
