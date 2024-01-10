import os
import json
import sys
import time
import openai

sys.path.insert(0,'/app/')

from pyknowledgegraph import pdf

infile = sys.argv[1]
outfile = sys.argv[2]
title = sys.argv[3]
    
openai.api_key = os.getenv("OPENAI_API_KEY")

content = open(infile).read()
new_sentences = []
paragraphs = content.split("\n\n")
for p in paragraphs:
    new_sentences.extend(p.replace("\n"," ").replace("- ","").split("."))
    
import string
printable = set(string.printable)

valid_sentences = []
for i, sent in enumerate(new_sentences):
    sent = ''.join(filter(lambda x: x in printable, sent))
    
    valid_sentences.append(sent.strip())

contents = "\n".join(valid_sentences)

sections = {}
sections['title'] = title
sections['contents'] = contents
open(outfile,"w").write(json.dumps(sections))