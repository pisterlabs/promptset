import os
import openai
import spacy

import sys
infile = sys.argv[1]
content = open(infile).read()
outfile = sys.argv[2]

from spellchecker import SpellChecker
import torch
import sys
import numpy as np
import pandas as pd
import re

from pysbd.utils import PySBDFactory
from spacy.language import Language

nlp = spacy.blank('en')
# Caution: works with spaCy<=2.x.x
@Language.factory('PySBDFactory')
def get_PySBDFactory(nlp, name):
    return PySBDFactory(nlp)

nlp.add_pipe('PySBDFactory')

from transformers import GPT2Tokenizer, GPT2LMHeadModel

assert torch.cuda.is_available()

def clean_and_score_content(content):
    # Check to see if it is better to combine adjacent words as sometimes they are split incorrectly.
    spell = SpellChecker()
    new_sentences = []
    sentences = content.split("\n")
    for sentence in sentences:
        new_words = []
        words = sentence.split(" ")
        i = 0
        while i < len(words)-1:
            word1 = words[i].replace("-","")
            word2 = words[i+1].replace("-","")
            if len(list(spell.unknown([word1,word2]))) > 0 and len(list(spell.unknown([word1+word2]))) == 0: # at least one is unknown
                word1 = word1+word2 # better to combine the words
                i+=1
            new_words.append(word1)
            i+=1
        new_words.append(words[-1])
        new_sentences.append(" ".join(new_words))

    # Load pre-trained model (weights)
    with torch.no_grad():
        model = GPT2LMHeadModel.from_pretrained('gpt2')
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        model = model.to(device)
        model.eval()
    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    def score(sentence):
        tokenize_input = tokenizer.encode(sentence)
        tensor_input = torch.tensor([tokenize_input]).to(device)
        loss=model(tensor_input, labels=tensor_input)[0]
        #return np.exp(loss.detach().numpy())
        return np.exp(loss.detach().cpu().numpy())

    new_sentences2 = []
    i=0
    while i < len(new_sentences)-1:
        if len(new_sentences[i].strip()) > 0 and len(new_sentences[i+1].strip()) > 0:
            new_sentences2.append(new_sentences[i]+" "+new_sentences[i+1])
            i+=1
        else:
            new_sentences2.append(new_sentences[i])
        i+=1
    new_sentences2.append(new_sentences[-1])
    paragraphs = ("\n".join(new_sentences2)).split("\n\n")
    new_paragraphs = []
    for p in paragraphs:
        p = p.replace("\n"," ").strip()
        if len(p) > 0:
            doc = nlp(p)
            new_sentences2 = [str(s) for s in list(doc.sents)]
            new_paragraphs.append(" ".join(new_sentences2))

    paragraphs = new_paragraphs
    print("\n\n".join(paragraphs[:10]))

    pnums = []
    sent_scores = []
    new_sents = []
    snums = []
    for i,p in enumerate(paragraphs):
        print("%d/%d"%(i,len(paragraphs)))
        for j,sent in enumerate(p.split(".")):
            sent = re.sub(' +', ' ', sent).strip()
            if len(sent) == 0:
                continue
            if len(sent) > 1024: # max limit
                sent = sent[:1024]
            s = score(sent)
            words = sent.split(" ")
            s2 = None
            if len(words) > 1 and words[0][0].isupper() and words[1][0].isupper():
                new_sent = " ".join(words[1:])
                s2 = score(new_sent)
            if s2 is not None and s2 < s:
                sent = new_sent
            pnums.append(i)
            snums.append(j)
            sent_scores.append(s)
            new_sents.append(str(sent))

    results = pd.DataFrame({"Paragraph ID": pnums, "Sentence ID": snums, "Score": sent_scores, "Sentence": new_sents})
    return results

results = clean_and_score_content(content)
print(results)
results.to_csv(outfile)