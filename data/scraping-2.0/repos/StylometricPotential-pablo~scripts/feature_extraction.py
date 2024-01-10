from string import punctuation
from collections import Counter
from itertools import chain
import re
import nltk
from nltk.corpus import stopwords
import openai
stop_words = set(stopwords.words('english'))
import gensim
from gensim.models import Word2Vec
import numpy as np





class documentFeatures():

    punctuationList = ".?\"',-!:;()[]/"

    # Calculating the relative rarity of words (doesn't scale well with size) 
    def calculate_hapax_legomena(self, path):
        file = open(path)
        list_of_words = re.findall('\w+', file.read().lower())
        freqs = {key: 0 for key in list_of_words}
        for word in list_of_words:
            freqs[word] += 1
        for word in freqs:
            if freqs[word] == 1:
                self.hapax_legomena.append(word)
            if freqs[word] == 2:
                self.dislegomenon.append(word)
    

    def calculate_texts(self, path):
        with open(path) as f:
            contents = f.read()
            
            
        doc = []
        with open(path, 'r') as f:

            for line in f:
                for word in line.split():
                    if word not in stop_words:
                        doc.append(word)
                    if (len(word)<4):
                        self.short_word_number+=1
                    if (word in self.uni_grams):
                        self.uni_grams[word]+=1
                    else:
                        self.uni_grams[word]=1
                    self.word_number+=1
        self.uni_grams = dict(sorted(self.uni_grams.items(), key=lambda item: item[1], reverse=True))

        i = 0
        for key in self.uni_grams.keys():
            if i == 100:
                break
            self.uni_gram_frequency.append(self.uni_grams[key])
            i+=1

        input = " ".join(doc)
        bi_grams = {}

        for i in range(len(doc)-1):
            thisGram = doc[i] + doc[i+1]
            thisGram = thisGram.lower()
            if thisGram in bi_grams:
                bi_grams[thisGram]+=1
            else:
                bi_grams[thisGram]=1
        bi_grams = dict(sorted(bi_grams.items(), key=lambda item: item[1], reverse=True))

        

        self.periods = input.count('.')
        self.commas = input.count(',')
        self.bi_grams = bi_grams

        i = 0
        for key in self.bi_grams.keys():
            if i == 20:
                break
            self.bi_gram_frequency.append(self.bi_grams[key])
            i+=1
        print(bi_grams) 
        print(self.bi_gram_frequency)
        # print(input)

        response = openai.Embedding.create(
        input= input,
        model="text-embedding-ada-002"
        )
        embeddings = response['data'][0]['embedding']


        self.embedding = embeddings


    def calculate(self, path):
        self.calculate_texts(path)
        self.calculate_hapax_legomena(path)

            
         

    def __init__(self, path):
        self.text_length = 0        #text length
        self.word_number = 0        #number of words
        self.word_length = 0        #average worth length
        self.short_word_number = 0  #number of short words
        self.digit_cap_prop = 0     #proportion of digits + capital letters
        self.letter_freqs  = 0      #individual letters + digit frequencies
        self.digit_freqs = 0        #frequency of digits
        self.hapax_legomena = []    #hapax-legomena
        self.dislegomenon = []
        self.richness = 0           #measure of text richness
        self.twelve_freq = [0] * 12 #frequency of 12 punctuation marks
        self.uni_grams = {}         #most common words
        self.uni_gram_frequency = []
        self.vector = 0
        self.embedding = 0
        self.bi_grams = {}
        self.bi_gram_frequency = []
        self.periods = 0
        self.commas = 0

        
        self.calculate(path)

    