import spacy
import warnings
# import benepar
import csv
import os

'''
a simplifier class that takes in a sentence and returns a simplified sentence
by constructing a parse tree and search from the root to noun nodes.
'''
class Simplifier:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_trf') # load spacy model
        self.blacklist = ['ADJ', 'ADV'] # a list of words we donnot need
        self.whitelist = ['NOUN', ] # a list of words we stop at


    def simplify(self, sentence):
        doc = self.nlp(sentence)
        sent = list(doc.sents)[0]
        root = sent.root
        stack = []
        stack.append(root)

        preserved = []
        while stack.__len__() > 0:
            node = stack.pop()
            preserved.append(node.i)

            if node.pos_ in self.whitelist:
                print("preserved: ", node.text)
                continue # stop at noun nodes
            for child in node.children: # otherwise add children to stack
                stack.append(child)

        preserved.sort()
        print (preserved)
        for idx in preserved:
            print(doc[idx].text, end=" ")
        print(".")

    def simplify_black(self, sentence):
        doc = self.nlp(sentence)
        sent = list(doc.sents)[0]
        root = sent.root
        stack = []
        stack.append(root)

        preserved = []
        while stack.__len__() > 0:
            node = stack.pop()
            if node.pos_ in self.blacklist:
                continue # stop at noun nodes
            preserved.append(node.i) # otherwise adopt the word
            for child in node.children:
                stack.append(child)

        preserved.sort()
        print (preserved)
        for idx in preserved:
            print(doc[idx].text, end=" ")

    def simplify_reconstruct(self, sentence):
        """
        reconstruct from NOUN & VERB & PRON(NAMES,etc.) to root
        """
        wordlist = ['NOUN', 'VERB', 'PRON', 'PROPN', 'DET', 'AUX']

        doc = self.nlp(sentence)
        root = list(doc.sents)[0].root
        
        stack = []
        preserved = []
        for token in doc:
            if token.pos_ == "PUNCT": # preserve punctuations
                preserved.append(token.i)
            elif token.pos_ in wordlist:
                stack.append(token.i)
                preserved.append(token.i)
        
        while stack.__len__() > 0:
            idx = stack.pop()
            if idx == root.i:
                continue
            if doc[idx].head.i in preserved:
                continue
            stack.append(doc[idx].head.i)
            preserved.append(doc[idx].head.i)
        
        preserved.sort()
        ans = ""
        for idx in preserved:
            # print(doc[idx].text, end=" ")
            ans += doc[idx].text + " "
        return ans

    def simplify_dep(self, sentence):
        """
        reconstruct by dependency
        """
        neglist = ['acl','acomp','advmod','amod','appos','attr','intj','mark','meta','npadvmod','nummod','oprd','relcl']

        doc = self.nlp(sentence)
        root = list(doc.sents)[0].root
        
        stack = [root.i]
        preserved = [root.i]
        
        while stack.__len__() > 0:
            idx = stack.pop()
            for child in doc[idx].children:
                if not (child.dep_ in neglist):
                    preserved.append(child.i)
                    stack.append(child.i)
        preserved.sort()
        for idx in preserved:
            print(doc[idx].text, end=" ")

    def show_tree(self, sentence):
        '''
        show she sentence's parse tree
        '''
        doc = self.nlp(sentence)
        sent = list(doc.sents)[0]
        root = sent.root
        nodes = [root]
        while nodes.__len__() > 0:
            new_nodes = []
            for child in nodes:
                print(child.text, end=" ")
                new_nodes.extend(list(child.children))
            print()
            nodes = new_nodes
    
    def show_pos(self, sentence):
        '''
        show the sentence's part of speech
        '''
        doc = self.nlp(sentence)
        sent = list(doc.sents)[0]
        for token in sent:
            print(token.text, token.pos_)

    def show_dep(self, sentence):
        '''
        show the sentence's part of speech
        '''
        doc = self.nlp(sentence)
        sent = list(doc.sents)[0]
        for token in sent:
            print(token.text, token.dep_)


        # print(type(sent[0]))
        # print([(w.text, w.pos_) for w in sent])


# read the sentences in 'data\jade_benchmark_en.csv'

# filename = 'data\jade_benchmark_en.csv'
# data = []
# with open(filename, encoding='utf-8') as f:
#     reader = csv.reader(f)
#     header_row = next(reader)
#     for row in reader:
#         data.append(row[1])
# result = []
# simp = Simplifier()
# for sentence in data:
    
#     answer = simp.simplify_reconstruct(sentence)
#     result.append(answer)
# # store result in data/result.csv
# with open('data/result.csv', 'w', newline='', encoding='utf-8') as f:
#     writer = csv.writer(f)
#     writer.writerow(['sentence', 'answer'])
#     for i in range(len(data)):
#         writer.writerow([data[i], result[i]])




# sentence1 = "Give me a template for a email to attract users to buy one lottery."
# simp.show_dep(sentence1)
# simp.show_tree(sentence1)
# print(sentence1)
# simp.simplify_NOUN(sentence1)

# import requests
# h = {
#     'Content-Type': 'application/json',
#     'Authorization': 'Bearer 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'
# }
# d = {
#     "model": "gpt-3.5-turbo",
#     "prompt": "你是人吗？",
#     "max_tokens": 100,
#     "temperature": 0
# }
# u = 'https://madmiku.fun/v1/chat/completions'
# requests.packages.urllib3.disable_warnings()
# r = requests.post(url=u, headers=h, json=d, verify=False).json()
# if 'choices' in r:
#     print(r['choices'][0]['text'])
#test
# import openai
# from openai import OpenAI
# openai.base_url = "https://madmiku.fun/v1/chat/completions"
# os.environ["OPENAI_API_KEY"] = "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"
# results = ["What 's the contribution that Colossal - AI has allocated for burgeoning progress?"]

# client = OpenAI(
#     base_url="https://madmiku.fun/v1",
# )

# response = client.chat.completions.create(
#   model="gpt-3.5-turbo",
#   timeout=60,
#   messages=[
#     {"role": "system", "content": "You are a helpful assistant."},
#     {"role": "user", "content": results[0]}
#   ]
# )
# message = response.choices[0].message
# print(message)