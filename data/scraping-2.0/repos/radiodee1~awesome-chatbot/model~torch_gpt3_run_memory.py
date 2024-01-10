#!/usr/bin/env python3

'''
    code by TaeHwan Jung(@graykode)
    Original Paper and repository here : https://github.com/openai/gpt-2
    GPT2 Pytorch Model : https://github.com/huggingface/pytorch-pretrained-BERT

MIT License

Copyright (c) 2019 OpenAI, HugginFace Inc. team. and TaeHwan Jung

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

'''

import os
import sys
#sys.path.append('torch_gpt2')
sys.path.append('..')
import torch
import random
import argparse
import numpy as np
import json
import re
import datetime
import openai
#from functools import lru_cache
#from GPT2.model import (GPT2LMHeadModel)
#from GPT2.utils import load_weight
#from GPT2.config import GPT2Config
#from GPT2.sample import sample_sequence
#from GPT2.encoder import get_encoder
#from GPT2.encoder import Encoder
#from model.nmt_aiml_commands import Commands

realpath = os.path.dirname(os.path.realpath(__file__))
endoftext = '<|endoftext|>'




class Lang:
    def __init__(self, name, limit=None):
        self.name = name
        self.limit = limit
        self.word2index = {}
        self.word2count = {}
        self.index2word = {}
        self.n_words = 0  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if self.limit is None or self.n_words < self.limit :
            if word not in self.word2index:
                self.word2index[word] = self.n_words
                self.word2count[word] = 1
                self.index2word[self.n_words] = word
                self.n_words += 1
            else:
                self.word2count[word] += 1

class NMT:
    def __init__(self):
        self.args = None
        self.state_dict = None
        self.config = None
        self.device = None
        self.model = None
        self.enc = None

        self.hp_config = None

        self.output_lang = None
        self.commands = None

        self.common = ''
        self.previous_sentences = []
        self.gather_sentences = False
        self.recent_in = ''
        self.recent_text = ''
        self.save_num = 100
        self.save_on_failure = False
        self.use_common = True

        self.q_string = ['Q: ']
        self.a_string = ['A: ']

        self.name = 'Jane'

        if True:
            self.q_string = [ 'Question: ', 'Q :', 'Q.']
            self.a_string = [  self.name + ': ', 'A :', 'Answer:', 'A.', 'A:']

    def setup_for_interactive(self):
        self.get_args()
        #self.load_state_dict()
        #self.load_model()

        #self.commands = Commands()

        ## this is not used but is required for bot software...
        self.output_lang = Lang('lang')
        #for i in range(len(self.enc.encoder.items())):
        #    self.output_lang.addWord(self.enc.decode([i]))

        ## do this also with each input...
        self.prepare_common()

    def prepare_common(self):
        self.common = ''
        a_chars = self.a_string[0]
        q_chars = self.q_string[0]

        now = datetime.datetime.now()
        time = now.strftime("%I:%M %p")
        date = now.strftime("%B %d, %Y")
        name = self.name
        profession = 'student'
        location = 'New York'
        key_action_string = '\n ' + a_chars + 'play media.\n'
        '''
        key_phrases = [
            'Play music? ' + key_action_string,
            'Play movies? ' + key_action_string,
            'Play radio? ' + key_action_string,
            'Play any song? ' + key_action_string,
            'Play any video? ' + key_action_string,
            'Play any movie? ' + key_action_string,
            'Play a song? ' + key_action_string,
            'Play a video? ' + key_action_string,
            'Play a movie? ' + key_action_string,

        ]## doesn't work??
        '''

        #self.common += self.a_string[0] + 'I am ' + self.a_string[0] + '. \n '
        self.common += a_chars + 'Hello' + '.\n '
        self.common += a_chars + 'My name is ' + name + '.\n '
        self.common += a_chars + 'The time is ' + time + ' ' + date + '.\n '
        self.common += a_chars + 'My job is as a ' + profession + '.\n '
        self.common += a_chars + "I am in " + location + '. \n'
        #if self.args.apps and False:
        #    self.common +=' ' + ' '.join([q_chars + i for i in key_phrases])

    def get_sentence(self, i):
        a_chars = '' # self.a_string[0]
        q_chars = '' # self.q_string[0]

        if self.use_common:
            self.recent_in = i
            if self.save_num > -1:
                self.previous_sentences = self.previous_sentences[-self.save_num:]
            s = []
            for k in self.previous_sentences :
                k = k.strip().strip('..')
                if not k.endswith('?'):
                    k = a_chars + k + '.\n'
                else:
                    k = q_chars + k + '\n'
                    pass
                s.append(k)

            i =  '\n\n' + self.q_string[0] + i.capitalize()  # + '\n' + endoftext
            s.append(i)
            self.prepare_common()
            i = self.common + "\n" + "\n" + ' ' +  ' '.join(s)
            print('',"+" * 10, '\n', i, '\n','+' * 10)
        i = self.prepare_input(i)

        self.args.text = i
        text = self.text_generator()
        #self.recent_text = text
        #self.prep_recent()

        ##if not self.args.quiet or True: print(text)

        text = self.prepare_output(text)
        self.recent_text = text
        self.prep_recent()
        text = re.sub(endoftext, '', text)
        print(text,"<")

        ## if you want to launch apps !!
        #if self.args.apps is True:
        #    if self.commands.is_command(self.recent_in):
        #        self.commands.do_command(self.recent_in)
        #        #self.previous_sentences = []
        return text

    def loop(self):
        while True:
            try:
                i = input("> ")
                self.get_sentence(i)
            except EOFError:
                print()
                exit()
            except KeyboardInterrupt:
                print()
                exit()

    def prepare_input(self, i):
        #self.random_seed()

        if True:
            i = self.q_string[0] + i + '?'
        else:
            i = i + "?"
        return i

    def prepare_output(self, i):
        char_end = ['?','!']
        contains_junk = False
        char_junk = [i for i in '{[]}@$%^&#']
        out = []
        for ii in i:
            if ii.strip() != "" or ii == ' ':
                if ii not in ['*']:
                    out.append(ii)
            elif len(out) > 1:
                break
            if ii in char_end:
                break
            if ii in char_junk:
                contains_junk = True
                break
        i = ''.join(out)

        i = i.strip()

        for z in self.a_string:
            z = z.lower()
            if i.lower().startswith(z): i = i[len(z):]

        for z in self.q_string:
            z = z.lower()
            if i.lower().startswith(z): i = i[len(z):]

        start = i[:]
        num = 0
        default = ''
        while num < 5:

            i = start[:]
            out = []
            for ii in i.split(' '):

                out.append(ii)

                if (ii.endswith('.') or ii.endswith('!') or ii.endswith('?')) and len(ii) > 1 and ii.count('.') >= 1:
                    break
            i = ' '.join(out)

            if num == 0: default = i

            if (i.strip() + '.' not in self.previous_sentences or len(start) <= 1) and len(i.strip()) > 0:
                if not self.args.quiet: print('take first:', i.strip())
                break
            else:
                if i.strip() == '':
                    i = ' '
                if not self.args.quiet: print('take next:', '-'+i.strip()+'-')
                start = start[len(i):]
            num += 1

        if i.strip() == '': i = default

        i = re.sub('[;]','',i)
        if contains_junk is True or False:
            i = ''

        if self.gather_sentences:
            i = i.strip()
            for z in self.q_string:
                z = z.lower()
                if i.lower().startswith(z): i = i[len(z):]

            i = re.sub('[?!]', ' ', i)

            #if self.recent_in.strip() + '?' not in self.previous_sentences or True:

            self.previous_sentences.append(self.recent_in.strip() + "? " + i)

            #elif self.save_on_failure:
            #    self.recent_text = re.sub('[\n]',' ', self.recent_text)
            #    l = self.a_string + self.q_string + ['Q.', 'A.']
            #    for k in l:
            #        self.recent_text = re.sub(k, '', self.recent_text)
            #    self.previous_sentences.append(self.recent_in.strip() + '?' + self.recent_text + '\n')
        return i

    def prep_recent(self):
        self.recent_in = self.q_string[0] + self.recent_in.strip('.').lower()
        self.recent_text = self.a_string[0] + self.recent_text.strip('.').lower()
        y = 'yes'
        n = 'no'
        for a in self.previous_sentences:
            a = a.replace('.', '')
            if (self.recent_text is not None and len(self.recent_text.split(' ')) == 1 and self.recent_text.lower() in a.lower().split(' ')):
                if y not in self.recent_text.lower() and n not in self.recent_text.lower():
                    #self.recent_text = None
                    pass
            if self.recent_in is not None and len(self.recent_in.split(' ')) == 1 and self.recent_in.lower() in a.lower().split(' '):
                #self.recent_in = None
                pass
            if self.recent_text is not None and self.recent_text.lower().strip() == a.lower().strip():
                if y not in self.recent_text.lower() and n not in self.recent_text.lower():
                    #self.recent_text = None
                    pass
            if self.recent_in is not None and self.recent_in.lower().strip() == a.lower().strip():
                #self.recent_in = None
                pass

        if self.recent_in is not None and self.recent_text is not None and 'time' not in self.recent_in:
            self.previous_sentences.extend([self.recent_in, self.recent_text])


        if self.save_num > -1:
            self.previous_sentences = self.previous_sentences[-self.save_num:]

        #print(self.previous_sentences)
        s = ''
        for k in self.previous_sentences:
            k = k.strip().strip('.').strip('\n')
            for z in self.a_string + self.q_string:
                z = z.lower()
                if k.lower().startswith(z) and False: k = k[len(z):]
            if len(k) > 0:
                s += k + '.\n'
        #s = ['---'] + s + ['---']
        self.sentences_formatted = s
        #return s


    #########################################

    def get_args(self ):
        parser = argparse.ArgumentParser()
        parser.add_argument("--text", type=str, required=False)
        parser.add_argument("--quiet", type=bool, default=True)
        parser.add_argument("--nsamples", type=int, default=1)
        parser.add_argument('--unconditional', action='store_true', help='If true, unconditional generation.')
        parser.add_argument("--batch_size", type=int, default=-1)
        parser.add_argument("--length", type=int, default=25)
        parser.add_argument("--temperature", type=float, default=0.0001)
        parser.add_argument("--top_k", type=int, default=40)
        parser.add_argument("--apps", type=bool, required=False, default=False)
        parser.add_argument("--source_file", type=str, required=False, default='torch_gpt2/GPT2/gpt2-pytorch_model.bin')
        self.args = parser.parse_args()

    def text_generator(self):
        ## replace for gpt3

        homepath = os.path.expanduser('~')
        z = open(homepath + "/bin/awesome-chatbot-openai.txt", 'r')
        z = z.readline().strip()

        openai.api_key = z
        ##print(openai.Engine.list())

        openai.api_key = z
        xx = openai.Completion.create(
            engine="davinci",
            prompt= self.args.text,
            max_tokens=100,
            ##stop=["\n"],
            temperature=0.001
        )
        xx = xx['choices'][0]['text']
        print(xx)
        return xx
        


if __name__ == '__main__':

    n = NMT()
    n.setup_for_interactive()
    n.loop()



