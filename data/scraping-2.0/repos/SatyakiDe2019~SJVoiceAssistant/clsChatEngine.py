#####################################################
#### Written By: SATYAKI DE                      ####
#### Written On: 26-Dec-2022                     ####
#### Modified On 28-Jan-2023                     ####
####                                             ####
#### Objective: This is the main calling         ####
#### python script that will invoke the          ####
#### ChatGPT OpenAI class to initiate the        ####
#### response of the queries in python.          ####
#####################################################

import os
import openai
import json
from clsConfigClient import clsConfigClient as cf

import sys
import errno

# Disbling Warning
def warn(*args, **kwargs):
    pass

import warnings
warnings.warn = warn

###############################################
###           Global Section                ###
###############################################

CODE_PATH=str(cf.conf['CODE_PATH'])
MODEL_NAME=str(cf.conf['MODEL_NAME'])

###############################################
###    End of Global Section                ###
###############################################

class clsChatEngine:
    def __init__(self):
        self.OPENAI_API_KEY=str(cf.conf['OPENAI_API_KEY'])

    def findFromSJ(self, text):
        try:
            OPENAI_API_KEY = self.OPENAI_API_KEY

            # ChatGPT API_KEY
            openai.api_key = OPENAI_API_KEY

            print('22'*60)

            try:
                # Getting response from ChatGPT
                response = openai.Completion.create(
                engine=MODEL_NAME,
                prompt=text,
                max_tokens=64,
                top_p=1.0,
                n=3,
                temperature=0,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                stop=["\"\"\""]
                )
            except IOError as e:
                if e.errno == errno.EPIPE:
                    pass

            print('44'*60)
            res = response.choices[0].text

            return res

        except IOError as e:
            if e.errno == errno.EPIPE:
                pass

        except Exception as e:
            x = str(e)
            print(x)

            print('66'*60)

            return x
