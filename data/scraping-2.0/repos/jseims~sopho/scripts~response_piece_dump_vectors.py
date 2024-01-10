#!/usr/bin/env python

import requests
import sys
import time
import db
import openai
import hashlib
import random
import argparse  # for running script from command line
from pathlib import Path
import ast
import json

OPENAI_EMBEDDING_MODEL = ""

try:
  from localsettings import *
except:
  print("Error reading localsettings")

def make_requests():
    response_list = list(db.query("""SELECT * FROM response_piece WHERE matches is NULL"""))
    request_list = []

    for response in response_list:
        #print(prompt)
        id = response['id']
        text_type = response['text_type']

        if text_type == 'plain_text':
            continue

        text = response['text']
        if text_type == "test_question":
            try:
                item = ast.literal_eval(text)
                text = item['explanation']
            except Exception as e:      
                print("Error in parse_response_text")
                print(id)
                print(text)
                print(e)
                sys.exit()

        request = {
               "input": text,
               "model": OPENAI_EMBEDDING_MODEL,
               "metadata" : {'id' : id}}
        request_list.append(request)

    return request_list

def dump_request(request, file):
    json_string = json.dumps(request)
    file.write(json_string + "\n")



def main(argv):
    request_list = []
    request_list = make_requests()

    print("starting %s requests" % len(request_list))
    with open("requests.json", "w") as f:
        i = 0
        for request in request_list:
            #print("doing request %s out of %s" % (i, len(request_list)))
            dump_request(request, f)
            i = i + 1
        f.close()
    print("done")


if __name__ == "__main__":
   main(sys.argv[1:])
