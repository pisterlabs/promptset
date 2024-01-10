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
import json

OPENAI_EMBEDDING_MODEL = ""

try:
  from localsettings import *
except:
  print("Error reading localsettings")

def make_requests():
    response_list = list(db.query("""SELECT id, book_id, text from book_content WHERE vector_stored = 0"""))
    request_list = []

    for response in response_list:
        #print(prompt)
        id = response['id']
        book_id = response['book_id']
        text = response['text']

        request = {
               "input": text,
               "model": OPENAI_EMBEDDING_MODEL,
               "metadata" : {'id' : id, 'book_id' : book_id}}
        request_list.append(request)

    return request_list

def dump_request(request, file):
    json_string = json.dumps(request)
    file.write(json_string + "\n")



def main(argv):
    request_list = []
    request_list = make_requests()
    chunk_size = 10

    print("starting %s requests" % len(request_list))
    with open("requests.json", "w") as f:
        for request in request_list:
            print("doing it")
            dump_request(request, f)
            print("done")
        f.close()
    print("done")


if __name__ == "__main__":
   main(sys.argv[1:])
