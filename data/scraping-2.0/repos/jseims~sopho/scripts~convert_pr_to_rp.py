#!/usr/bin/env python

# converts the old prompt_response tables (with a large blob of text) into the response_piece table where every paragraph and question is a distinct entry
import requests
import sys
import time
import db
import openai
import hashlib
import random
import json

try:
  from localsettings import *
except:
  print("Error reading localsettings")


def dictValuePad(key):
    return '%(' + str(key) + ')s'

def insertFromDict(table, dict):
    """Take dictionary object dict and produce sql for 
    inserting it into the named table"""
    sql = 'INSERT INTO ' + table
    sql += ' ('
    sql += ', '.join(dict)
    sql += ') VALUES ('
    sql += ', '.join(map(dictValuePad, dict))
    sql += ');'
    return sql

def save_response_pieces(prompt_response_id, prompt_name, response_list):
    try:
        # some prompt_responses were objects with {'response' : <list>}
        if "response" in response_list:
            print("FIX response for id %s" % prompt_response_id)
            response_list = response_list['response']

        for i in range(len(response_list)):
            # save new
            id = random.randint(1, 18446744073709551615)
            args = {}
            args['id'] = id
            args['prompt_response_id'] = prompt_response_id
            args['position'] = i
            args['text'] = response_list[i]
            if prompt_name == "test":
                args['text_type'] = "test_question"
            else:
                args['text_type'] = "book_text"

            sql = insertFromDict("response_piece", args)
            db.query(sql, args)
    except KeyError:  # includes simplejson.decoder.JSONDecodeError
        print("KeyError for pr %s, %s responses" % (prompt_response_id, len(response_list)))
        print(response_list)


def main(argv):
    prompt_responses = list(db.query("""SELECT r.id, r.response_text, p.name FROM prompt_response r, prompt p WHERE p.id = r.prompt_id"""))

    #print(book_ids)
    #print(prompt_ids)
    found_errors = 0
    for prompt_response in prompt_responses:
        try:
            response_list = json.loads(prompt_response['response_text'])
            save_response_pieces(prompt_response['id'], prompt_response['name'], response_list)
        except ValueError:  # includes simplejson.decoder.JSONDecodeError
            found_errors += 1

    print("%s errors out of %s, %s fixed" % (found_errors, len(prompt_responses), found_errors))

if __name__ == "__main__":
   main(sys.argv[1:])

"""
import json
import db
from localsettings import *

prompt_responses = list(db.query("""SELECT r.id, r.response_text, p.name FROM prompt_response r, prompt p WHERE p.id = r.prompt_id AND r.id = 7723103865787212168"""))
response_list = json.loads(prompt_responses[0]['response_text'])

"""