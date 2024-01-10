#!/usr/bin/env python

import requests
import sys
import time
import db
import openai
import hashlib
import random

OPENAI_API_KEY = ''

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


def fix_position(book_id, prompt_id):
    prompt_responses = list(db.query("""SELECT id, position FROM prompt_response WHERE book_id = %s AND prompt_id = %s ORDER BY position""", [book_id, prompt_id]))

    for i in range(len(prompt_responses)):
        pr_id = prompt_responses[i]['id']
        db.query("""UPDATE prompt_response SET position = %s WHERE id = %s""", [i, pr_id])

        response_pieces = list(db.query("""SELECT id, position FROM response_piece WHERE prompt_response_id = %s ORDER BY position""", [pr_id]))
        for j in range(len(response_pieces)):
            rp_id = response_pieces[j]['id']
            db.query("""UPDATE response_piece SET position = %s WHERE id = %s""", [j, rp_id])


def main(argv):
    book_ids = list(db.query("""SELECT id FROM book"""))
    prompt_ids = list(db.query("""SELECT id FROM prompt"""))

    #print(book_ids)
    #print(prompt_ids)
    for book_id in book_ids:
        for prompt_id in prompt_ids:
            fix_position(book_id['id'], prompt_id['id'])

if __name__ == "__main__":
   main(sys.argv[1:])
