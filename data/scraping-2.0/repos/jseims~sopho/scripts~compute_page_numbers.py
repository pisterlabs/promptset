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


MIN_CHARS_PER_PAGE = 2000

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



def compute_page_numbers(book_id):
    book_content_list = list(db.query("""SELECT id, length(text) FROM book_content WHERE book_id = %s ORDER BY position ASC""", [book_id]))
    
    page = 1
    char_count = 0

    for content in book_content_list:
        id = content['id']
        length = content['length(text)']

        db.query("""UPDATE book_content SET page = %s WHERE id = %s""", [page, id])

        char_count = char_count + length
        if char_count > MIN_CHARS_PER_PAGE:
           char_count = 0
           page = page + 1

def main(argv):
    book_ids = list(db.query("""SELECT id FROM book"""))

    #print(book_ids)
    for book_id in book_ids:
        id = book_id['id']
        compute_page_numbers(id)

if __name__ == "__main__":
   main(sys.argv[1:])

"""
import db
from localsettings import *

id = 6127970443347761332
pr_id = 12026640671696316930

query = "SELECT * FROM response_piece WHERE prompt_response_id = %s ORDER BY position"
rp_list = list(db.query(query, [pr_id]))


query = "SELECT rp.matches, pr.book_id, b.title FROM response_piece rp, prompt_response pr, book b WHERE rp.id = %s AND rp.prompt_response_id = pr.id AND pr.book_id = b.id"
rp_list = list(db.query(query, [id]))

"""