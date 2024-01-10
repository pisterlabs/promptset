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
import re

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


def save_to_db(text, book_id, position):
    position = position / 2
    id = random.randint(1, 18446744073709551615)
    args = {}
    args['id'] = id
    args['book_id'] = book_id
    args['position'] = position
    args['text'] = text

    sql = insertFromDict("book_content", args)
    try:
      db.query(sql, args)
    except Exception as e:      
        print("Error in save_to_db")
        print(text)
        print(e)
   


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="book.txt")
    parser.add_argument("--id", default="-1")
    args = parser.parse_args()
    filepath = args.file
    id = args.id

    if id == "-1":
       print("need a book id")
       sys.exit(1)

    txt = Path(filepath).read_text()

    paragraphs = re.split(r"(\.|\!|\?|\n|\")\n", txt)

    for i in range(0, len(paragraphs), 2):
       #print("\n\nParagraph:")
       str = paragraphs[i]
       if i + 1 < len(paragraphs):
        str = str + paragraphs[i+1]
       
       #print(str)
       save_to_db(str, id, i)


if __name__ == "__main__":
   main(sys.argv[1:])
