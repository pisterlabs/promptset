# In this program we open some chunk files and create indices from them. We then write the indices to S3.

import time
start_time = time.time()

import argparse
from dumbvector.dumb_index import create_dumb_index, get_dumb_index_file_writer
from dumbvector.docs import path_to_docs_list
from dumbvector.util import time_function
import openai
import json

end_time = time.time()
print (f'import time: {end_time - start_time}')

def read_credentials():
    with open('credentials.json', 'r') as f:
        return json.load(f)

def main():
    # usage: python create_index.py index_name index_path docs_path

    parser = argparse.ArgumentParser()

    parser.add_argument('index_name', help='the name of the index file to create')
    parser.add_argument('index_path', help='the path of the index file to create')
    parser.add_argument('docs_path', help='the path of the docs file to create')

    args = parser.parse_args()

    index_name = args.index_name
    index_path = args.index_path
    docs_path = args.docs_path

    # read the credentials
    credentials = read_credentials()

    openai.api_key = credentials['openaikey']

    # first read all the docs
    all_docs = time_function(path_to_docs_list)(docs_path)

    # create the index
    def get_vector(doc):
        return doc['embedding']

    dumb_index = time_function(create_dumb_index)(index_name, all_docs, get_vector)

    writer = get_dumb_index_file_writer(index_path)

    time_function(writer)(dumb_index)

    print ("done")

if __name__ == '__main__':
    main()

