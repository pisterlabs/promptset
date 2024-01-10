from langchain.document_loaders import PyPDFLoader
from conf.constants import *

import sys
import re

import argparse

def remove_data_dir(collection_name): 
    dir = TEXT_DIR+collection_name
    if os.path.exists(dir):
        for the_file in os.listdir(dir):
            file_path = os.path.join(dir, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                else:
                    clear_folder(file_path)
                    os.rmdir(file_path)
            except Exception as e:
                print(e)
     
def create_data_dir(collection_name): 
    # Create a directory to store the text files
    if not os.path.exists(TEXT_DIR):
            os.mkdir(TEXT_DIR)

    if not os.path.exists(TEXT_DIR+collection_name+"/"):
            os.mkdir(TEXT_DIR + collection_name + "/")
    
            
# arguments
parser = argparse.ArgumentParser(description='Extract PDF pages')
parser.add_argument('-c', '--collection', help='The target collection name', required=True)
parser.add_argument('-s', '--start', help='Start page number', required=False, default=0)
parser.add_argument('-f', '--filename', help='The PDF file to parse', required=True)
args = parser.parse_args()

# recreate data dirs
remove_data_dir(args.collection)
create_data_dir(args.collection)

# load files and split into chunks (page wise)
# [hb] see if we can do chapter wise
print("Loading ", args.filename)
loader = PyPDFLoader(args.filename)
pages = loader.load_and_split()

print("Parsing PDF ...")
offset = int(args.start)
for page in pages[offset:]:
    page_num = page.metadata["page"]
    print("Parsing page ", str(page_num))
    try:        
        with open(TEXT_DIR+args.collection+'/page_'+str(page_num) + ".txt", "w", encoding="UTF-8") as f:                                
            f.write(page.page_content)
    except Exception as e:
        print("Unable to parse page " + page_num)
        print(e)