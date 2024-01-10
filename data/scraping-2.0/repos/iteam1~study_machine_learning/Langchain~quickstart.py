'''
python3 Langchain/quickstart.py
'''
import langchain
import openai
import os

print('Import successfuly!')

SERCET_KEY_PATH = "/home/gom/Documents/secret_key.txt"

with open(SERCET_KEY_PATH,'r') as f:
    sercet_key = f.read()

os.environ['OPENAI_API_KEY'] = sercet_key
print(os.environ.get('OPENAI_API_KEY'))