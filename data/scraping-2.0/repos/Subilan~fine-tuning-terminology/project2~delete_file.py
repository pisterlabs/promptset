from apikey import OPENAI_API_KEY
import openai
import argparse

openai.api_key = OPENAI_API_KEY

parser = argparse.ArgumentParser()
parser.add_argument('fileids', nargs='*')
args =parser.parse_args()

fileids = args.fileids

for i in fileids:
    openai.File.delete(i)
    print(f'deleted: {i:s}')