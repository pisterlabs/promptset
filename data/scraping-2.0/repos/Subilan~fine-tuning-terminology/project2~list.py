import argparse
import openai
from apikey import OPENAI_API_KEY

openai.api_key = OPENAI_API_KEY

parser = argparse.ArgumentParser()
parser.add_argument('-l', '--limit')
args = parser.parse_args()

limit = int(args.limit) if args.limit != None else 1

print(openai.FineTuningJob.list(limit=limit))