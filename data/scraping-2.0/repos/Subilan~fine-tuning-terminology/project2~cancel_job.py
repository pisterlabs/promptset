from apikey import OPENAI_API_KEY
import openai
import argparse

openai.api_key = OPENAI_API_KEY

parser = argparse.ArgumentParser()
parser.add_argument('jobid', nargs='*')
args = parser.parse_args()

jobids = args.jobid

for i in jobids:
    print(openai.FineTuningJob.cancel(i))
