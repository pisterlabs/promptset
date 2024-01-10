from openai import OpenAI
import os
import json
from extraction import Extraction
from omission import Omission
from evidence import Evidence
from prune import Prune
from gpt import n_shot_verification

with open('key.json', 'r') as f:
    key = json.load(f)['OPENAI_API_KEY']

model = 'gpt-4-1106-preview'
client = OpenAI(api_key=key)


snippit = 'On _%#MMDD2007#%_ if ok with Dr. _%#NAME#%_, increase Lovenox to 100 mg subcu b.i.d. Discontinue Lovenox 24 hours prior to next surgery on _%#MMDD2007#%_. Will hold aspirin and Coumadin until after next scheduled surgery on _%#MMDD2007#%_. 3. Hypertension, controlled. Hold Lopressor, Lasix, spironolactone for SBP less than 110. Hold metoprolol for pulse less than 55. 4. Diabetes type 2, currently controlled. Hold glyburide for blood sugar less than 100. Change IV fluid to normal saline plus 20 KCl at 125 ml per hour.'

out, prompt_tokens, completion_tokens = n_shot_verification(client, snippit, n_shots=1)

print(out)

input_fee = 0.01  # per 1k tokens
output_fee = 0.03 # per 1k tokens

input_cost = (prompt_tokens / 1000) * input_fee
output_cost = (completion_tokens / 1000) * output_fee
total_cost = (input_cost + output_cost)

print(f'Input tokens used: {prompt_tokens}')
print(f'Output tokens used: {completion_tokens}')
print(f'Total fee: ${total_cost}')

with open('test.txt', 'w') as f:
    f.write(out)

