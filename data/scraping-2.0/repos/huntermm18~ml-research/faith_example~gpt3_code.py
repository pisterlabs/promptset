import sys
import time
import openai
import pandas as pd
from tqdm import tqdm
from keys import *

openai.api_key = keys['openAI']
def do_query( prompt, max_tokens=512, engine=None, temperature=0.7 ):
    response = openai.Completion.create(
        engine=engine,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=1,
        logprobs=100,
    )
    return response

# ====================================================================================

PROMPT_START = """When interviewed about their experiences with racism, a person responded with:

"I have faced racism in the form of microaggressions and discrimination in the workplace, 
as well as subtle forms of bias in my personal life. Being a woman and a person of color, 
I have often felt like my accomplishments and qualifications have been overlooked or dismissed. 
Additionally, as a religious person, I have faced discrimination and prejudice based on my beliefs. 
It is important to acknowledge and address these forms of racism, and to work towards creating a 
more inclusive and equitable society."

Classify that story using the following tags:

Discrimination
Prejudice
Systemic racism
Interpersonal racism
Institutional racism

Tag: discrimination

When interviewed about their experiences with racism, a person responded with:

\""""

PROMPT_END = """\"

Classify that reason using the following tags:

Discrimination
Prejudice
Systemic racism
Interpersonal racism
Institutional racism

Tag:"""

df = pd.read_csv("./results_simple_multipass_text-davinci-003.csv")


ids = []
prompts = []
codes = []
rcnts = {}

for pid in tqdm( range(len(df)) ):

    person = df.iloc[pid]

    prompt = PROMPT_START
    tmp = person['response']
    tmp = tmp.replace("<1:[RECORD VERBATIM]>:","")

    if len(tmp) < 5:
        print( "SKIPPED..." )
        continue

    prompt += tmp
    prompt += PROMPT_END

    print("---------------------------------------------------")
    # print( prompt )

    done = False
    while not done:
        try:
            response = do_query( prompt, max_tokens=2, engine="text-davinci-003", temperature=0.1 )
            resp_text = response.choices[0]['text']
            done = True
        except:
            print("Error!  Sleeping for 10 seconds...")
            time.sleep(10.0)

    resp_text = resp_text.strip().lower()
    # print( resp_text )
    if not resp_text in rcnts:
        rcnts[resp_text] = 0
    rcnts[resp_text] += 1

    ids.append(pid)
    prompts.append(prompt)
    codes.append(resp_text)

newdf = pd.DataFrame({'ids':ids,'prompt':prompts,'code':codes})
newdf.to_csv("./results_code_simple_multipass_text-davinci-003.csv")

#print( rcnts )

total = 0
for k in rcnts.keys():
    total += rcnts[k]
for k in rcnts.keys():
    pct = rcnts[k]/total
    print( f"{k}: {pct:0.2f}" )
