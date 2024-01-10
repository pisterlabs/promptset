import os
import re
import openai
import datetime
import random
import time
from contextlib import redirect_stdout
import sys
from ruamel.yaml import YAML

#### LOAD ENTITY CONFIG
yaml=YAML(typ='safe')
yaml.default_flow_style = False
configfile="../data/earth_land_landcover_nairobi.yaml"

with open(configfile, encoding='utf-8') as f:
   econfig = yaml.load(f)
#### END CONFIGn

oa = openai
oa.api_key = os.getenv("OPENAI_API_KEY")


maxlength = 300
elementmodel = econfig['entitydescr']['element']
gentype = econfig['entitydescr']['id']
gencount = 1
retrycount = 9
selectedmodel = "unknown"

if elementmodel == "earth":
   selectedmodel = "davinci:ft-personal-2022-05-08-13-37-54"
elif elementmodel == "water":
   selectedmodel =  "davinci:ft-personal:water-2022-03-31-23-56-04"
elif elementmodel == "fire":
   selectedmodel = "davinci:ft-personal:fire-2022-07-06-02-12-31"
elif elementmodel == "air":
   selectedmodel = "davinci:ft-personal:air-2022-07-05-23-19-23"
else:
   selectedmodel = "unknown"
   print("Selected model is unknown")


def trim_output(completion):
    try:
        if completion[-1] in ('.', '?', '!'):
            # print("matched end")
            trimmedoutput = completion
        else:
            try:
                # print("matched incomplete")
                re.findall(r'(\.|\?|\!)( [A-Z])', completion)
                indices = [(m.start(0), m.end(0)) for m in re.finditer(r'(\.|\?|\!)( [A-Z])', completion)]
                splittuple = indices[len(indices) - 1]
                trimmedoutput = completion[0:splittuple[0] + 1]
            except:
                trimmedoutput = completion
    except:
        trimmedoutput = completion

    return trimmedoutput


def get_act(myprompt, maxt, element):
    lengthext = random.randint(1, 56)
    maxt = maxt + lengthext
    print("Requesting generation with model: " + element)
    print("Requesting generation with maxlength: " + str(maxt))
    response = openai.Completion.create(
        model=element,
        prompt=myprompt,
        temperature=0.8,
        max_tokens=maxt,
        top_p=1,
        frequency_penalty=1,
        presence_penalty=1,
        stop=["ACT1","ACT2","ACT3","ACT4"]
    )
    story = response.choices[0].text

    # START TEST
    #story = "I am a boring story that is a placeholder for testing that uses the model: " + selectedmodel

    lstory = story.replace("\n", " ")
    lstory = lstory.replace("I'm a forest,", "I am")
    lstory = lstory.replace("I am a forest,", "I am")
    lstory = lstory.replace("I'm just a forest,", "I am")
    lstory = lstory.replace("I am just a forest,", "I am")
    lstory = lstory.replace("Forest: ","")

    return ' '.join(lstory.split())

def clean_response(rawresponse):
    try:
        response = openai.Edit.create(
          model="text-davinci-edit-001",
          input=rawresponse,
          instruction="Remove the garbled text. Correct the grammar in all sentences. Convert all text to sentence case. Make sure that each sentence is written in the first person tense. Make all pronouns gender neutral",
          temperature=0.8,
          top_p=1
        )
        cleanedresp = response.choices[0].text
    except:
        print("The edit operation failed for some reason.\r Returning raw response:")
        cleanedresp = rawresponse

    return cleanedresp

intro = econfig['prompt']['intro']
entitybio = econfig['entitydescr']['bio']
act0descr = econfig['prompt']['act0descr']
act1descr = econfig['prompt']['act1descr']
act2descr = econfig['prompt']['act2descr']
act3descr = econfig['prompt']['act3descr']

def writeout_act(actrawprompt, actprettyprompt,promptfile):
    for p in range(retrycount):
        prompt = actrawprompt
        print("\n\n<PRETTYPROMPT>")
        print(actprettyprompt)
        print("</PRETTYPROMPT>")
        print("<RAWPROMPT>" + prompt + "</RAWPROMPT>\n\n")
        actraw = get_act(prompt, maxlength, selectedmodel)
        time.sleep(1)
        # act = trim_output(actraw)
        actcl = clean_response(actraw)
        # print(actraw)
        actstatic = actcl + '\\n\\n'
        print("This is actraw (raw version): ")
        print("----------------------------------")
        print(actraw)
        print("----------------------------------")
        print("This is actstatic (cleaned version): ")
        print("----------------------------------")
        print(actstatic)
        print("----------------------------------")

        with open(promptfile, 'w', encoding='utf-8') as f:
            f.write(actraw)

        with open('prompt_buffer_temp.txt', 'w', encoding='utf-8') as f:
            f.write(actstatic)

        promptstatus = input(f"Is {promptfile} OK? y/n: ")
        if promptstatus == "y":
            promptstatus2 = input("Please press 'c' to read from an updated prompt file or 'yyy' to use as is: ")
            # Use a prompt file that has been updated.
            if promptstatus2 == "c":
                with open(promptfile, encoding='utf-8') as f:
                    actstatic = f.read()
            # Use the prompt as is:
            if promptstatus2 == "yyy":
                actstatic = actstatic
            else:
                # Use the file just in case I didnt press 'c' or 'yyy' properly
                with open(promptfile, encoding='utf-8') as f:
                    actstatic = f.read()
            break

    return actstatic

act1rawprompt = intro + act0descr + entitybio + '\\n\\n' + act1descr
act1prettyprompt = intro.replace('\\n','\n') + act0descr.replace('\\n','\n') +  entitybio + '\n\n' + act1descr.replace('\\n',' \n')
act1static = writeout_act(act1rawprompt, act1prettyprompt,"prompt_act1_temp.txt")

act2rawprompt = act1rawprompt + act1static + '\\n\\n' + act2descr
act2prettyprompt = act1prettyprompt + act1static.replace('\\n', '\n') + act2descr.replace('\\n', '\n')
act2static = writeout_act(act2rawprompt, act2prettyprompt,"prompt_act2_temp.txt")

act3rawprompt = act2rawprompt + act2static + '\\n\\n' + act3descr
act3prettyprompt = act2prettyprompt + act2static.replace('\\n', '\n') + act3descr.replace('\\n', '\n')
act3static = writeout_act(act3rawprompt, act3prettyprompt,"prompt_act3_temp.txt")

# datetime object containing current date and time
dt_string = datetime.datetime.now().strftime("%d-%m-%Y_%H_%M_%S")

story = '-----------' + '\n\n\nSample #' + dt_string + \
        ': ' +'\nACT 1: ' + \
        act1static.replace('\\n','\n') + '\nACT 2: ' + \
        act2static.replace('\\n','\n') + '\nACT 3: ' + \
        act3static.replace('\\n','\n')

print(story)

finalfile = '../generations/' + dt_string + '_' + gentype + '.txt'

###### WRITE GENERATIONS TO YAML
genid = dt_string
act1gen = act1static.replace('\\n','\n')
act2gen = act2static.replace('\\n','\n')
act3gen = act3static.replace('\\n','\n')
genpayload = {
    'aagen_id': genid,
    'act1gen': act1gen.strip(),
    'act2gen': act2gen.strip(),
    'act3gen': act3gen.strip()
              }
econfig['storygenerations'].append(genpayload)

#yaml.dump(econfig, sys.stdout)

with open(configfile, 'w', encoding='utf-8') as f:
    yaml.dump(econfig, f)

####### BACK UP GENERATIONS TO LOG

finalfile = '../generations/master_generation_log.txt'

try:
    with open(finalfile, 'a', encoding="utf-8") as f:
        with redirect_stdout(f):
            print(story)
except:
    print("File write error")