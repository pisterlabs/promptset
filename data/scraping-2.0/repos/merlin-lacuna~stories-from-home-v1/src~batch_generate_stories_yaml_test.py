import os
import re
import openai
import datetime
from contextlib import redirect_stdout
import sys
from ruamel.yaml import YAML

#### LOAD ENTITY CONFIG
yaml=YAML(typ='safe')
yaml.default_flow_style = False
configfile="../data/water_land_ndwi_hongkong.yaml"


with open(configfile, encoding='utf-8') as f:
   econfig = yaml.load(f)
#### END CONFIG


oa = openai
oa.api_key = os.getenv("OPENAI_API_KEY")

maxlength = 256
elementmodel = econfig['entitydescr']['element']
gentype = econfig['entitydescr']['id']
gencount = 1
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
    # response = openai.Completion.create(
    #     model=element,
    #     prompt=myprompt,
    #     temperature=1,
    #     max_tokens=maxt,
    #     top_p=1,
    #     frequency_penalty=1,
    #     presence_penalty=1,
    #     stop=["Act "]
    # )
    # story = response.choices[0].text
    #
    # START TEST
    story = "I am a boring story that is a placeholder for testing that uses the model: " + selectedmodel

    lstory = story.replace("\n", " ")
    lstory = lstory.replace("I'm a forest,", "I am")
    lstory = lstory.replace("I am a forest,", "I am")
    lstory = lstory.replace("I'm just a forest,", "I am")
    lstory = lstory.replace("I am just a forest,", "I am")
    lstory = lstory.replace("Forest: ","")

    return ' '.join(lstory.split())



intro = econfig['prompt']['intro']

entitybio = econfig['entitydescr']['bio']

act0descr = econfig['prompt']['act0descr']

act1descr = econfig['prompt']['act1descr']

act2descr = econfig['prompt']['act2descr']

act3descr = econfig['prompt']['act3descr']

for x in range(gencount):
    # GET PROMPT FOR ACT1
    act1rawprompt = intro + act0descr + entitybio + '\\n\\n' + act1descr
    act1prettyprompt = intro.replace('\\n','\n') + act0descr.replace('\\n','\n') +  entitybio + '\n\n' + act1descr.replace('\\n',' \n')
    prompt = act1rawprompt
    print("\n\n<PRETTYPROMPT>")
    print(act1prettyprompt)
    print("</PRETTYPROMPT>")
    print("<RAWPROMPT>")
    print(prompt)
    print("</RAWPROMPT>\n\n")
    act1raw = get_act(prompt, maxlength, selectedmodel)
    act1 = trim_output(act1raw)
    # print(act1)
    act1static = act1 + '\\n\\n'

    # GET PROMPT FOR ACT2#
    act2rawprompt = act1rawprompt +  act1static  + '\\n\\n' + act2descr
    act2prettyprompt = act1prettyprompt + act1static.replace('\\n','\n') + act2descr.replace('\\n','\n')
    prompt = act2rawprompt
    print("\n\n<PRETTYPROMPT>")
    print(act2prettyprompt)
    print("</PRETTYPROMPT>")
    print("<RAWPROMPT>")
    print(prompt)
    print("</RAWPROMPT>\n\n")
    act2raw = get_act(prompt, maxlength, selectedmodel)
    # print(act2raw)
    act2 = trim_output(act2raw)
    # print(act2)
    act2static = act2 + '\\n\\n'

    # GET PROMPT FOR ACT3
    act3rawprompt= act2rawprompt + act2static  + '\\n\\n' + act3descr
    act3prettyprompt = act2prettyprompt + act2static.replace('\\n','\n') + act3descr.replace('\\n','\n')
    prompt = act3rawprompt
    act3raw = get_act(prompt, maxlength, selectedmodel)
    act3 = trim_output(act3raw)
    print("\n\n<PRETTYPROMPT>")
    print(act3prettyprompt + act3)
    print("</PRETTYPROMPT>")
    print("<RAWPROMPT>")
    print(prompt+act3)
    print("</RAWPROMPT>\n\n")


    story = '\nACT 1: ' + act1static.replace('\\n','\n') + 'ACT 2: ' + act2static.replace('\\n','\n') + 'ACT 3: ' + act3.replace('\\n','\n')

    # datetime object containing current date and time
    dt_string = datetime.datetime.now().strftime("%d-%m-%Y_%H_%M_%S")

    print('-----------')
    print('\n\n\nSample #' + dt_string + ":")
    print(story)

    ###### WRITE GENERATIONS TO YAML
    genid = dt_string
    act1gen = act1static.replace('\\n','\n')
    act2gen = act1static.replace('\\n','\n')
    act3gen = act3.replace('\\n','\n')
    genpayload = {
        'gen_id': genid,
        'act1gen': act1gen.strip(),
        'act2gen': act2gen.strip(),
        'act3gen': act3gen.strip()
                  }
    econfig['storygenerations'].append(genpayload)

    #yaml.dump(econfig, sys.stdout)

    with open(configfile, 'w', encoding='utf-8') as f:
        yaml.dump(econfig, f)