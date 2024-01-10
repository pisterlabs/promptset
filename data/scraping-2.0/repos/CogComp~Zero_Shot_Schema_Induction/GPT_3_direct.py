import requests
import os
from os import listdir
from os.path import isfile, join
import json
import openai
openai.api_key = "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"
from Information_Extractor import event_extractor

dir_name = "/shared/kairos/Data/LDC2020E25_KAIROS_Schema_Learning_Corpus_Phase_1_Complex_Event_Annotation_V4/docs/ce_profile"
onlyfiles = [f for f in listdir(dir_name) if isfile(join(dir_name, f)) and f[-4:] == ".txt"]

scenarios = ['Bombing Attacks', 'Pandemic Outbreak', 'Civil Unrest', 'International Conflict', 'Disaster and Rescue', 'Terrorism Attacks', 'Election', 'Sports Games', 'Kidnapping', 'Business Change', 'Mass Shooting']
#scenarios = ['Cyber Attack']
for f in onlyfiles:
    #print(" ".join(f.split("_")[2:-1]))
    scenarios.append(" ".join(f.split("_")[2:-1]))

def call_openai_api(event, n, temperature, stop, presence_penalty):
    prompt = "Write essential steps for " + event + ":\n\n1."
    print(prompt)
    response = openai.Completion.create(
        #engine="davinci",
        engine="davinci-instruct-beta-v3",
        prompt=prompt,
        max_tokens=512,
        temperature=temperature,
        stop=stop,
        n=n,
        presence_penalty=presence_penalty
    )
    texts = []
    for choice in response["choices"]:
        texts.append(choice["text"])
    print("This api call ended!")
    return texts, response["id"]

for scenario in scenarios:
    f = open("output/" + scenario + "_direct.txt", 'w')
    scn = scenario.lower()
    res = call_openai_api(scn, 1, 0.9, None, 0.1)
    result = "1." + res[0][0]
    print("GPT-3 result:\n", file = f)
    print(result, file = f)
    
    headers = {'Content-type':'application/json'}
    SRL_response = requests.post('http://dickens.seas.upenn.edu:4039/annotate', json={"sentence": result}, headers=headers)

    if SRL_response.status_code != 200:
        print("SRL_response:", SRL_response.status_code)
    SRL_output = json.loads(SRL_response.text)
    
    print("\nevents:\n", file = f)
    response = event_extractor(result, 0, False)
    events = []
    for view in response['views']:
        if view['viewName'] == 'Event_extraction':
            for constituent in view['viewData'][0]['constituents']:
                print("'" + constituent['label'].lower() + "'", "appears in GPT-3 direct, mentions: {}, arguments:", end='', file = f)
                arguments = {}
                for arg in constituent['properties'].keys():
                    if 'ARG' in arg:
                        arguments[arg] = constituent['properties'][arg]['mention']
                print(arguments, file = f)
                events.append(constituent['label'].lower())

    print("\ntemporal relations:\n", file = f)
    num_events = len(events)
    for i_e in range(0, num_events-1):
        #for j_e in range(i_e+1, num_events):
        print("'('" + events[i_e] + ", '" + events[i_e + 1] + "')'", "appears in GPT-3 direct", file = f)
            
    f.close()
    
    