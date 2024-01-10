import argparse
from pathlib import Path
from dotenv import load_dotenv
import time
import random
import os
import openai
import json
parser = argparse.ArgumentParser()
parser.add_argument("numberofpromnts")
parser.add_argument("path")
args = parser.parse_args()

openai.api_key = os.getenv("OPENAI_API_KEY")

target_dir = Path(args.path)
numberofpromnts=int(args.numberofpromnts)


load_dotenv()
openai.api_key = os.getenv("OPENAI")
outputdir=os.getenv("OUTDIR")



def generate_promnt(input_file,output_dir,i):
    text_file = open(input_file, "r")

    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=text_file.read(),
        temperature=0.7,
        max_tokens=500,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    text_file.close
    destination_file = open(output_dir+'/'+os.path.basename(input_file)+'_call_'+str(i)+".api", 'w')
    print(response, file = destination_file)
    destination_file.close()
    time.sleep(random.randint(20,40))

def print_json(input_file, output_dir, i):
    json_file_path=os.path.join(input_file)
    with open(json_file_path) as json_file:
        json_data = json.load(json_file)
        data_to_insert=json_data['choices'][0]['text']
        destinationFile = open(output_dir+'/'+os.path.basename(input_file).replace(".api",".md"), 'w')
        print(''.join(data_to_insert), file = destinationFile)
        destinationFile.close()

if not os.path.exists(outputdir):
   # Create a new directory because it does not exist
   os.makedirs(outputdir)

if not target_dir.exists():
    print("The target directory doesn't exist")
    raise SystemExit(1)

for entry in target_dir.iterdir():
    if entry.is_file():
        for i in range(numberofpromnts):
            #generate_promnt(entry, outputdir, i)
            print_json(entry,outputdir,i)
