""",
Generate the prompts for individual examples.
"""
import argparse, os
import pandas as pd
import json
import getpass
import openai
from tqdm import tqdm

def readArgs():
    parser = argparse.ArgumentParser(description='Generate the prompts for the GPT models')
    parser.add_argument ('--template-file', type=str, required=True, help="File contains the template for the prompt")
    parser.add_argument ('--input-file', type=str, required=True, help="Path to the input TSV file")
    parser.add_argument ('--field-name', type=str, required=False, default='context_100', help="Text field")
    parser.add_argument ('--temperature', type=float, required=False, default=0.2, help="Temperature parameter")
    parser.add_argument ('--model-name', type=str, required=False, default='gpt-3.5-turbo', choices={'gpt-4', 'gpt-3.5-turbo'}, help="Model name")
    parser.add_argument ('--output-dir', type=str, required=True, help="Path to the output director that contains the output")
    args = parser.parse_args()
    return args

def readTemplateFile (filename):
    with open (filename) as fin:
        text = fin.read()
    
    return text

def generate_prompt (text, template):
    return template.replace ('{{TEXT}}', text)

def make_openai_call (prompt, model_name, temperature=0.2):
    response = openai.ChatCompletion.create(
            model=model_name,
            messages = [
                {"role": "user", "content": prompt}
            ],
            temperature=temperature
        )
    return response

def main (args):
    os.makedirs(args.output_dir, exist_ok=True)
    APIKEY = getpass.getpass(prompt='Enter OpenAI API key:')
    openai.api_key = APIKEY

    template = readTemplateFile (args.template_file)

    with open(args.input_file) as file:
        file.readline()
        for line in file:
            cols=line.rstrip().split("\t")
            if len(cols) != 7:
                print(cols)
            idd, book_id, char_text, place_text, context_10, context_50, context_100=cols


            field=None
            if args.field_name == "context_100":
                field=context_100
            elif args.field_name == "context_50":
                field=context_50
            elif args.field_name == "context_10":
                field=context_10

            outfile="%s/%s.txt" % (args.output_dir, idd)
            if not os.path.exists(outfile):
                prompt=generate_prompt(field, template)
                response=make_openai_call(prompt, args.model_name, args.temperature)
                with open(outfile, "w") as out:
                    out.write("%s\t%s\t%s\t%s\t%s" % (idd, args.template_file, args.field_name, field, json.dumps(response)))
            else:
                print("%s exists, skipping..." % outfile)

    # tqdm.pandas(desc='Progressbar')
    
    # df = pd.read_csv (args.input_file, sep='\t')

    # df['prompt'] = df.progress_apply (lambda x: generate_prompt (x[args.field_name], template), axis=1)
    # df['response'] = df.progress_apply (lambda x: make_openai_call (x['prompt'], args.model_name, args.temperature), axis=1)

    # df.to_csv (args.output_file, sep='\t', index=False, header=True)

if __name__ == '__main__':
    main (readArgs())