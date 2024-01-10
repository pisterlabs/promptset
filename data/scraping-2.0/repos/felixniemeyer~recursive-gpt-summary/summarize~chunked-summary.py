import os 
import openai
import argparse

from pathlib import Path

CHARS_PER_TOKEN = 4.5

parser = argparse.ArgumentParser(
    prog='Chunked Summary',
    description='Summarizes a long text of a long text using chatGPT', 
)

parser.add_argument('-d', '--directory', type=str, help='Directory with input files', required=True)
parser.add_argument('-s', '--startswith', type=str, help='Read only files that start like this', required=True)
parser.add_argument('-t', '--target', type=str, help='Target directory for output files', required=True)

parser.add_argument('-pp', '--prepromt', type=str, help='Prompt text before each chunk', required=False, 
                    default='Fasse diesen Text in der Ich-Perspektive zusammen.')

parser.add_argument('-cp', '--contextpromt', type=str, help='Target directory for output files', required=False, 
                    default='Hier ist das Ende der vorherigen Zusammenfassung: <context> Schließe daran an.')

parser.add_argument('-ol', '--overlap', type=int, help='Number of chars to overlap', required=False, 
                    default=300)
parser.add_argument('-cl', '--context', type=int, help='Number of chars from previous summary as context', 
                    required=False, default=300)
parser.add_argument('-c', '--chunksize', type=int, help='Target chunk size in tokens', required=False, 
                    default=2500)
# factor has to be less than 1
parser.add_argument('-f', '--factor', type=int, help='Factor by which chatGPT should reduce by summary', required=False, 
                    default=0.33)

args = parser.parse_args()

target_char_count = args.chunksize * CHARS_PER_TOKEN * args.factor 
if(target_char_count < args.overlap):
    print("Overlap has to be smaller than target summary size")
    exit(1)

if(args.factor < 0.05 or args.factor > 0.75):
    print("Factor has to be between 0.05 and 0.5")
    exit(1)

if(args.target[-1] != '/'):
    args.target += '/'

condense_prompt = "Verwende etwa <wc> Wörter und schreibe in der Ich-Perspektive"

Path(args.target).mkdir(parents=True, exist_ok=True)

openai.api_key = "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"

def summarize(context, thoughts, outfile):
    print("Summarizing " + thoughts)

    messages = []

    messages.append({
        "role": "user", "content": thoughts
    }) 
    messages.append({
        "role": "system", "content": args.prepromt
    }) 

    if(len(context) > 0):
        messages.append({
            "role": "system", "content": args.contextpromt.replace("<context>", context)
        }) 

    target_word_count = len(thoughts) / 5 * args.factor
    target_word_count = int(target_word_count / 50) * 50

    messages.append({
        "role": "system", "content": condense_prompt.replace("<wc>", str(target_word_count))
    })

    print("Prompts: ", messages)

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
    )

    summary = response["choices"][0]["message"]["content"]

    print("Summary: " + summary)

    with open(args.target + outfile, "w") as f:
        f.write(summary)

    return summary

thoughts = ""
first_file = True
start_file = "" 
current_file = ""
context = ""
previous_outfile = None
outfile_counter = 0
for filename in sorted(os.listdir(args.directory)):
    print(filename)
    if filename.startswith(args.startswith):
        print("processing")
        if(first_file): 
            first_file = False
            start_file = filename
        current_file = filename
        with open(args.directory + filename, "r") as f:
            print("reading " + filename)
            thoughts += f.read()

            char_count = len(thoughts) + len(args.prepromt) + len(context) + len(condense_prompt) + args.overlap
            if(len(context) > 0):
                char_count += len(args.contextpromt)

            if(char_count > args.chunksize * CHARS_PER_TOKEN):
                excess_chars = char_count - args.chunksize * CHARS_PER_TOKEN
                chars = int(len(thoughts) - excess_chars)

                outfile = start_file + "_" + current_file 
                if(outfile == previous_outfile):
                    outfile_counter += 1
                    outfile += "_" + str(outfile_counter)
                else:
                    outfile_counter = 0
                    previous_outfile = outfile

                print("Summarizing " + str(chars) + " chars")
                summary = summarize(context, thoughts[:chars], outfile)

                context = summary[-args.context:]
                thoughts = thoughts[chars - args.overlap:]

                start_file = current_file

summarize(context, thoughts, start_file + "_" + current_file)
