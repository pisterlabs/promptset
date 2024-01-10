import os 
import openai
import argparse
import json

from pathlib import Path

# support .env file
from dotenv import load_dotenv
load_dotenv()

# open ai key
openai.api_key = os.getenv("OPENAI_API_KEY")

CHARS_PER_TOKEN = 4.5

parser = argparse.ArgumentParser(
    prog='Chunked Summary',
    description='Summarizes a long text of a long text using chatGPT', 
)

parser.add_argument('-d', '--directory', type=str, help='Directory with input files', required=True)

parser.add_argument('-pp', '--prepromt', type=str, help='Prompt text before each chunk', required=False, 
                    default='Fasse diesen Text in der Ich-Perspektive zusammen.')

parser.add_argument('-cp', '--contextpromt', type=str, help='Target directory for output files', required=False, 
                    default='Hier ist das Ende der vorherigen Zusammenfassung: <context> Schließe daran an.')

parser.add_argument('-ol', '--overlap', type=int, help='Number of chars to overlap', required=False, 
                    default=300)
parser.add_argument('-cl', '--context', type=int, help='Number of chars from previous summary as context', 
                    required=False, default=300)
parser.add_argument('-c', '--chunksize', type=int, help='Target chunk size in tokens', required=False, 
                    default=2400)
# factor has to be less than 1
parser.add_argument('-f', '--factor', type=int, help='Factor by which chatGPT should reduce by summary', required=False, 
                    default=0.3)

parser.add_argument('-fk', '--fake-summary', action='store_true', help='Use fake summarization instead of chatGPT', 
                    default=False)

args = parser.parse_args()

target_char_count = args.chunksize * CHARS_PER_TOKEN * args.factor 
if(target_char_count < args.overlap):
    print("Overlap has to be smaller than target summary size")
    exit(1)

if(args.factor < 0.05 or args.factor > 0.75):
    print("Factor has to be between 0.05 and 0.5")
    exit(1)

# remove trailing slash
if(args.directory.endswith("/")):
    args.directory = args.directory[:-1]

# check that subdirectory 'original' exists
original_dir = args.directory + "/original"
if not os.path.exists(original_dir):
    print("Directory " + original_dir + " does not exist")
    exit(1)

# make dir maps 
Path(args.directory + '/maps').mkdir(parents=True, exist_ok=True)

condense_prompt = "Verwende etwa <wc> Wörter und schreibe in der Ich-Perspektive"

def summarize(context, thoughts):
    if(args.fake_summary):
        return fakeSummarize(context, thoughts)
    else:
        return chatGPTSummarize(context, thoughts)

def chatGPTSummarize(context, thoughts):
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

    print('request: ' + str(messages))

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
    )

    print('response: ' + str(response))

    summary = response["choices"][0]["message"]["content"]

    return summary

def fakeSummarize(context, thoughts):
    counter = 0
    takeEach = 1 / args.factor
    summary = ""
    for c in thoughts:  
        counter += 1
        if(counter > takeEach):
            counter -= takeEach
            summary += c
    return summary


def processFolder(fromFolderName, level, origins, summaries, previous_char_map):
    print('processing folder ' + fromFolderName)

    file_list = []
    char_map = []

    fromFolder = args.directory + '/' + fromFolderName
    toFolder = args.directory + '/' + str(level)
    Path(toFolder).mkdir(parents=True, exist_ok=True)
    thoughts = ""
    current_file = ""
    # new set 
    origin_files = []
    context = ""
    file_counter = 0

    def loadExistingSummary(): 
        nonlocal toFolder
        nonlocal file_counter

        # pad file_counter with zeros
        new_file_name = str(file_counter).zfill(5)

        already_existing_summary = None

        try: 
            path = toFolder + '/' + new_file_name
            with open(path, "r") as f:
                already_existing_summary = f.read()
            print('found ' + path + '. Skipping')
        except:
            pass

        return already_existing_summary



    def createFile(summary): 
        nonlocal file_counter
        nonlocal origin_files
        nonlocal origins
        nonlocal summaries

        # pad file_counter with zeros
        new_file_name = str(file_counter).zfill(5)

        with open(toFolder + '/' + new_file_name, "w") as f:
            f.write(summary)

        file_id = str(level) + '/' + str(file_counter).zfill(5)
        origins[file_id] = origin_files
        for origin in origin_files:
            if origin not in summaries:
                summaries[origin] = []
            summaries[origin].append(file_id)

        file_counter += 1

        return summary

    file_list = sorted(os.listdir(fromFolder))

    most_recent_file_start = 0
    most_recent_file_end = 0

    build_initial_char_map = False
    if len(previous_char_map) == 0:
        build_initial_char_map = True

    sanity_check = 0

    file_index = 0
    for filename in file_list:
        current_file = fromFolderName + '/' + filename
        origin_files.append(current_file)
        with open(fromFolder + '/' + filename, "r") as f:
            most_recent_thoughts = f.read()
            if(most_recent_thoughts == ''): 
                continue
            most_recent_file_length = len(most_recent_thoughts)
            sanity_check += most_recent_file_length
            if build_initial_char_map: 
                most_recent_file_start = most_recent_file_end
                most_recent_file_end += most_recent_file_length
                previous_char_map.append(most_recent_file_end) 
                print(file_index)
            else: 
                most_recent_file_start = most_recent_file_end
                print(file_index)
                if(file_index == 330): 
                    print(previous_char_map)
                    print(len(previous_char_map))
                most_recent_file_end = previous_char_map[file_index]
                
            previous_thoughts_length = len(thoughts)
            thoughts += most_recent_thoughts

            char_count = len(thoughts) + len(args.prepromt) + len(context) + len(condense_prompt) + args.overlap
            if(len(context) > 0):
                char_count += len(args.contextpromt)

            if(char_count > args.chunksize * CHARS_PER_TOKEN):
                excess_chars = char_count - args.chunksize * CHARS_PER_TOKEN
                chars = int(len(thoughts) - excess_chars)

                chunk = thoughts[:chars]
                print('chunk: ' + chunk)

                summary = loadExistingSummary()
                if summary == None: 
                    summary = summarize(context, chunk)
                createFile(summary)

                # create new set of origin files 
                origin_files = [current_file]

                context = summary[-args.context:]
                thoughts = thoughts[chars - args.overlap:]

                progress_in_most_recent_file = (chars - previous_thoughts_length) / most_recent_file_length
                most_recent_file_width = most_recent_file_end - most_recent_file_start
                char_map.append(
                    int(progress_in_most_recent_file * most_recent_file_width + most_recent_file_start)
                )
                    
        file_index += 1

    summary = loadExistingSummary()
    if summary == None: 
        summary = summarize(context, thoughts)
    char_map.append(most_recent_file_end)
    createFile(summary)

    return file_counter, file_list, char_map

def saveCharMap(char_map, level): 
    with open(args.directory + '/maps/' + level + '.json', 'w') as f:
        json.dump(char_map, f)

origins = {}
summaries = {} # a file may have 2 files as a summary, when it's contents are split

level = 1
original_char_map = []
file_count, file_list, char_map  = processFolder('original', level, origins, summaries, original_char_map)
saveCharMap(original_char_map, 'original')
saveCharMap(char_map, str(level))

while file_count > 1: 
    file_count, file_list, char_map = processFolder(str(level), level + 1, origins, summaries, char_map)
    level += 1
    saveCharMap(char_map, str(level))

# write origins and summaries into json files
with open(args.directory + '/origins.json', 'w') as f:
    json.dump(origins, f)

with open(args.directory + '/summaries.json', 'w') as f:
    json.dump(summaries, f)

# write number of levels into file
with open(args.directory + '/levels', 'w') as f:
    f.write(str(level))

