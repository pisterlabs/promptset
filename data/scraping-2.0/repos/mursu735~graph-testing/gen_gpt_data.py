import os
import re
import time
import openai
import tiktoken
import helpers


def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

def determine(file):
    filename = file.split(".")
    filename = filename[0]
    return os.path.isfile(f"output/GPT/locations/{filename}.csv")

key = ""
with open("key.txt") as file:
    key = file.read()
print(key)

openai.api_key = key

# list models
#models = openai.Model.list()

# print the first model's id
#print(models.data)

instruction = helpers.get_instruction()
print(instruction)

if not os.path.exists("output/GPT"):
    os.mkdir("output/GPT")
    
#if not os.path.isfile("output/GPT/locations.csv"):
#   with open("output/GPT/locations.csv", "w", encoding="utf-8") as file:
#       file.write("Location;Latitude;Longitude;Order\n")

if not os.path.exists("output/GPT/locations"):
    os.mkdir("output/GPT/locations")

if not os.path.exists("output/GPT/transport"):
    os.mkdir("output/GPT/transport")

if not os.path.exists("output/GPT/summary"):
    os.mkdir("output/GPT/summary")

prompt = ""
summary = ""
files = os.listdir("input/Chapters")
files = natural_sort(files)
#print("Before:")
#print(files)
# Only include chapters that haven't been parsed yet to reduce load and cost
files = [file for file in files if not determine(file)]
print(f"Processing files {files}")
#print("After:")
#print(files)
# Check if summaries already exist and get the last one
if len(os.listdir("output/GPT/summary")) > 0:
    summaries = natural_sort(os.listdir("output/GPT/summary"))
    print(f"Summaries written {summaries}")
    print(f"last summary {summaries[-1]}")
    with open(f"output/GPT/summary/{summaries[-1]}") as file:
        summary = file.read()

#print(f"Summary: {summary}")

last_request = time.time()

for file in files:
    filename = int(file.split(".")[0])
    print(f"Processing chapter {filename}")
    #with open(f"input/Chapters/{file}", encoding='utf-8') as file:
    prompt = helpers.get_chapter(filename)

    # Check for number of tokens sent in the last minute
    whole_prompt = instruction + "\n" + prompt
    encoding = tiktoken.encoding_for_model("gpt-4")
    num_tokens = len(encoding.encode(whole_prompt))
    time_to_wait = last_request - time.time() + 60
    # Don't wait for the first response
    if time_to_wait < 0:
        print(f" {time_to_wait} seconds passed since last request, no need to wait, sending request now")
    else:
        if time_to_wait < 59.5:
            print(f"Send only one request per minute, wait for {time_to_wait} seconds")
            time.sleep(time_to_wait)

    print("Send request to GPT")
    # create a chat completion
    last_request = time.time()
    chat_completion = openai.ChatCompletion.create(model="gpt-4", messages=[{"role": "system", "content": instruction}, {"role": "user", "content": prompt}], temperature=0)
    #chat_completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "system", "content": instruction}, {"role": "user", "content": prompt}])
    # print the chat completion
    message = chat_completion.choices[0].message.content
    print(message)
    prompt_tokens = chat_completion.usage["prompt_tokens"]
    completion_tokens = chat_completion.usage["completion_tokens"]
    total_tokens = chat_completion.usage["total_tokens"]
    if not os.path.isfile(f"output/GPT/tokens/tokens.csv"):
        with open("output/GPT/tokens/tokens.csv", "w", encoding='utf-8') as file:
            file.write("Chapter;Prompt;Completion;Total\n")

    with open("output/GPT/tokens/tokens.csv", "a", encoding="utf-8") as file:
        file.write(f"{filename};{prompt_tokens};{completion_tokens};{total_tokens}\n")
    #print(message)

    parts = message.split("////")
    # Put all main locations to single file
    #locations = parts[0]
    #with open("output/GPT/locations.csv", "a", encoding="utf-8") as file:
    #    file.write(locations)
    # Put locations of people to different files
    people = "Person;City;Latitude;Longitude;Location;Order;Importance\n"
    people += parts[0]
    with open(f"output/GPT/locations/{filename}.csv", "a", encoding="utf-8") as file:
        file.write(people)
    transport = "Person;Mode;Order\n"
    transport += parts[1]
    with open(f"output/GPT/transport/{filename}.csv", "a", encoding="utf-8") as file:
        file.write(transport)
    # Put summaries to different files
    #summary = parts[3]
    #if not os.path.isfile("output/GPT/summary/{filename}.txt"):
        #with open(f"output/GPT/summary/{filename}.txt", "a", encoding="utf-8") as file:
            #file.write(summary)
    
#print(prompt)

'''
print(files)

for file in files:
    with open(f"input/texts/{file}", encoding='utf-8') as f:
        prompt = prompt + ''.join(line for line in f)

print(prompt)
with open("testing.txt", "w", encoding='utf-8') as file:
    file.write(prompt)
'''


#with open("output/output_test.csv", "w") as file:
#   file.write(chat_completion.choices[0].message.content)
