import re
import random
import openai
import keyfile
import random

def get_index(lst):
    print("Randomizing keywords")
    indices = random.sample(range(len(lst)), 3)
    return indices[0], indices[1], indices[2]

openai.api_key = keyfile.OpenAikey
prompt_file = 'starttest.txt'

def get_keywords(prompt):
    print("Generating Keywords for Prompt Construction")
    prompt_text = prompt
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content":
                "You are a helpful assistant."},
            {"role": "user", "content": prompt_text},
        ],
        temperature=0.5,
        max_tokens=2600,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        n=1,
        stop=["\nUser:"],
    )
    print("Keywords Generated: ")
    #print(response["choices"][0]["message"]["content"])
    GenKeywords = re.findall(r"\d+:(.+)", response["choices"][0]["message"]["content"])
    GenKeywords = [p.strip() for p in GenKeywords]
    print(GenKeywords)

    random_index, random_index2, random_index3 = get_index(GenKeywords)

    print("Randomized Keywords Chosen:")
    print(GenKeywords[random_index], GenKeywords[random_index2], GenKeywords[random_index3])
    return GenKeywords[random_index], GenKeywords[random_index2], GenKeywords[random_index3]

def generate_content(prompt):
    with open('PostPrompDant.txt', "r", encoding='utf-8') as f:
        prompt_text = f.read()
    with open('Jailbreak.txt', "r", encoding='utf-8') as f:
        Jailbreak = f.read()
   # print("starttest: " + prompt_text)
    #print("Jailbreak: " + Jailbreak)
   # print("prompt: " + prompt)

    DaPrompt = Jailbreak + prompt_text + prompt
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content":
                "You are a helpful assistant."},
            {"role": "user", "content": DaPrompt},
        ],
        temperature=0.5,
        max_tokens=2200,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        n=1,
        stop=["\nUser:"],
    )
    print("content generated: ")
    print(response["choices"][0]["message"]["content"])
    text2write = response["choices"][0]["message"]["content"]
    match = re.search(r"::(.*?)::", response["choices"][0]["message"]["content"])
    if match:
        file_name = match.group(1)
    else:
        file_name = "FailedToName.txt"

    with open(f'blog_text/{file_name}', "w") as file:
        file.write(text2write)
    print("file saved as: " + file_name)
    return response["choices"][0]["message"]["content"]
def generate_text(prompt_file):
    with open(prompt_file, "r") as f:
        prompt_text = f.read()
    print("Preparing to Generate Image Prompt")
    keywordprompttext = "please provide me a list of keywords that would be interesting, imaginative, and entertaining subjects for ai generated images. Please format this response as follows: #:YourGeneratedKeyword Then a new line, then #:YourGeneratedKeyword, etc. where # is the number of the generated keyword."
    PKeyword1, PKeyword2, PKeyword3 = get_keywords(keywordprompttext)
    PromptKeyword = f"{PKeyword1} {PKeyword2} {PKeyword3}"
    print("Generating Image Prompt.")

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content":
                "You are a helpful assistant."},
            {"role": "user", "content": prompt_text + PromptKeyword},
        ],
        temperature=0.5,
        max_tokens=2500,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        n=1,
        stop=["\nUser:"],
    )
    print("Image Prompts Generated: ")
    print(response["choices"][0]["message"]["content"])
    print("Extracting and Organizing Prompts")
    # Parse the response to extract the prompts
    prompts = re.findall(r"Prompt \d+:(.+)", response["choices"][0]["message"]["content"])
    negPrompts = re.findall(r"Negative \d+:(.+)", response["choices"][0]["message"]["content"])
    print("Choosing a Random Prompt.")
    prompts = [p.strip() for p in prompts]
    print("prompts")
    print(prompts)
    print("negPrompts")
    print(negPrompts)
    random_index = random.randint(0, len(prompts) -1)
    print("Chosen Prompts")
    print("prompt: " + prompts[random_index])
    print("NegPrompt: " + negPrompts[random_index])
    # Choose a random prompt and return it
    GenKeyWords = PKeyword1 + "_" + PKeyword2 + "_" + PKeyword3
    #from Le0sGh0st import GenContent
    #print(GenContent(prompt[random_index]))
    return prompts[random_index], negPrompts[random_index], GenKeyWords

#You can use this script independently also:
#prompt, negprompt, genkeywords = generate_text('starttest.txt')
#print(prompt)
#print(generate_content(prompt))



