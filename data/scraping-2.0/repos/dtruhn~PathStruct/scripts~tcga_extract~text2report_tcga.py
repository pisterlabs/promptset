from pathlib import Path 
import openai
import json 
import tiktoken


# ------- GET API KEY -------
with open('secret_api_key.txt') as f:
    openai.api_key = f.read()


def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301"):
    # https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb 
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        # print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model == "gpt-3.5-turbo":
        # print("Warning: gpt-3.5-turbo may change over time. Returning num tokens assuming gpt-3.5-turbo-0301.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301")
    elif model == "gpt-4":
        # print("Warning: gpt-4 may change over time. Returning num tokens assuming gpt-4-0314.")
        return num_tokens_from_messages(messages, model="gpt-4-0314")
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif model == "gpt-4-0314":
        tokens_per_message = 3
        tokens_per_name = 1
    else:
        raise NotImplementedError(f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens



def get_prompt(model, txt_path):
    with open(txt_path, 'r') as file:
        txt_content = file.read()

    prompt = '''
    You are given a histopathological report. Please extract information in a structured manner.
    For each report fill the following categories by choosing one possible answer. If it says "number" you are allowed to answer with an integer or with "not mentioned":

    Category:
    Grade

    Possible Answers:
    G1
    G2
    G3
    G4
    not mentioned


    Category:
    T of TNM stage

    Possible Answers:
    T0
    Tis
    T1
    T2
    T3
    T4a
    T4b
    Tx (not known)
    not mentioned


    Category:
    N of TNM stage

    Possible Answers:
    Nx (cannot be determined)
    N0
    N1
    N1a
    N1b
    N2
    N2a
    N2b
    not mentioned


    Category:
    M of TNM stage

    Possible Answers:
    Mx (cannot be determined)
    M0
    M1a
    M1b
    M1c
    not mentioned


    Category:
    Number of Lymph Nodes examined

    Possible Answers:
    Number
    not mentioned


    Category:
    Number of Lymph Nodes that are positive

    Possible Answers:
    Number
    not mentioned


    Category:
    Lymphatic invasion

    Possible Answers:
    yes
    no
    not mentioned


    Category:
    Tumor free resection margin (R0)

    Possible Answers:
    yes
    no
    not mentioned


    Give the results as one json file.

    The report reads:
    '''+ txt_content

    
    messages=[{"role": "user", "content": prompt}]

    tokens = num_tokens_from_messages(messages, model)
    return messages, tokens



def generate_report(model, messages):
    response = openai.ChatCompletion.create( 
        model=model, #
        messages=messages,
        max_tokens=2048,  # The maximum number of tokens to generate in the completion.
        n=1, # How many completions to generate for each prompt.
        temperature=0, # Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic.
        top_p=1, # An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass. 
        stop=None, # Up to 4 sequences where the API will stop generating further tokens. 
        frequency_penalty=0, # Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics.
        presence_penalty=0 # Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics.
    )
    report = response.choices[0].message['content']
    return report

# Specify input data folder 
path_root = Path.cwd()/'data/reports_TCGA_txt/TCGA-COAD'

# Create output folders 
path_out_txt = Path.cwd()/'results/TCGA-COAD/text'
path_out_txt.mkdir(parents=True, exist_ok=True)
path_out_json = Path.cwd()/'results/TCGA-COAD/json'
path_out_json.mkdir(parents=True, exist_ok=True)

model = 'gpt-4' # "gpt-3.5-turbo", gpt-4 , https://platform.openai.com/docs/models/gpt-4 
extracted_files1 = [path.stem for path in path_out_txt.iterdir()] # ChatGPT could extract report  
extracted_files2 = [path.stem for path in path_out_json.iterdir()] # report was in json format 
counter = len(extracted_files2) 

for n, path_file in enumerate(path_root.glob('*.txt')):
    if counter>=100:
        break 
    filename = path_file.stem
    if filename in extracted_files1:
        continue


    print(f"Extracting -{n}: {filename}")
    
    # Analyze number of tokens 
    messages, tokens = get_prompt(model, path_file)
    if tokens >= 8192:
        print("Skip - too many tokens")
        continue
    
    # Try to run ChatGPT 
    print(f"Try to generate report {counter}")
    report = generate_report(model, messages)

    # Log raw ChatGPT output 
    with open(path_out_txt/f'{filename}.txt', 'w') as f:
        f.write(report)
    
    # Try to bring output into json-file format 
    try:
        report = json.loads(report)
    except Exception as e:
        print("Error in JSON format - skip")
        continue
    
    # Save json-file 
    with open(path_out_json/f'{filename}.json', 'w') as f:
        json.dump(report, f)

    counter += 1 
