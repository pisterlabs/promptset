from pathlib import Path 
import openai
import json 
import tiktoken
import pandas as pd 

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
        raise NotImplementedError(f"""num_tokens_from_messages() is not implemented for model {model}. 
        See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens



def get_prompt(model, txt_content):

    prompt = '''
    You are given a histopathological report. Please extract information in a structured manner.

    For each report fill the following categories by choosing one possible answer:

    Category:
    IDH1 codon 132 mutation

    Possible Answers:
    positive
    negative (wild type)
    not mentioned


    Category:
    IDH2 codon 172 mutation

    Possible Answers:
    positive
    negative (wild type)
    not mentioned


    Category:
    ATRX expression

    Possible Answers:
    loss
    mutation
    normal or retained
    not mentioned


    Category:
    Deletion of 1p and 19q

    Possible Answers:
    deleted
    not deleted
    not mentioned


    Category:
    highest Ki67 proliferation fraction in tissue

    Possible Answers:
    0-5%
    5-10%
    10-15%
    15-20%
    20-25%
    25-30%
    30-35%
    35-40%
    40-45%
    45-50%
    50-55%
    55-60%
    60-65%
    65-70%
    70-75%
    75-80%
    80-85%
    85-90%
    90-95%
    95-100%
    no number given, high
    no number giben, moderate
    no number given, low
    not mentioned


    Category:
    TP53 mutation or strong nuclear expression of p53 in >10% of tumor cells

    Possible Answers:
    yes
    no
    not mentioned


    Category:
    microvascular invasion

    Possible Answers:
    yes
    no
    not mentioned


    Category:
    necrosis

    Possible Answers:
    yes
    no
    not mentioned


    Category:
    EGFR gene amplification

    Possible Answers:
    yes
    no
    not mentioned


    Category:
    TERT promoter mutation

    Possible Answers:
    yes
    no
    not mentioned


    Category:
    Chromosome 7 gain/10 loss

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


path_database = Path.cwd()/'data/UCL_Glioma_flatdata_MASTER.xlsx'
df = pd.read_excel(Path.cwd()/'data/UCL_Glioma_flatdata_MASTER.xlsx', sheet_name='Flat Data')
df = df.drop_duplicates(subset='NH', keep=False)

# Create output folders 
path_out_txt = Path.cwd()/'results/UCL_full/text'
path_out_txt.mkdir(parents=True, exist_ok=True)
path_out_json = Path.cwd()/'results/UCL_full/json'
path_out_json.mkdir(parents=True, exist_ok=True)

model = 'gpt-4' # "gpt-3.5-turbo", gpt-4 , https://platform.openai.com/docs/models/gpt-4 
extracted_files1 = [path.stem for path in path_out_txt.iterdir()] # ChatGPT could extract report  
extracted_files2 = [path.stem for path in path_out_json.iterdir()] # report was in json format 
counter = len(extracted_files2) 

for n, row in df.iterrows():

    filename = row['NH']
    if filename in extracted_files1:
        continue

    print(f"Extracting -{n}: {filename}")
    
    # Analyze number of tokens 
    messages, tokens = get_prompt(model, row['Micro'])
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
