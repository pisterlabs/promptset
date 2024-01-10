from pathlib import Path 
import openai
import json 


# ------- GET API KEY -------
with open('secret_api_key.txt') as f:
    openai.api_key = f.read()


def generate_report(txt_path):
    with open(txt_path, 'r') as file:
        txt_content = file.read()

    prompt = '''
        Below you will be provided a histopathological report following "Original Report". 
        The text has been extracted from a pdf file, so there might be errors. 
        Please bring the report into the following json format:
        {
        "Specimens": [{
            "ID":"",
            "Source Site": "",
            "Clinical features": "",
            "Surgery": "",
            "Tissue Examined": "",
            "Size Tissue examined": "",
            "Macroscopy": "",
            "Histology": "",
            "Differentiation": "",
            "Grade": "",
            "Tumor Size": "",
            "Invasion/Infiltration": "",
            "Resection Margins": "",
            "AJCC T": "",
            "AJCC N": "",
            "AJCC M": "",
            "Lymph nodes examined": "",
            "Lymph nodes positive": "",
            "Lymphatic Invasion/L": "",
            "Vascular Invasion/V": "",
            "Perineural Invasion/P": "",
            "Budding": "",
            "TILS": "",
            "IHC": "",
            "MSI": "",
            "others pathological findings": ""
        }]
        }
        "Original Report":
    '''+ txt_content

    
    

    response = openai.ChatCompletion.create( 
        model="gpt-4", # "gpt-3.5-turbo", gpt-4 , https://platform.openai.com/docs/models/gpt-4 
        messages=[{"role": "user", "content": prompt}],
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
path_root = Path.cwd()/'data'/'reports_txt'

# Create output folders 
path_out_txt = Path.cwd()/'results'/'reports_text'
path_out_txt.mkdir(parents=True, exist_ok=True)
path_out_json = Path.cwd()/'results'/'reports_json'
path_out_json.mkdir(parents=True, exist_ok=True)

for path_file in path_root.glob('*.txt'):
    filename = path_file.stem

    if filename in [path.stem for path in path_out_txt.iterdir()]:
        continue
    print(f"Extracting: {filename}")
    report = generate_report(path_file)


    with open(path_out_txt/f'{filename}.txt', 'w') as f:
        f.write(report)
    
    # Write report as json-file
    report = json.loads(report)
    with open(path_out_json/f'{filename}.json', 'w') as f:
        json.dump(report, f)

