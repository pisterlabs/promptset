import os
import csv
import yaml
import openai


def read_config():
    with open("openai_credentials.yaml", "r") as yamlfile:
        data = yaml.load(yamlfile, Loader=yaml.FullLoader)
        return data[0]['key']

def write_config():
    api_info = [
        {
            'key': 'OPENAI_API_KEY'
        }
    ]

    with open("openai_credentials.yaml", 'w') as yamlfile:
        data = yaml.dump(api_info, yamlfile)

def save_output(code_str, file_name):
    output_dir = "output"
    if not file_name.endswith('.py'):
        file_name += '.py'
    file_path = os.path.join(output_dir, file_name)
    with open(file_path, "w") as outfile:
        outfile.write(code_str)


# Read the specified prompt file (.csv)
# Column 0: Base prompt w/ no COT
# Column 1: COT-enhanced prompt
def read_prompts(fname):
    base_prompt_dir = "prompts"
    if not fname.endswith('.csv'):
        fname += '.csv'
    prompt_path = os.path.join(base_prompt_dir, fname)
    with open(prompt_path) as csvfile:
        reader = csv.reader(csvfile)
        header = True
        for row in reader:
            # Skip over the header line
            if header:
                header = False
                continue
            basic_prompt = row[0]
            #print(basic_prompt)
            chain = row[1]
            #print(chain)

# Read the specified prompt from the prompt file (.csv)
# Column 0: Base prompt w/ no COT
# Column 1: COT-enhanced prompt
def read_prompt(fname, idx):
    base_prompt_dir = "prompts"
    if not fname.endswith('.csv'):
        fname += '.csv'
    prompt_path = os.path.join(base_prompt_dir, fname)
    with open(prompt_path) as csvfile:
        reader = csv.reader(csvfile)
        rows = list(reader)
        idx = idx + 1 # skip header
        return rows[idx]


def run_codex(api_key, des_prompt, batch=False):
    openai.api_key = api_key
    response = openai.Completion.create (engine="code-davinci-002",
                    prompt=des_prompt,
                    temperature=0.1,
                    max_tokens=500, # To evenly divide into token rate limit
                    top_p=0.9,
                    frequency_penalty=0.7,
                    presence_penalty=0
                    )
    if batch:
        return [des_prompt[c.index] + c.text
                for c in sorted(response.choices, key=lambda c : c.index)]
    else:
        return response.choices[0].text

# TODO: Parse the codex output and ensure it contains valid python code.
# this basic approach does not sanitize the codex output, nor does it ensure that the
# output is properly formatted.
if __name__ == '__main__':
    api_key = read_config()
    prompt_name = 'dp_probs1'
    prompt_num = 0
    test_prompt = read_prompt(prompt_name, prompt_num)
    basic_prompt = test_prompt[0]
    cot_prompt = test_prompt[0] + test_prompt[1]
    codex_out = run_codex(api_key, cot_prompt)
    full_out = cot_prompt + codex_out
    save_output(full_out, prompt_name + '_' + str(prompt_num))

