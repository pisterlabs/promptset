import pandas as pd
import json 
from utils.openai_query import openai_chat
from utils.prompt_factory import make_user_prompt
from tqdm import tqdm
import openai
import argparse
import os

parser = argparse.ArgumentParser(description='Process range of gene sets.')
parser.add_argument('--config', type=str, required=True, help='Config file for LLM')
parser.add_argument('--input', type=str, required=True, help='Path to input df with gene sets')
parser.add_argument('--start', type=int, required=True, help='Start index for df range')
parser.add_argument('--end', type=int, required=True, help='End index for df range')
parser.add_argument('--input_sep', type=str, required=True, help='Separator for input csv')
parser.add_argument('--set_index', type=str, required=False, help='if set any column of the input csv as index, type the column name here')
parser.add_argument('--gene_column', type=str, required=True, help='Column name for gene sets')
parser.add_argument('--gene_sep', type=str, required=True, help='Separator for gene sets')
parser.add_argument('--gene_features', type=str, required=False, help='Path to a csv with additional gene features to include in the prompt')
parser.add_argument('--direct', action='store_true', help='If set, use direct instructions')
parser.add_argument('--customized_prompt', action='store_true', help='If set, use customized prompt txt file')

parser.add_argument('--out_file', type=str, required=True, help='Output file name (no extension))')

args = parser.parse_args()

config_file = args.config
input_file = args.input

ind_start = args.start
ind_end = args.end

input_sep = args.input_sep  
set_index = args.set_index  
gene_column = args.gene_column  
gene_sep = args.gene_sep  
gene_features = args.gene_features 
direct = args.direct
out_file = args.out_file
# load the config file
with open(config_file) as json_file:
    config = json.load(json_file)
    
    
if args.customized_prompt:
     with open(config['CUSTOM_PROMPT_FILE'], 'r') as f: # replace with your actual customized prompt file
        customized_prompt = f.read()
else:
    customized_prompt = None

# Load OpenAI key, context, and model used 
openai.api_key =  os.environ['OPENAI_API_KEY']

context = config['CONTEXT']
gpt_model = config['GPT_MODEL']
temperature = config['TEMP']
max_tokens = config['MAX_TOKENS']
rate_per_token = config['RATE_PER_TOKEN']
LOG_FILE = config['LOG_NAME']+'_log.json'
DOLLAR_LIMIT = config['DOLLAR_LIMIT']


df = pd.read_csv(input_file, sep = input_sep)
df = df.iloc[ind_start:ind_end] # only process the specified range of the input df

if set_index:
    df.set_index(set_index, inplace=True)
# Initialize columns for LLM name and analysis
df['LLM Name'] = None
df['LLM Analysis'] = None

llm_response_dict = {}
# Only process the specified range of 
i = 0
for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
    term_genes = row[gene_column]
    genes = term_genes.split(gene_sep)
    if len(genes) >1000:
        print(f'Gene set {idx} has more than 1500 genes, skipping')
        continue
    try:
        prompt = make_user_prompt(genes, feature_df=gene_features, direct=direct, customized_prompt=customized_prompt)
        analysis = openai_chat(context, prompt, gpt_model, temperature, max_tokens, rate_per_token, LOG_FILE, DOLLAR_LIMIT)
    
        if analysis:
            llm_name = analysis.split("\n")[0].replace("Process: ", "")
            df.loc[idx, 'LLM Name'] = llm_name
            
            llm_analysis = analysis.split('\n', 2)[2]
            df.loc[idx, 'LLM Analysis'] = llm_analysis

            llm_response_dict[idx] = {'prompt': prompt, 
                                      'responses': analysis, 
                                      'status': 'SUCCESS'}
        else:
            print(f'No analysis for {idx}')
            df.loc[idx, 'LLM Name'] = None
            df.loc[idx, 'LLM Analysis'] = None
            llm_response_dict[idx] = {'prompt': prompt,
                                      'responses': None,
                                      'status': 'NO RESPONSE'}
        i += 1
    except Exception as e:
        print(f'Error for {idx}: {e}')
        df.loc[idx, 'LLM Name'] = None
        df.loc[idx, 'LLM Analysis'] = None
        llm_response_dict[idx] = {'prompt': prompt,
                                    'responses': None,
                                    'status': 'ERROR: '+str(e)}
        continue
    
    if i % 10 == 0:
        with open(f'{out_file}.json', 'w') as fp:
            json.dump(llm_response_dict, fp)
        df.to_csv(f'{out_file}.tsv', sep='\t', index=True)

with open(f'{out_file}.json', 'w') as fp:
    json.dump(llm_response_dict, fp)

df.to_csv(f'{out_file}.tsv', sep='\t', index=True) 
