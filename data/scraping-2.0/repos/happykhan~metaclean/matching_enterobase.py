import typer
import pandas as pd
import os 
import textwrap
import openai
import logging 
import json 
import csv
import time
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

def fetch_eb_records(entero_file="entero_all_7.7.23.tsv.gz", records=1000):
    """
    Fetches 1000 records from enterobase_all.7.7.23.tsv.gz
    """
    df = pd.read_csv(entero_file, compression='gzip', sep='\t')
    # Filter where Sample ID is not null
    df = df[df['Sample ID'].notnull() & df['Source Type'].notnull()]
    eb_records = df.sample(n=records)
    eb_records = eb_records[['Sample ID', 'Source Type', 'Source Niche']]
    biosample_ids = eb_records['Sample ID'].tolist()
    values = list(set(df['Source Type'].tolist()))
    return biosample_ids, eb_records, values



def fetch_field_list(fields="fields.csv"):
    """
    Fetches list of fields from fields.csv
    """
    # desired fields 
    wanted_fields = [
    'collected_by', 
    'isolation_source',
    'attribute_package',
    'ontological term',
    'IFSAC+ Category',
    'source_type',
    'sample_type',
    'culture_collection',
    'package',
    'attribute package',
    'IFSAC+ category',
    'food_origin',
    'disease'
    ]
    # open fields.txt
    with open(fields, 'r') as f:
        all_field_list = f.readlines()[1:]
        # remove new line characters and remove empty strings
        clean_field_list = [field.split(",")[0].strip() for field in all_field_list  if field]
        field_list = [field.split(",")[0].strip() for field in clean_field_list  if field in wanted_fields or field.startswith("host")]
    return field_list
    

def fetch_host_info(biosample_ids, data_table, field_list):
    """
    Fetches host information from all_attributes.csv.gz
    """
    biosample_details = []
    null_values = ['missing', 'not available', 'not collected', 'not applicable', 'not provided', 'not reported', 'unknown']
    # open all_attributes.csv.gz
    df = pd.read_csv(data_table, compression='gzip')
    # Filter where Sample ID is in biosample_ids
    df = df[df['Sample ID'].isin(biosample_ids)]
    df = df.reset_index()
    # fetch dictionary of biosample details using field_list
    for i in range(len(df)):
        new_biosample = { 'Sample ID' : df['Sample ID'][i]}
        for field in field_list:
            if not pd.isna(df[field][i]):
                if not df[field][i].lower() in null_values:   
                    new_biosample[field] = df[field][i]        
        biosample_details.append(new_biosample)
    return biosample_details

def create_prompt(biosample_details, eb_prompt_dir, values):
    """
    Creates a prompt text file for a biosample.
    """
    if not os.path.exists(eb_prompt_dir):
        os.makedirs(eb_prompt_dir)
    for biosample in biosample_details:
        accession = biosample.pop('Sample ID')
        biosample_string = ','.join([str(key) + ':' + str(value) for key, value in biosample.items()])
        prompt = "You will be given sample metadata. Classify the metadata into one of these host categories. Only use the categories given. Reply with the category only. Reply Not determined if unsure\nHost categories:\n{}\n::\n{}.".format(','.join(values), biosample_string)
        with open("{}/{}.txt".format(eb_prompt_dir, accession), 'w') as f:
            f.write(prompt)    

def run_prompt(prompt):
    response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {
        "role": "system",
        "content": "You will be given sample metadata. Classify the metadata into one of these host categories. Only use the categories given. Reply with the category only. Reply Not determined if unsure\n\nHost categories:\n\nCamelid,\nNot determined,\nEquine,\nFeline,\nMarine mammal,\nDairy,\nPrimate,\nSoil/Dust,\nMeat,\nHuman,\nAir,\nSwine,\nDeer,\nAmphibian,\nReptile,\nBat,\nBovine,\nShellfish,\nMarsupial,\nInvertebrates,\nOvine,\nWater,\nCanine,\nOther Mammal,\nAnimal Feed,\nFish,\nRodent,\nAvian,\nPlant,\nLaboratory,\nComposite Food."
        },
        {
        "role": "user",
        "content": "collected_by:PHE,isolation_source:human,host:Homo sapiens"
        },
        {
        "role": "assistant",
        "content": "Human"
        },
        {
        "role": "user",
        "content": "collected_by:Minnesota Department Of Health,isolation_source:pig intestine,ontological term:swine:FOODON_03411136, intestine:UBERON_0000160,IFSAC+ Category:veterinary clinical/research| pig,source_type:animal."
        },
        {
        "role": "assistant",
        "content": "Swine"
        },
        {
        "role": "user",
        "content": prompt
        }        

    ],
    temperature=0,
    max_tokens=20,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )    
    return response

def run_missing_prompts(eb_prompt_dir, prompt_output_dir='class', model="gpt-3.5-turbo", force=False):
    openai.organization = "org-HYH3Ehg9p6TomTCnKHyxm1u7"
    openai.api_key = open('metaclean_key', 'r').read().strip()
    model_list = [x.id for x in openai.Model.list()['data']]
    if not os.path.exists(prompt_output_dir):
        os.makedirs(prompt_output_dir, exist_ok=True)    
    if model in model_list:
        for prompt_file in [os.path.join(eb_prompt_dir, x) for x in os.listdir(eb_prompt_dir)]:
            prompt = open(prompt_file, 'r').read().replace('\n', ' ').strip()
            accession = os.path.basename(prompt_file).split('.')[0]
            if not os.path.exists(os.path.join(prompt_output_dir, accession + '.json')) or force:
                try:
                    if model.startswith('gpt-3') or model.startswith('gpt-4'):
                        prompt = prompt.split('::')[1]
                        response = run_prompt(prompt)
                        # response = openai.ChatCompletion.create(model=model, 
                        # messages = [{"role": "user", "content": prompt},{"role": "system", "content": "Only use categories you've been provided with"}], 
                        # temperature=0,
                        # )
                        time.sleep(1)                        
                    else:
                        # This probably doesnt work.
                        response = openai.Completion.create(
                            model=model,
                            prompt=prompt,
                            temperature=0,
                            max_tokens=10,
                            frequency_penalty=0,
                            presence_penalty=0,
                        )
                    prediction_out = {'sample_id': accession, 
                    'model': response['model'], 
                    'choices': response['choices'][0]['message']['content'], 
                    'tokens': response['usage']['total_tokens'], 
                    'prompt': prompt
                    }
                    # Write output to json file
                    with open(os.path.join(prompt_output_dir, accession + '.json'), 'w') as f:
                        f.write(json.dumps(prediction_out))
                    logging.info('Added {} to {}'.format(accession, prompt_output_dir))
                except openai.error.RateLimitError:
                    logging.error('You exceeded your current quota, please check your plan and billing details.')
                except openai.error.ServiceUnavailableError:
                    # Sleep for 3 seconds and try again
                    logging.error('Service unavailable, sleeping for 3 seconds')
                    logging.error('May have skipped {}'.format(accession))
                    time.sleep(3)
                    continue
    else:
        logging.error(f'Model {model} not found. Please check your spelling and try again.')

def compare_eb_records(entero_file, prompt_output_dir='class'):
    df = pd.read_csv(entero_file, compression='gzip', sep='\t')
    # Filter where Sample ID is not null
    df = df[df['Sample ID'].notnull() & df['Source Type'].notnull()]
    eb_records = df[['Sample ID', 'Source Type', 'Source Niche']]
    prompt_comp = []
    match_rate = 0
    total = 0
    for prompt_file in [os.path.join(prompt_output_dir, x) for x in os.listdir(prompt_output_dir)]:
        total += 1
        prompt_results = json.load(open(prompt_file, 'r'))
        accession = os.path.basename(prompt_file).split('.')[0]
        eb_record = eb_records[eb_records['Sample ID'] == accession]
        eb_record_result = eb_record['Source Type'].values[0]
        prompt_out_res = {'Sample_ID': accession, 'EB_Source_type': eb_record_result, 'GPT_Source_type': prompt_results['choices'], 'Prompt': prompt_results['prompt'].strip()}
        prompt_comp.append(prompt_out_res)
        if eb_record_result == prompt_results['choices']:
            logging.info('Matched {}. {}'.format(accession, eb_record_result))
            match_rate += 1
        else:
            logging.warn('No match {}. EB: {} vs GPT: {}. The prompt was: {}'.format(accession, eb_record_result, prompt_results['choices'], prompt_results['prompt']))
    logging.info('{} out of {}. Match rate: {}%'.format(match_rate, total, round(match_rate/total * 100)))
    # write prompt_comp to file ith dictwriter
    with open('prompt_comp.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['Sample_ID', 'EB_Source_type', 'GPT_Source_type', 'Prompt'])
        writer.writeheader()
        for row in prompt_comp:
            writer.writerow(row)

def main(fields:str="fields.csv", data_table:str="all_attributes.csv.gz", entero_file:str ="entero_all_7.7.23.tsv.gz", num_records:int=100, eb_prompt_dir:str="eb_prompts"):
    # Pick 1000 enterobase records from enterobase_all.7.7.23.tsv
    biosample_ids, eb_records, values = fetch_eb_records(entero_file=entero_file, records=num_records)
    # Fetch list of fields and extract valid fields to use for host 
    field_list = fetch_field_list(fields)
    # Fetch corresponging host information from all_attributes.csv.gz
    input_data = fetch_host_info(biosample_ids, data_table, field_list)
    # Generate prompt text file for each biosample record
    create_prompt(input_data, eb_prompt_dir, values)
    # Run missing prompts - careful, this costs money!
    # run_missing_prompts(eb_prompt_dir)
    # Compare prompt responses to EnteroBase records
    compare_eb_records(entero_file, prompt_output_dir='class')

if __name__ == "__main__":
    typer.run(main)

