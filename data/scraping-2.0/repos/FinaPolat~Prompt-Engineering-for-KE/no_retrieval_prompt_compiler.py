

from langchain import PromptTemplate
from tqdm import tqdm
import json
import os
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logging.info('Start Logging')

def get_examples(path_to_examples):
    """Read from an external file. List of dicts.
        Dicts keys: 'Text', 'Entities', 'Relations', 'Triples', 'Wiki Triples'
    """	
    examples = []
    with open(path_to_examples, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for example in lines:
            example = json.loads(example)
            example['Hint'] = [entity[:2] for entity in example['Entities']]
            example['Entities'] = str(example['Entities'])
            example['Relations'] = str(example['Relations'])
            example['Triples'] = str(example['Triples'])
            example['Wiki Triples'] = str(example['Wiki Triples'])
            examples.append(example)
    
    return examples


def get_fixed_prompt_components(path_to_prompt_components):
    """Read from an external file: json file
    """	
    with open(path_to_prompt_components, 'r', encoding='utf-8') as f:
        prompt_dict = json.load(f)
        
    return prompt_dict
    
        
  
def main():  
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--input_file', type=str, default='data/RED-fm/test.jsonl', help='Input file')
    parser.add_argument('--prompt_components', type=str, default='prompt_templates/three_shot_prompt.json', help='Prompt components')
    parser.add_argument('--output_dir', type=str, default='data_with_prompts', help='Output directory to save prompts')    
    args = parser.parse_args()
    
    if not os.path.exists(args.output_dir): # Create a new directory because it does not exist
        os.makedirs(args.output_dir)
        logging.info(f'Created a new directory ({args.output_dir}) to save the data with prompts')
    
    data = get_examples(args.input_file)
    total_instances = len(data)
    
    prompt_template_name = args.prompt_components.split('/')[-1]
    prompt_template_name = prompt_template_name.split('.')[0]
    #print(prompt_template_name)
        
    prompt_dict = get_fixed_prompt_components(args.prompt_components)
    
    
    formatter = prompt_dict['formatter']
    #print(formatter)
    input_variables = prompt_dict['input variables']
    #print(input_variables)

    
    zero_shot_template = PromptTemplate(
                input_variables=input_variables,  
                template=formatter,
                )
    
    input_file = args.input_file.split('/')[-1]
    input_file = input_file.split('.')[0]   
    
    prompts = []
    # Wrap the loop with tqdm to create a progress bar
    for index, instance in enumerate(tqdm(data, desc=f"Compiling prompts and writing them to outfile", total=total_instances)):
        text = instance['Text']
        prompt = zero_shot_template.format(Text = text)
        
        #print(prompt)
        prompts.append(prompt)

    with open(f'{args.output_dir}/{prompt_template_name}s_{input_file}.json', 'w', encoding='utf-8') as just_prompts:
        json.dump(prompts, just_prompts, indent=4, ensure_ascii=False)
        
    logging.info('Prompts compiled and saved successfully')
    
    
if __name__ == '__main__':
    
    main()
