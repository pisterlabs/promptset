# prompt formatter - read from an external file
# example_variables - read from an external file
# prompt template - langChain
# few-shot prompt template - langChain
# prefix + suffix - read from an external file
# examples - read from an external file
# example selector - langChain
# input variables - read from an external file
# outfile - write to an external file

from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain import PromptTemplate, FewShotPromptTemplate
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector, MaxMarginalRelevanceExampleSelector
from langchain.vectorstores import Chroma
from tqdm import tqdm
import json
import random
import os
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logging.info('Start Logging')



def sample_examples(data_folder, training_file, sample_size):
    """Sample examples from a list of examples. 
    Write the sampled examples to a new file.
    Write the remaining examples to a new file.
    """	
    all_examples = []
    with open(f'{data_folder}/{training_file}', 'r', encoding='utf-8') as f:
        examples = f.readlines()
        for example in examples:
            example = json.loads(example)
            all_examples.append(example)
    
    selected_items = random.sample(all_examples, sample_size)
    
    remaining_items = [item for item in all_examples if item not in selected_items]
    
    with open(f'{data_folder}/train_after_sampling.jsonl', 'w', encoding='utf-8') as f:
        for item in remaining_items:
            f.write(f'{json.dumps(item)}\n')
            
    with open(f'{data_folder}/sample_from_train.jsonl', 'w', encoding='utf-8') as f:
        for item in selected_items:
            f.write(f'{json.dumps(item)}\n')
    

def get_examples(path_to_examples):
    """Read from an external file. List of dicts.
        Dicts keys: 'Text', 'Entities', 'Entity Types', 'Entity wiki ids', 'Relations', 'Relation wiki ids', 'Triples', 'Wiki Triples'
    """	
    examples = []
    with open(path_to_examples, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for example in lines:
            example = json.loads(example)
            example['Entities'] = str(example['Entities'])
            example['Entity Types'] = str(example['Entity Types'])
            example['Entity wiki ids'] = str(example['Entity wiki ids'])
            example['Relations'] = str(example['Relations'])
            example['Relation wiki ids'] = str(example['Relation wiki ids'])
            example['Triples'] = str(example['Triples'])
            example['Wiki Triples'] = str(example['Wiki Triples'])
            example['Mixed Triples'] = str(example['Mixed Triples'])
            example['Corrupted Triples'] = str(example['Corrupted Triples'])
            example['Explanations'] = str(example['Explanations'])
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
    
    parser.add_argument('--sample_exists', type=bool, default=True, help='Sample for example selection exists or not')
    parser.add_argument('--data_folder', type=str, default='data/RED-fm', help='Data folder for sampling')
    parser.add_argument('--training_file', type=str, default='train.jsonl', help='Training file for sampling')
    parser.add_argument('--input_file', type=str, default='data/RED-fm/test.jsonl', help='Input file')
    parser.add_argument('--sample_file', type=str, default='data/RED-fm/sample_from_train.jsonl', help='Sample file')
    parser.add_argument('--prompt_components', type=str, default='prompt_templates/one_shot_RAG_prompt.json', help='Prompt components')
    parser.add_argument('--example_selector', type=str, default='max_marginal_relevance', help='Example selector type')
    parser.add_argument('--k', type=int, default=1, help='number of example to add to the prompt')
    parser.add_argument('--output_dir', type=str, default='data_with_prompts', help='Output directory to save prompts')    
    args = parser.parse_args()
    
    if not args.sample_exists:
        sample_examples(args.data_folder, args.training_file, 300)
        logging.info('Examples sampled successfully')
        
    if not os.path.exists(args.output_dir): # Create a new directory because it does not exist
        os.makedirs(args.output_dir)
        logging.info(f'Created a new directory ({args.output_dir}) to save the data with prompts')
        
    examples = get_examples(args.sample_file)
    
    embeddings = HuggingFaceInstructEmbeddings(query_instruction="Represent the query for retrieval: ")
    k = args.k
    
    if args.example_selector == 'semantic_similarity':
        example_selector = SemanticSimilarityExampleSelector.from_examples(
            examples,
            embeddings,
            Chroma,
            k
            )
        
    if args.example_selector == 'max_marginal_relevance':
        example_selector = MaxMarginalRelevanceExampleSelector.from_examples(
            examples,
            embeddings,
            Chroma,
            k
            )
        
    prompt_dict = get_fixed_prompt_components(args.prompt_components)
    
    prefix = prompt_dict['prefix']
    example_variables = prompt_dict['example variables']
    #print(example_variables)
    formatter = prompt_dict['formatter']
    input_variables = prompt_dict['input variables']
    #print(input_variables)
    suffix = prompt_dict['suffix']
    
    prompt_template = PromptTemplate(
                input_variables=example_variables,  
                template=formatter,
                )
    
    few_shot_template = FewShotPromptTemplate(
        example_selector= example_selector, 
        example_prompt=prompt_template,
        prefix=prefix,
        suffix=suffix,
        input_variables=input_variables,
        example_separator="\n"
        )
    
    data = get_examples(args.input_file)
    total_instances = len(data)
    
    prompt_template_name = args.prompt_components.split('/')[-1]
    prompt_template_name = prompt_template_name.split('.')[0]
    #print(prompt_template_name)
        
    input_file = args.input_file.split('/')[-1]
    input_file = input_file.split('.')[0]   
    
    prompts = []
    # Wrap the loop with tqdm to create a progress bar
    for index, instance in enumerate(tqdm(data, desc=f"Compiling prompts and writing them to outfile", total=total_instances)):
        text = instance['Text']
        prompt = few_shot_template.format(Text = text)
        #print(prompt)
        prompts.append(prompt)

    with open(f'{args.output_dir}/{prompt_template_name}s_{input_file}.json', 'w', encoding='utf-8', ) as just_prompts:
        json.dump(prompts, just_prompts, indent=4, ensure_ascii=False)
        
    logging.info('Prompts compiled and saved successfully')
    
    
if __name__ == '__main__':
    
    main()
