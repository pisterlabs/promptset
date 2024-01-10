import os
import json
import openai
from retry import retry
import argparse
import logging
import time
from tqdm import tqdm

os.environ["OPENAI_API_KEY"] = 'your_api_key'
openai.api_key = os.environ["OPENAI_API_KEY"]


def read_prompts(input_file): 
    
    with open(input_file, 'r', encoding='utf-8') as prompt_file:
        prompts = json.load(prompt_file)
        
    return prompts


# Get an answer from the OpenAI-API
@retry(tries=3, delay=5, max_delay=25)
def GPT_repsonse(prompt, model, temperature, max_tokens ): 
    message=[{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
                                            model=model,
                                            messages=message,
                                            temperature=temperature,
                                            max_tokens=max_tokens,
    )
    response = response.choices[0]["message"]["content"]
    
    return response


def main(): 
    
    logging.basicConfig(level=logging.INFO)
    logging.info('Start Logging')
    # Record the start time
    start_time = time.time()
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--input_folder', type=str, default='data_with_prompts', help='Input directory')
    parser.add_argument('--model_name', type=str, default='gpt-4', help='OpenAI model name')
    parser.add_argument('--out_dir', type=str, default='generated_output_by_GPT4', help='Output directory')
    
    args = parser.parse_args()
    
    if os.path.exists(args.input_folder) and os.path.isdir(args.input_folder):
       
        file_names = os.listdir(args.input_folder)
    else:
        print(f"The folder '{args.input_folder}' does not exist or is not a directory.")
        
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    
    for file in file_names[5:11]:
        prompt_file = f'{args.input_folder}/{file}'
        prompts = read_prompts(prompt_file)
        #prompts = prompts[:3]
        total_instances = len(prompts)
    
        logging.info(f'Read the input file: {prompt_file}. Number of instances: {total_instances}. Extraction started.')
    
        generated_extractions = []
    
        for i, prompt in enumerate(tqdm(prompts, desc=f"Passing inputs through {args.model_name} for inference", total=total_instances)):
            extraction = GPT_repsonse(prompt, args.model_name, temperature=0.5,  max_tokens=2000)
            generated_extractions.append({'index': i, 'prompt': prompt, 'extraction': extraction})
    
        outfile = prompt_file.split('/')[-1]
    
        logging.info(f'Extraction finished. Saved the results to {args.out_dir}/{outfile}')
    
        with open(f'{args.out_dir}/{outfile}', 'w', encoding='utf-8') as output_file:
            json.dump(generated_extractions, output_file, indent=4, ensure_ascii=False)
        
    end_time = time.time()
    
    # Calculate the elapsed time
    elapsed_time = (end_time - start_time) / 60
    print(f"Elapsed time: {elapsed_time:.2f} minutes")
        
if __name__ == '__main__':
    main()



