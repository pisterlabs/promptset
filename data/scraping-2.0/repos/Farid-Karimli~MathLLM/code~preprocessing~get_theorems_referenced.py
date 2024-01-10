import os
from openai import AsyncOpenAI, APIConnectionError, APIConnectionError, RateLimitError
import asyncio
from dotenv import load_dotenv
from tqdm import tqdm
import json
import argparse
import pandas as pd
import re
# Set your OpenAI API key here

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY') 
client = AsyncOpenAI()


function_schema = {
  "name": "proof_statements",
  "description": "A function that takes as input all theorems, definitions, corollaries, and propositions used in a proof from a math text",
  "parameters": {
      "referenced_statements": {
                "type": "object",
                "description": "Dictionary of type of statement referenced and the identifying number (and possibly letter of it.)",
                "properties": {
                    "statement_type": {
                        "type": "string",
                        "enum": ["theorem", "definition", "corollary", "proposition"],
                        "description": "The type of statement referenced."
                    },
                    "statement_label": {
                        "type": ["string"],
                        "description": "The identifying number and, if present, letter of the statement."
                    }
                }
            }
        },
        "required": ["referenced_statements"]
}

def create_sample_resopnses(json_input):
    return {"role": "assistant", "content": None, "function_call": {"name": "clean_text", "arguments": json_input}}

async def extract_theorems(chapter_text, model_type="gpt-3.5-turbo-1106"):
    global content_example, example_output, example_4, output_4, example_5, output_5, example_6, output_6, example_7
    while True:
        try: 
            response =  await client.chat.completions.create(
                model=model_type,
                messages=[{"role": "system", "content": "You are a machine that takes as input statements from a math text, and \
                passes the relevant parts of it as input into the given function provided in JSON."},
                {"role": "user", "content": "Parse, add, and extract the relevant data from this input to use as arguments to pass into the given function provided:" + content_example},
                create_sample_resopnses(json.dumps(example_output)),
                 {"role": "user", "content": "Parse, add, and extract the relevant data from this input to use as arguments to pass into the given function provided:" + example_3},
                create_sample_resopnses(json.dumps(output_3)),
                {"role": "user", "content": "Parse, add, and extract the relevant data from this input to use as arguments to pass into the given function provided:" + example_4},
                create_sample_resopnses(json.dumps(output_4)),
                {"role": "user", "content": "Parse, add, and extract the relevant data from this input to use as arguments to pass into the given function provided:" + example_5},
                create_sample_resopnses(json.dumps(output_5)),
                 {"role": "user", "content": "Parse, add, and extract the relevant data from this input to use as arguments to pass into the given function provided:" + example_6},
                create_sample_resopnses(json.dumps(output_6)), 
                {"role": "user", "content": chapter_text}],
                functions=[function_schema],
                function_call={"name": "clean_text"},
                temperature=0,
            )
            break
        except (APIConnectionError, APIConnectionError, RateLimitError) as e: # handle ratelimit error
            await asyncio.sleep(60)
        except Exception as e:
            exit(1)
    return response.choices[0].message.function_call.arguments.strip()


def string_to_dicts(ret_dict):
    # using get with empty dictionary in case gpt fails on output
    return ret_dict.get('statement',{})

def safe_list_get(l, idx, default = "Error, no proof value given."):
  try:
    return l[idx]
  except Exception:
    return default

async def extract_correct_theorems(text):
    #try getting JSON
    global example_output_empty
    model = "gpt-4-1106-preview" 
    ret = await extract_theorems(text, model_type=model)
    ret = json.loads(ret)
    return string_to_dicts(ret)['statement']

def sorting_key(s):
    # Regular expression to match the number and optional letter
    match = re.match(r"(\d+\.\d+)(\w*)", s.replace(" ", "").replace("(", "").replace(")", ""))
    number = float(match.group(1))
    letter = match.group(2) if match.group(2) else ""
    return (number, letter)

def sort_df(df):
    df['sorting_key'] = df.iloc[:, 0].apply(sorting_key)
    df_sorted = df.sort_values(by='sorting_key').drop('sorting_key', axis=1)
    return df_sorted

async def process_md_files(step, folder_path):

    with open(f"{folder_path}/theorems.csv", 'a+', newline='', encoding='utf-8') as theorems_file, \
         open(f"{folder_path}/definitions.csv", 'a+', newline='', encoding='utf-8') as definitions_file, \
         open(f"{folder_path}/corollaries.csv", 'a+', newline='', encoding='utf-8') as corollaries_file, \
         open(f"{folder_path}/propositions.csv", 'a+', newline='', encoding='utf-8') as propositions_file:

        theorems_writer = csv.DictWriter(theorems_file, fieldnames=theorem_headers)
        definitions_writer = csv.DictWriter(definitions_file, fieldnames=definition_headers)
        corollaries_writer = csv.DictWriter(corollaries_file, fieldnames=corollary_headers)
        propositions_writer = csv.DictWriter(propositions_file, fieldnames=proposition_headers)

        theorems_writer.writeheader()
        definitions_writer.writeheader()
        corollaries_writer.writeheader()
        propositions_writer.writeheader()

        tasks = []
        for filename in os.listdir(folder_path):
            if filename.endswith('.md'):
                file_path = os.path.join(folder_path, filename)
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                # Split content into chunks
                chunks = get_chunks(content)
                for chunk in chunks:
                    # Create and append the task
                    task = asyncio.create_task(extract_correct_theorems(chunk, folder_path, 0))
                    tasks.append(task)

        #current = current_process()
        pbar = tqdm(total=len(chunks), desc=f"Processing file {filename}", position=step + 1)

        # write to csv ask tasks complete  
        for task in asyncio.as_completed(tasks):
            theorems_temp, definitions_temp, corollaries_temp, propositions_temp = await task
            for key, value in theorems_temp.items(): 
                theorems_writer.writerow({'Theorem': key, 'Statement': safe_list_get(value, 0, "No statement given"), 'Proof': safe_list_get(value, 1)})
            for key, value in definitions_temp.items():
                definitions_writer.writerow({'Definition': key, 'Statement': value})
            for key, value in corollaries_temp.items():
                corollaries_writer.writerow({'Corollary': key, 'Statement': safe_list_get(value, 0, "No statement given"), 'Proof': safe_list_get(value, 1)})
            for key, value in propositions_temp.items():
                propositions_writer.writerow({'Proposition': key, 'Statement': safe_list_get(value, 0, "No statement given"), 'Proof': safe_list_get(value, 1)})

            pbar.update(1)

        pbar.close()

    return 


async def main(mathllm_folder, book, step):   
    await process_md_files(step, os.path.join(mathllm_folder, f'raw_data/{book}'))
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--book",  type=str)
    parser.add_argument("-p", "--path",  type=str)
    parser.add_argument("-s", "--step", type=int)
    args = parser.parse_args()

    asyncio.run(main(args.path,args.book, args.step))
    print('Done')
