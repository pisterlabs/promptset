import os
import sys
import json
import time
import ast
import argparse
import openai
from tqdm import tqdm
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate


class CoTAugmenter(object):
    def __init__(self, model: str = "gpt-4", contex_size: str = "8K", temperature: float = 0.7, top_p: float = 1.0, max_tokens: int = 2048, frequency_penalty: float = 0.0, presence_penalty: float = 0.0):
        # Loading credentials
        try:
            self.load_credentials()
        except:
            print("Credentials not found. Please create a .env file with OPENAI_API_KEY and HF_USER_ACCESS_TOKEN")
            sys.exit(1)
        #instantiating the LLM
        self.llm = ChatOpenAI(
            model_name=model,
            temperature=temperature, 
            max_tokens=max_tokens
        )
        # For cost estimation
        if model == "gpt-4":
            # for 8K contex size model
            if contex_size == "8K":
                self.input_token_cost = 0.03/1000
                self.output_token_cost = 0.06/1000
            elif contex_size == "32K":
                self.input_token_cost = 0.06/1000
                self.output_token_cost = 0.12/1000
        elif model == "gpt-3.5-turbo":
            if contex_size == "4K":
                self.input_token_cost = 0.0015/1000
                self.output_token_cost = 0.002/1000
            elif contex_size == "16K":
                self.input_token_cost = 0.003/1000
                self.output_token_cost = 0.004/1000
        else:
            self.token_cost = 0.0000000
        # Class variables
        self.system_prompt = str()
        self.output_file = str()
        self.raw_data = list()
        self.augmented_data = list()

    def load_credentials(self):
        load_dotenv()
        self.hf_user_token = os.getenv("HF_USER_ACCESS_TOKEN")
        openai.api_key = os.getenv("OPENAI_API_KEY")
        openai.organization = os.getenv("OPENAI_ORG_ID")

    def parse_cot_query(self, cot_query: str):
         # Parse Cot augmented Cypher query in the valid JSON format
        try:
            parsed_cot_query = ast.literal_eval(cot_query)
            return parsed_cot_query
        except:
            parsed_cot_query = self.fix_cot_query(cot_query)
            if parsed_cot_query is None:
                print("Error in parsing Cot augmented Cypher query")
                return None
            else:
                return parsed_cot_query

    def fix_cot_query(self, cot_query: str):
        json_backticks_delimiter = "```json\n"
        standard_backticks_delimiter = "```\n"
        standard_output_backsticks_delimiter = "\n```"
        try:
            if json_backticks_delimiter in cot_query:
                cot_query = cot_query.split(json_backticks_delimiter)
                for split in cot_query:
                    if standard_output_backsticks_delimiter in split:
                        final_split = split.split(standard_output_backsticks_delimiter)
                        for f_split in final_split:
                            if f_split.startswith("{"):
                                fixed_cot_query = f_split
            elif standard_backticks_delimiter in cot_query:
                cot_query = cot_query.split(standard_backticks_delimiter)
                for split in cot_query:
                    if split.startswith("{"):
                        fixed_cot_query = split
                        fixed_cot_query = fixed_cot_query.replace(standard_output_backsticks_delimiter, "")
            fixed_cot_query = ast.literal_eval(fixed_cot_query)
            assert isinstance(fixed_cot_query, dict)
            return fixed_cot_query
        except:
            return None

    def augment(self, system_prompt: str, raw_data: list, output_file: str):
        self.system_prompt = system_prompt
        self.raw_data = raw_data
        self.output_file = output_file

        # Define Cot augmentation prompt template
        cot_augment_prompt = "{system_prompt}\n\n```{cypher_query}```"
        cot_augment_template = ChatPromptTemplate.from_template(cot_augment_prompt)

        # Iterate on each input Cypher query and use tqdm to show progress bar
        for raw_sample in tqdm(self.raw_data):
            # Generate CoT augment query from template
            cot_augment_query = cot_augment_template.format_messages(
                cypher_query=raw_sample['query'],
                system_prompt=self.system_prompt
            )
            
            # Get and time Cot augmented Cypher query from LLM
            start_time = time.time()
            cot_augment_response = self.llm(cot_augment_query)
            execution_time = time.time() - start_time

            # Get number of tokens in completion prompt and response prompt
            num_completion_tokens = self.llm.get_num_tokens_from_messages(cot_augment_query)
            num_response_tokens = self.llm.get_num_tokens(cot_augment_response.content)
            cost_completion_tokens = num_completion_tokens * self.input_token_cost
            cost_response_tokens = num_response_tokens * self.output_token_cost
            total_query_tokens = num_completion_tokens + num_response_tokens
            total_query_cost = cost_completion_tokens + cost_response_tokens

            print(cot_augment_response.content)

            cot_query = self.parse_cot_query(cot_augment_response.content)

            if cot_query is None:
                print("Error in parsing Cot augmented Cypher query. Saving war LLM response to file.")
                cot_query = cot_augment_response.content

            # Store Cot augmented Cypher query and execution time in a new dictionary
            cot_augment_sample = {
                "raw_query": raw_sample['query'], 
                "cot_query": cot_query,
                "placeholders": raw_sample['placeholders'],
                "source": raw_sample['source'],
                "execution_time": f"{execution_time} seconds",
                "num_completion_tokens": num_completion_tokens,
                "num_response_tokens": num_response_tokens,
                "cost_completion_tokens": cost_completion_tokens,
                "cost_response_tokens": cost_response_tokens,
                "total_query_tokens": total_query_tokens,
                "total_query_cost": total_query_cost
                }
            # Append the augmented data to the list of augmented data
            self.augmented_data.append(cot_augment_sample)
        
        # Write augmented data to JSON output file
        with open(self.output_file, 'w') as f:
            json.dump(self.augmented_data, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--input_file', type=str, help='Path to input file')
    parser.add_argument('--output_file', type=str, help='Path to output file')
    parser.add_argument('--model', type=str, default='gpt-4', help='OpenAI model to use')
    parser.add_argument('--temperature', type=float, default=0.0, help='Sampling temperature for OpenAI model')
    parser.add_argument('--max_tokens', type=int, default=4096, help='Maximum number of tokens to generate with OpenAI model')
    args = parser.parse_args()

    input_file = args.input_file
    output_file = args.output_file
    
    # load CoT raw test JSON from data/English-Cypher folder
    with open(input_file, 'r') as f:
        cot_raw_data = json.load(f)
    
    # 502 system prompt tokens - pretty stable system prompt for CoT augmentation
    system_prompt = "Act as an experimented Cypher query language developer. Your job is to decompose the following Cypher query  into their corresponding parse tree constituent Cypher queries and only generate a valid JSON file (using ```delimeter) as a result, avoiding completely any additional text descriptions.\n\nAs an obligatory rule, you must totally avoid breaking a Cypher query by only considering  a candidate constituent Cypher query if and only if it starts with one of the following principal Cypher clauses: MATCH, RETURN, OPTIONAL, UNION, CALL, YIELD.\nPlay special attention to placeholders, which are the main variables in the input Cypher query. Placeholder variables are delimited by # token, therefore a placeholder variable will be in the format #placeholder# and be expressed without any change within the corresponding constituent Cypher query.\n\nFor each of the constituent Cypher queries create a new entry into the  output JSON file  with key in format '{step_number}', store the constituent Cypher query in  the corresponding JSON file with the key ['query'] and write a concise and concrete description explaining the rationale behind that constituent Cypher query, storing that description in the key ['description']. The description must be as human-readable and short as possible. \nBear in mind that if the entire input Cypher query is composed by just only one MATCH Cypher clause  the resulting JSON file must only contain the entire input Cypher query in an unique constituent Cypher query. Otherwise, if the input Cypher query is composed by more than one MATCH Cypher clause, proceed by generating the set of constituent Cypher queries.\nTherefore, the final format for each constituent Cypher query must be:\n{'{step_number}' : {'query' : 'constituent Cypher query', 'description': 'constituent Cypher query description'}}\n\nFinally, generate a last entry into the output JSON file  with the key ['titles'] containing a list of 5 English sentences that would be used as titles for summarizing the input Cypher query. Each title must summarize the meaning of the entire input Cypher query but taken carefully attention to the placeholder variables within the query, which must drive the generation of the candidate titles. Each title candidate must be generated in isolation to the other title candidates carefully satisfying  the constraint that the candidate must not  contain any of the following special character like [',', '.', ':', ';']\n\nThe most important rule is that the result must be given only as a  JSON file without any additional text introductions or textual descriptions accompanying the output JSON file itself.\nHere is  the input Cypher query delimited by triple backticks:"
    
    cot_augmenter = CoTAugmenter(model=args.model, 
                                 temperature=args.temperature, 
                                 max_tokens=args.max_tokens)
    
    cot_augmenter.augment(system_prompt=system_prompt,
                          raw_data=cot_raw_data,
                          output_file=output_file)