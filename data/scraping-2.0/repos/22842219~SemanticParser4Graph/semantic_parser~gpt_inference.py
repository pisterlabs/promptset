# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import Optional
import fire, json, os
import openai
import time

API_KEY = '<your_api_key>'
os.environ["OPENAI_API_KEY"] = API_KEY
openai.api_key = os.getenv("OPENAI_API_KEY")
_URL = os.getcwd()
Cyspider_dev_fp = _URL + "/data/text2cypher/dev.json"
Cyspider_db_fp =  _URL + "/data/text2cypher/schema.json"

schema_cache=dict()
with open (Cyspider_db_fp) as f:
    schema_cache=json.load(f)


def GPT4_generation(prompt):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            n = 1,
            stream = False,
            temperature=0.0,
            max_tokens=600,
            top_p = 1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop = ["Q:"]
        )
        print('gpt4 response', response)

        return response['choices'][0]['message']['content']
    except openai.error.RateLimitError as e:
        time.sleep(2)
        return GPT4_generation(prompt)


def main():
    nlq_indicator = {'>': ["above", "after", "older than", "heavier than", "more than", "larger/bigger than", "higher than"], 
                    '<':["below", "before", "younger than", "lighter than", "less than", 'smaller than', 'lower than'], 
                    '<=': ["at least", "or less"], '>=': ["at most", "or more"],
                    'order by ... asc': ["ascending", "in alphabetical order", "in lexicographical order", "from the youngest to the oldest",
                                        "from young to old", "from low to high"], 
                    'order by ... desc': ["descending", "in reverse alphabetical order", "in reversed lexicographical order",
                                        "from the oldest to the youngest", "from old to young", "from old to young"],
                    "order by ... asc limit": ["least", 'lowest', 'smallest', 'youngest', 'earliest', 'shortest', 'minimum', 
                                                            'fewest number', 'fewest amount'],
                    "order by ... desc limit": ['most', 'highest', 'largest', 'oldest', 'latest', 'longest', 'maximum', 'greatest number', 'greatest amount']}

    for split in [ 'dev']:
        with open(_URL + "/data/text2cypher/{}.json".format(split), encoding="utf-8") as f:
            cyspider = json.load(f)
            for idx, sample in enumerate(cyspider):
                codex=sample
                db_id = sample["db_id"] 
                nl = sample['question']
                ref_query = sample['query']
                schema = schema_cache[db_id]
                tables = schema['tag_names']
                table_column = {}
                for index, (id, col) in enumerate(schema['property_names']):
                    if id!=-1:
                        tb=tables[id]
                        if tb not in table_column:
                            table_column[tb]=[]
                        else:
                            table_column[tb].append(col)

                linearized_schema = [db_id]
                for tb, cols in table_column.items():
                    each = '{}:{}'.format(tb, ' , '.join(cols))
                    linearized_schema.append(each)
                linearized_schema = '|'.join(linearized_schema)
                
                prompt = f"Text-to-Cypher is the process of converting natural language queries or statements into Cypher queries. \
                            For example, if a user provides a question like 'What is the average, minimum, and maximum age of all singers from France?'\
                            the Text-to-Cypher would involve converting this into a Cypher query like 'MATCH (singer:`concert_singer.singer`) WHERE singer.Country = 'France' RETURN avg(singer.Age),min(singer.Age),max(singer.Age)'\
                            This Cypher query retrieves all nodes labeled as '`concert_singer.singer`' with the pair of single quote '``', 'Country' and 'Age' are the properties of the graph nodes. \
                            Implementing a Text-to-Cypher system involves natural language processing (NLP) techniques to parse and understand the user's input and then translate it into a valid Cypher query based on a predefined schema. \
                            The common natural language question indicator tokens for Cypher query keywords are {nlq_indicator}.\
                            Align the natural language indicator tokens to correspoding schema items. \
                            Given the schema for {db_id}, generate a Cypher query for each question.\
                            Question: {nl} \
                            The schema: {linearized_schema} \
                            Strictly follow the grammar of Cypher query language. For example, do not generate 'Select' keyword.\
                            Please output the generated Cypher query directly.\
                            The generated Cypher query is:"
                codex['struct_in']=linearized_schema
                
                CYPHER = GPT4_generation(prompt)
                print('generated Cypher query: ', CYPHER)     
                codex['prediction']=CYPHER.strip('"')
                print(f'ground truth: {ref_query}')
                print("\n==================================\n")
        
                with open(_URL+'/output_gpt_{}.jsonl'.format(split), "a") as json_file:
                    # for record in codex_total[split]:
                    json.dump(codex, json_file)
                    json_file.write('\n')  
                    print(f"Data has been written to {json_file}")


if __name__ == "__main__":
    fire.Fire(main)
