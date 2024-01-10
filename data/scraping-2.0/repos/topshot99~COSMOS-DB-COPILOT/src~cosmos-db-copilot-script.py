#Note: The openai-python library support for Azure OpenAI is in preview.
import os
import sys
import openai

openai.api_type = "azure"
openai.api_base = "https://<instancr>.openai.azure.com/"
openai.api_version = "2023-05-15"
openai.api_key = "api_key" #replace with api key

code_snippet = sys.argv[1]

rules = f"""
You are cosmos db code owner whose role is to assist people in writing code and suggesting best practices related to cosmos db code based on certain rules provided below.
Apart from the rules you are also allowed to answer based on your own knowledge if something is not present in rules.
Only output the corrected code in the end nothing else no text;
 Rules: 
 
 1) For all the item or document related methods partition key should be present in request, if not convey to developer, exclude query method cases 
 
 2) Always create query iterator once and use same object for performing operations associated with it 
 
 3) we dont support Queries with COUNT from a DISTINCT subquery​
  
 4) we dont support Direct TCP Mode access​ 
 
 5) we don't support Continuation token for cross partitions queries 
 
 6) We don't support Change Feed: Processor 
 
 7) We don't support Change Feed: Read multiple partitions key values 
 
 8) We don't support Change Feed: Read specific time 
 
 9) We don't support Change Feed: Read from the beginning 
 
 10) We don't support Change Feed: Pull model 
 
 11) We don't support Cross-partition ORDER BY for mixed types 


Keeping these rules in mind provide recommendations for this snippet: 

"""

def find_cosmos_db_best_practices(rules, code_snippet) :
    response = openai.ChatCompletion.create(
        engine = "code-copilot-cosmos-db",
        messages=[
            {"role": "system", "content":  rules},
            {"role": "user", "content": code_snippet},
        ]
    )
    return response['choices'][0]['message']['content']


print(find_cosmos_db_best_practices(rules, code_snippet))
