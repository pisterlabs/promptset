"""
Generate Controls from a document

Steps
1. Group N number of files into a file
2. Generate controls for each file
3. Summarize by Control Category
4. Summarize by Control Name in each Control Category
5. Generate final report in Markdown format
"""
import os
import time
import json
import openai
from dotenv import load_dotenv
from aoai import aoai
import tiktoken

ai = aoai("gpt4")

def num_tokens_from_string(string: str) -> int:
    encoding_name ='cl100k_base'
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def grouping_chunked_txts(dir_path, number_of_files: int = 10):
    """
    Group N number of files into 1 file
    If N is 1 then there is no grouping
    """
    print("Grouping windows size: {}".format(number_of_files))
    chunk_txts = []
    temp_txts = ""

    for i, file in enumerate(os.listdir(dir_path)):
        if i % number_of_files == 0 and i != 0:
            chunk_txts.append(temp_txts)
            temp_txts = ""
        elif i == len(os.listdir(dir_path))-1:
            chunk_txts.append(temp_txts)
        
        with open(os.path.join(dir_path, file), "r", encoding='utf-8', errors='replace') as f:
            temp_txts += f.read()
    return chunk_txts


sys_prompt = """
You are a chief information security officer working as a information security auditor.

## Control Categories
The Control Category can be [{{categories}}]
The Control Category can have one or more Control Names

## Steps
1. Read the given data
2. Identify Control Category from the data, if the Control Catgory can't be decided then return 'Control not found' and stop processing
3. If Control Categories are classified then use the Control Categories and create a Contorl Name that can represent the data and the Control Category
4. Create a Contorl Description that can represent the data based on the Control Name and Control Category
5. Create a Contorl Implementation Guidance that can represent the data which include details on how to implement the control based on the Control Name and Control Category

## Response
Do not include the steps in the response. 
The response should be in the following format includes and use markdown format for the following:
Control Category

Control Name [realted to Control Category]

Control Description [realted to Control Name and it's Control Category]

Control Implementation Guidance [realted to Control Name and it's Control Category]
- details 

---
"""

categories = """Access Control,
Data Encryption,
Audit
"""

user_prompt = """Generate contorls for the following data:
--- data ---
{{data}}
------
"""

# Make sure your chunked documents are in data/doc/chunking folder
current_dir = os.path.dirname(__file__)
source_folder = os.path.join(current_dir,"data","doc","chunking")

generated_responses = []

# Step 1. Group N number of files into a file
# Step 2. Generate controls for each file
for i, txt in enumerate(grouping_chunked_txts(source_folder, 1)):
    updated_sys_prompt = sys_prompt.replace("{{categories}}", categories)
    updated_user_prompt = user_prompt.replace("{{data}}", txt)
    token_size = num_tokens_from_string(updated_sys_prompt)+ num_tokens_from_string(updated_user_prompt)
    if  token_size > 7192:
        print(f"""Prompt is too long. The prompt is {token_size} tokens long. The maximum allowed is 8192 tokens.""")
        print(f"{num_tokens_from_string(updated_sys_prompt)}")
        print(f"{num_tokens_from_string(updated_user_prompt)}")
        raise ValueError("Prompt is too long.")
    else:
        prompt = [{"role":"system", "content":updated_sys_prompt},
                  {"role":"user","content":updated_user_prompt}]

        response = ai.run(prompt, temperature=0.1, max_tokens=2000, top_p=1)
        generated_responses.append(response)
        print("{} grouping completed".format(i))

with open(os.path.join(current_dir,"data","doc","generated_response.txt"), "w") as f:
    f.write("\n".join(generated_responses))


# Step 3. Summarize by Control Category
# May take 9 minute with GPT 4 for PCI_DSS_V4
# Aggregate responses using gpt
agg_sys_prompt = sys_prompt.replace("{{categories}}", categories)
agg_user_prompt = """Review data and summarize it by removing overlaps and keep the most important information
If there are more than one Control Category then return each control Category after remove overlaps within each Control Category and keep the most important information
If there are more than one Control Name in each Control Category then return each Control Name after reorganize and remove overlaps within each Control Name and keep the most important information

--- data ---
{{data}}
------
"""

group_by_category_responses = []
agg = ""

for i, res in enumerate(generated_responses):
    # every 10th file add the accumulated string into a list
    # reset the string
    if i % 2 == 0 and i != 0:
        updated_agg_user_prompt = agg_user_prompt.replace("{{data}}", agg)
        prompt = [{"role":"system", "content":agg_sys_prompt},
                    {"role":"user","content":updated_agg_user_prompt}]            
        group_by_category_responses.append(ai.run(prompt, temperature=0.1, max_tokens=2000, top_p=1))
        agg = ""
    elif i == len(generated_responses)-1:
        updated_agg_user_prompt = agg_user_prompt.replace("{{data}}", agg)
        prompt = [{"role":"system", "content":agg_sys_prompt},
                  {"role":"user","content":updated_agg_user_prompt}]            
        group_by_category_responses.append(ai.run(prompt, temperature=0.1, max_tokens=2000, top_p=1))
    agg += "--- data {}---".format(i+1) + res + "--- ---"
    print("{} Summarize by a category completed".format(i+1))


# Step 4. Summarize by Control Name in each Control Category
# extract and summarize the responses by Control Category
# Access Control, Awareness and Training, Change Management,Continuity and Resilience,Data Privacy,Data Security,Human Resources,Incident Management,Organization and Management,Physical and Environmental,Processing Integrity,Risk Management,Security Operations,System Operations,Serverless Security,Vendor Management,Workstation Security
extract_sys_prompt = sys_prompt.replace("{{categories}}", categories)
extract_user_prompt = """Review data and extract {{category}} in Control Category 
And group it by each Control Name and summarize it by removing overlaps and keep the most important information for the Control Name

--- data ---
{{data}}
------

Includes and use markdown format for the following:
- Control Name:

- Control Description:

- Control Category: {{category}}

- Control Implementation Guidance
 - details

---
"""

control_categories = ["Access Control","Data Encryption","Audit"]

extract_by_category_responses = {}
extract = ""

for cate_idx, category in enumerate(control_categories):
    category_summary = []
    for i, agg_res in enumerate(group_by_category_responses):
        if i % 2 == 0 and i != 0:
            updated_extract_user_prompt = extract_user_prompt.replace("{{category}}", category).replace("{{data}}", extract)
            prompt = [{"role":"system", "content":extract_sys_prompt},
                        {"role":"user","content":updated_extract_user_prompt}]            
            category_summary.append(ai.run(prompt, temperature=0.1, max_tokens=2000, top_p=1))
            extract = ""
        elif i == len(group_by_category_responses)-1:
            updated_extract_user_prompt = extract_user_prompt.replace("{{category}}", category).replace("{{data}}", extract)
            prompt = [{"role":"system", "content":extract_sys_prompt},
                    {"role":"user","content":updated_extract_user_prompt}]            
            category_summary.append(ai.run(prompt, temperature=0.1, max_tokens=2000, top_p=1))
        extract += "--- data {} ---".format(i+1) + agg_res + "--- ---"
    extract_by_category_responses[category] = category_summary
    print("{} {} extraction completed".format(cate_idx+1, category))
    time.sleep(1)

# Step 5. Generate final report
# extract and summarize the responses by Control Category
# Access Control, Awareness and Training, Change Management,Continuity and Resilience,Data Privacy,Data Security,Human Resources,Incident Management,Organization and Management,Physical and Environmental,Processing Integrity,Risk Management,Security Operations,System Operations,Serverless Security,Vendor Management,Workstation Security
finaly_sys_prompt = sys_prompt.replace("{{categories}}", categories)
finaly_user_prompt = """Reorganize data for final report
make a clen report for each Control Name

--- data ---
{{data}}
------
"""


final_responses = {}
final = ""

for category in control_categories:
    for extract_reponse in extract_by_category_responses[category]:
        final += extract_reponse

    updated_finaly_user_prompt = finaly_user_prompt.replace("{{data}}", final)
    prompt = [{"role":"system", "content":finaly_sys_prompt},
                {"role":"user","content":updated_finaly_user_prompt}]            
    final_responses[category] = ai.run(prompt, temperature=0.1, max_tokens=2000, top_p=1)
    final = ""

report = """# Summary

"""

for category in control_categories:
    report += "## {}".format(category) + "\n"
    report += final_responses[category] + "\n"

with open(os.path.join(current_dir,"data","doc","doc-summary.md"), "w") as f:
    f.write(report)

