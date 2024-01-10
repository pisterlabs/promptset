from dotenv import load_dotenv
import openai
import json
import time
import os

def retrieve_data(dict):
    with open(r'output\job_data.json') as json_file:
        data = json.load(json_file)
    for i, item in enumerate(data):
        dict[i] = item['description']

def gpt3_analysis(index, job_description):
    
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")

    system_role = "You are analysis assistance tasked with labeling upwork job description, with one or more of following label Data Analyst (Excel), Data Analyst (Python), Data Visualization , Non Data Analyst Job. Label them if in your opinion they fall into one or more of those 4 categories. Use format of the label separated by comma. For example: Data Analyst (Excel), Data Analyst (Python), Data Visualization"

    prompt = f"This is the text need to be analyze: {job_description}"

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
        {"role": "system", "content": f"{system_role}"},
        {"role": "user", "content": f"{prompt}"}
        ]
    )

    gpt3_analysis_dict = {}
    gpt3_analysis_dict["id"] = index
    gpt3_analysis_dict["analysis"] = response['choices'][0]['message']['content']

    with open(r'output\gpt3_analysis.json', 'a', encoding='utf-8') as outfile:
        json.dump(gpt3_analysis_dict, outfile, ensure_ascii=False)
        outfile.write('\n')

    print(f"Job description {index} has been analyzed")

def main():
    data_dict = {}
    retrieve_data(data_dict)
    
    for i, job_description in data_dict.items():
        gpt3_analysis(i, job_description)
        time.sleep(1)
    
    # Turn JSON file into a list of dictionaries
    list_of_dicts = []
    with open(r'output\gpt3_analysis.json', 'r', encoding='utf-8') as json_file:
        for line in json_file:
            dict_data = json.loads(line)
            list_of_dicts.append(dict_data)

    with open(r'output\gpt3_analysis.json', 'w', encoding='utf-8') as json_file:
        json.dump(list_of_dicts, json_file, ensure_ascii=False, indent=4)
    
if __name__ == "__main__":
    main()