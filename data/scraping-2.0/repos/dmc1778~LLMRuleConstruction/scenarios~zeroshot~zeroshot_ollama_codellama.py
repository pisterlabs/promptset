from langchain_community.llms.ollama import Ollama
import time, json
codellama = Ollama(base_url='http://localhost:11434', model='codellama')
codellama13b = Ollama(base_url='http://localhost:11434', model='codellama:34b')

def create_prompt_fix_sugesstion(item):
    prompt_ = f"""
    You are a bug fix chatbot. You will receive a buggy code snippet along with the corresponding context information 
    about the bug and output the fixed code. Do NOT try to generate the fix code if you don't know how to fix the buggy code.
    ONLY generate the code, do not explain how to fix it. 
    For each buggy code, you have the following context (wrappted by @@):

    API Name: @@{item['API Name']}@@
    Vulnerability Category: @@{item['Vulnerability Category']}@@
    Description of how the vulnerability is triggered when calling @@{item['API Name']}@@: {item["Trigger Mechanism"]}
    Vulnerable code in the backend: @@{item['Vulnerable Code']}@@
    
    Please generate the patches/code (C++ code) fix in given json format (generate correct JSON format):    
    <answer json start>,
    "Patch":"Generated patch"

    """

    return prompt_

if __name__ == '__main__':
    lib_name = 'tf'
    rules_path = f"scenarios/{lib_name}_code_fixes.json"

    with open(f'scenarios/{lib_name}_bug_data_sample.json') as json_file:
        data = json.load(json_file)
        for j, item in enumerate(data):
            print(f"Record {j}/{len(data)}")
            prompt_ = create_prompt_fix_sugesstion(item)
            output = codellama13b(prompt_)     
            try:
                # output = output.split('\n')
                x = json.loads(output)
                x.update({'Link': item['Commit Link']})
                x.update({'API name': item['API Name']})
                with open(rules_path, "a") as json_file:
                    json.dump(x, json_file, indent=4)
                    json_file.write(',')
                    json_file.write('\n')
            except Exception as e:
                    print(e)


color = 'Red'

print(output)
