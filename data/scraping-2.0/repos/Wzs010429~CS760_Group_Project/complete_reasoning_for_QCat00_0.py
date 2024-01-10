import json
import call_openai_API
import templates
import openai
import openai_API_keys
import subprocess
import csv

# Initialize the OpenAI API client
openai.api_key = openai_API_keys.OPENAI_API_KEY
#Define the file name
JSON_filename = 'PARARULE_plus_step2_People_sample.json'
PY_filename = 'pyDatalog_processing.py'


def extract_string(input_string):
    left_boundary = 'import'
    right_boundary = ')'

    start_index = input_string.find(left_boundary)
    end_index = input_string.rfind(right_boundary, start_index)

    if start_index != -1 and end_index != -1:
        extracted_string = input_string[start_index:end_index + 1]
        return extracted_string.strip()

    return None

def check_pos_neg(string):
    words = string.split()
    for word in words:
        if word.lower() == "true":
            return "true"
        elif word.lower() == "false":
            return "false"
    return None

def Judgement(demo, question, model):
    result_string = call_openai_API.ai_generation_check(demo, question, model)
    return result_string


# Complete Communication with ChatGPT
def Generation(demo, context, question, requirements, model):

    result_string = call_openai_API.ai_function_generation(demo, context, question, requirements, model)
    return result_string

# Communication(templates.templates["agent_engineer"], PARARULE_Plus.PARARULE_Plus_dataset['train'][200]['context'], PARARULE_Plus.PARARULE_Plus_dataset['train'][200]['question'], templates.templates["no_extra_content"], "gpt-3.5-turbo")

def Adjustment(demo, code, error_message, model):

    result_string = call_openai_API.ai_generation_adjustment(demo, code, error_message, model)
    return result_string

def write_record(filename, id, value, code, step, flag):
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        rows = list(reader)

    target_id = id
    target_index = None

    for i, row in enumerate(rows):
        if row[0] == target_id:
            target_index = i
            break

    if target_index is not None:
        rows[target_index].insert(step, value)
        rows[target_index].insert(4, flag)
        rows[target_index].insert(5, code)

    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(rows)


with open(JSON_filename, 'r') as file:
    data = json.load(file)

correct_num = 0
for i in range(0, 1):
    try:
        result_string = extract_string(Generation(templates.templates["agent_engineer"], data[i]['context'],
                        data[i]['question'],
                        templates.templates["no_extra_content"], "gpt-3.5-turbo"))
        print(result_string)
        with open(PY_filename, 'w') as file:
            file.write("{}".format(result_string))
        print("processing...")
        output = subprocess.check_output(['python', PY_filename], universal_newlines=True)
        flag = 0
        while(output.strip() != '1' and output.strip() != '0'):
            result_string = extract_string(Adjustment(templates.templates["adjustment_agent"],
                                                            result_string, output, "gpt-3.5-turbo"))
            with open(PY_filename, 'w') as file:
                file.write("{}".format(result_string))
            print("reprocessing...")
            output = subprocess.check_output(['python', PY_filename], universal_newlines=True)
            print("New output:" + output)
            print(type(output))
            flag+=1
            if(flag == 3):
                break
        if (output.strip() != '1' and output.strip() != '0'):
            continue
        if int(output.strip()) == data[i]['label']:
            correct_num += 1
        else:
            continue
    except Exception as e:
        continue
print(correct_num)