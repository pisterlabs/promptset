## this the the part we want to check the effectiveness of semantic part retrieval
import json
import call_openai_API
import templates
import openai
import subprocess
import csv
import os
import random

# Initialize the OpenAI API client
openai.api_key = api_key = os.getenv("OPENAI_API_KEY")
#Define the file name
# JSON_filename = 'PARARULE_plus_step2_People_sample.json'
# replace the file with the following JSON_filenames
# "PARARULE_plus_step2_Animal_sample.json",
# "PARARULE_plus_step3_Animal_sample.json",
# "PARARULE_plus_step4_Animal_sample.json",
# "PARARULE_plus_step5_Animal_sample.json",
# "PARARULE_plus_step2_People_sample.json",
# "PARARULE_plus_step3_People_sample.json",
# "PARARULE_plus_step4_People_sample.json",
# "PARARULE_plus_step5_People_sample.json"

file_names = [
    "../PARARULE_plus_step2_Animal_sample.json",
    "../PARARULE_plus_step3_Animal_sample.json",
    "../PARARULE_plus_step4_Animal_sample.json",
    "../PARARULE_plus_step5_Animal_sample.json",
    "../PARARULE_plus_step2_People_sample.json",
    "../PARARULE_plus_step3_People_sample.json",
    "../PARARULE_plus_step4_People_sample.json",
    "../PARARULE_plus_step5_People_sample.json"
]


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


def Judgement(demo, question, model):
    result_string = call_openai_API.ai_generation_check(demo, question, model = "gpt-4")
    return result_string


# Complete Communication with ChatGPT
def Generation(demo, context, question, requirements, model = "gpt-4"):

    result_string = call_openai_API.ai_function_generation(demo, context, question, requirements, model)
    return result_string

def BackConvertion(demo, code, model = "gpt-4"):
    result_string = call_openai_API.ai_function_backconvertion(demo, code, model)
    return result_string

# Communication(templates.templates["agent_engineer"], PARARULE_Plus.PARARULE_Plus_dataset['train'][200]['context'], PARARULE_Plus.PARARULE_Plus_dataset['train'][200]['question'], templates.templates["no_extra_content"], "gpt-3.5-turbo")

def Adjustment(demo, code, error_message, model = "gpt-4"):

    result_string = call_openai_API.ai_generation_adjustment(demo, code, error_message, model)
    return result_string

def Extraction(demo, text, model = "gpt-4"):
    result_string = call_openai_API.ai_function_extraction(demo, text, model)
    return result_string

def Comparison(demo, original, generated, model = "gpt-4"):
    result_string = call_openai_API.ai_function_comparison(demo,  original, generated, model)
    return result_string


def Regeneration(demo, code, text, model = "gpt-4"):
    result_string = call_openai_API.ai_function_regeneration(demo, code, text, model)
    return result_string

# load the data
data = []
for file_name in file_names:
    with open(file_name, 'r', encoding='utf-8') as json_file:
        tmp = json.load(json_file)
        data.extend(tmp)

# select 50 records randomly
data = random.sample(data, 500)
print(data)
# the basement without converting the propositions back to the code
accuracy = 0
for i in range(0, 5):
    try:
        # first time generate the code from propositions
        result_string = extract_string(Generation(templates.templates["agent_engineer"], data[i]['context'],
                        data[i]['question'],
                        templates.templates["no_extra_content"]))
        # print(result_string)

        # save the code into the file
        with open(PY_filename, 'w') as file:
            file.write("{}".format(result_string))
        output = subprocess.check_output(['python', PY_filename], universal_newlines=True)
        print(f"output: {output}")
        if (output.strip() != "1" and output.strip() != "0"):
            continue
        else:
            accuracy += 1
    except Exception as e:
        continue




# test the accuracy if we add the back convertion part in to the framework
correct_num_flag0 = 0
correct_num_flag3 = 0
for i in range(0, 50):
    try:
        # first time generate the code from propositions
        result_string = extract_string(Generation(templates.templates["agent_engineer"], data[i]['context'],
                        data[i]['question'],
                        templates.templates["no_extra_content"]))
        # print(result_string)

        # convert code back 2 propositions
        propositions_generated = BackConvertion(templates.templates["agent_engineer_neg"], result_string)

        # Comparison
        # zero-shot CoT is here
        tag = Comparison(templates.templates["check_error_part1"], f"Propositions:{data[i]['context']}, Question:{data[i]['question']}", propositions_generated)
        tag_final = Extraction(templates.templates["check_error_part2"], tag)
        print(f"tag: {tag}")
        print(f"tag_final: {tag_final}")
        # if it pass the comparison
        if "true" in tag_final:
            print("no need to regenerate")
            flag = 0
            with open(PY_filename, 'w') as file:
                file.write("{}".format(result_string))
            output = subprocess.check_output(['python', PY_filename], universal_newlines=True)
            print(f"output: {output}")
            while (output.strip() != "1" and output.strip() != "0"):
                result_string = extract_string(Adjustment(templates.templates["adjustment_agent"],
                                                            result_string, output))
                with open(PY_filename, 'w') as file:
                    file.write("{}".format(result_string))
                print("reprocessing...")
                output = subprocess.check_output(['python', PY_filename], universal_newlines=True)
                print("New output:" + output)
                print(type(output))
                if flag == 0 and (output.strip() == "1" or output.strip() == "0"):
                    correct_num_flag0 += 1
                flag += 1
                if (flag == 3):
                    break
        else:
            print("enter the regeneration part")
            # regenaration
            result_string = extract_string(Regeneration(templates.templates["regeneration"], f"Propositions:{data[i]['context']}, Question:{data[i]['question']}", result_string, tag_final))
            print(f"regeneration result: {result_string}")
            with open(PY_filename, 'w') as file:
                file.write("{}".format(result_string))
            output = subprocess.check_output(['python', PY_filename], universal_newlines=True)
            flag = 0
            while (output.strip() != "1" and output.strip() != "0"):
                result_string = extract_string(Adjustment(templates.templates["adjustment_agent"],
                                                            result_string, output))
                with open(PY_filename, 'w') as file:
                    file.write("{}".format(result_string))
                print("reprocessing...")
                output = subprocess.check_output(['python', PY_filename], universal_newlines=True)
                print("New output:" + output)
                print(type(output))
                if flag == 0 and (output.strip() == "1" or output.strip() == "0"):
                    correct_num_flag0 += 1
                flag += 1
                if (flag == 3):
                    break

        # check correctness
        # if (output.strip() != '1' and output.strip() != '0'):
        #     correct_num_flag0 += 1
        if int(output.strip()) == data[i]['label']:
            correct_num_flag3 += 1
        else:
            continue
    except Exception as e:
        continue

print(f"accuracy number: {accuracy}")
print(f"correct_num_0: {correct_num_flag0}")
print(f"correct_num_3: {correct_num_flag3}")