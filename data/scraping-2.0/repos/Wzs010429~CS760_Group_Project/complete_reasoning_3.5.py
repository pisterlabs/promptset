import json
import call_openai_API
import templates
import openai
import subprocess
import csv
import os

# Initialize the OpenAI API client
openai.api_key = api_key = os.getenv("OPENAI_API_KEY")
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


def Judgement(demo, question, model):
    result_string = call_openai_API.ai_generation_check(demo, question, model = "gpt-3.5-turbo")
    return result_string


# Complete Communication with ChatGPT
def Generation(demo, context, question, requirements, model = "gpt-3.5-turbo"):

    result_string = call_openai_API.ai_function_generation(demo, context, question, requirements, model)
    return result_string

def BackConvertion(demo, code, model = "gpt-3.5-turbo"):
    result_string = call_openai_API.ai_function_backconvertion(demo, code, model)
    return result_string

# Communication(templates.templates["agent_engineer"], PARARULE_Plus.PARARULE_Plus_dataset['train'][200]['context'], PARARULE_Plus.PARARULE_Plus_dataset['train'][200]['question'], templates.templates["no_extra_content"], "gpt-3.5-turbo")

def Adjustment(demo, code, error_message, model = "gpt-3.5-turbo"):

    result_string = call_openai_API.ai_generation_adjustment(demo, code, error_message, model)
    return result_string

def Extraction(demo, text, model = "gpt-3.5-turbo"):
    result_string = call_openai_API.ai_function_extraction(demo, text, model)
    return result_string

def Comparison(demo, original, generated, model = "gpt-3.5-turbo"):

    result_string = call_openai_API.ai_function_comparison(demo,  original, generated, model)
    return result_string


def Regeneration(demo, code, text, model = "gpt-3.5-turbo"):
    result_string = call_openai_API.ai_function_regeneration(demo, code, text, model)
    return result_string





with open(JSON_filename, 'r') as file:
    data = json.load(file)


correct_num = 0
for i in range(0, 40):
    try:
        # first time generate the code from propositions
        result_string = extract_string(Generation(templates.templates["agent_engineer"], data[i]['context'],
                        data[i]['question'],
                        templates.templates["no_extra_content"]))
        print(result_string)

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
                flag += 1
                if (flag == 3):
                    break

        # check correctness
        # if (output.strip() != '1' and output.strip() != '0'):
        #     continue
        if int(output.strip()) == data[i]['label']:
            correct_num += 1
        else:
            continue
    except Exception as e:
        continue
print(f"correct_num: {correct_num}")