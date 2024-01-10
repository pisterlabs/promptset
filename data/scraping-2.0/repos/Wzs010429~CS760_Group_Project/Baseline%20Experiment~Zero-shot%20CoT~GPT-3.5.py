import openai
import json
import csv
import re
import os

def ai_function_generation(demo, context, question, model = "gpt-3.5-turbo"):
    # parse args to comma separated string
    messages = [{"role": "system",
                "content": demo},
                {"role": "user",
                "content": f"Propositions: ```{context}```\nQuestion: ```{question}```"}]

    response = openai.ChatCompletion.create(
        model = model,
        messages = messages,
        temperature = 0
    )

    return response.choices[0].message["content"]

def ai_function_cot_part2(demo, context, model = "gpt-3.5-turbo"):
    # parse args to comma separated string
    messages = [{"role": "system",
                "content": demo},
                {"role": "user",
                "content": f"```{context}```"}]

    response = openai.ChatCompletion.create(
        model = model,
        messages = messages,
        temperature = 0
    )

    return response.choices[0].message["content"]



def remove_spaces(text):
    # Replace multiple spaces with a single space
    text = re.sub(r' +', ' ', text)
    # Remove leading and trailing spaces from each line
    text = re.sub(r'^ +| +$', '', text, flags=re.MULTILINE)
    return text

template = {
    "zero-shot-CoT-part1": remove_spaces("""Based on the closed world assumption, please help me complete this multi-step logical reasoning task. Answer whether this question is correct based on the propositions about facts and rules formed by these natural language propositions. \
                                            You should think through the question step by step, and show your full process. \n"""),
    "zero-shot-CoT-part2": remove_spaces("""Based on this thought process, please help me sum up only a number as the final answer (1 represents correct, 0 represents wrong).""")
}

openai.api_key = api_key = os.getenv("OPENAI_API_KEY")


def ZeroShotCoT_call1(demo, context, question, model = "gpt-3.5-turbo"):
    return ai_function_generation(demo, context, question, model)

def ZeroShotCoT_call2(demo, context, model = "gpt-3.5-turbo"):
    return ai_function_cot_part2(demo, context, model)

# List of json file names
json_files = [
    "../../PARARULE_plus_step2_Animal_sample.json"

]
# "../../PARARULE_plus_step3_Animal_sample.json",
# "../../PARARULE_plus_step4_Animal_sample.json",
# "../../PARARULE_plus_step5_Animal_sample.json",
# "../../PARARULE_plus_step2_People_sample.json",
# "../../PARARULE_plus_step3_People_sample.json",
# "../../PARARULE_plus_step4_People_sample.json",
# "../../PARARULE_plus_step5_People_sample.json"


# Open the CSV file for writing
with open("zeroshot-cot-3.5.csv", "w", newline="", encoding="utf-8") as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["step", "return", "label"])  # Write header

    for json_file in json_files:
        step = '_'.join(json_file.split("_")[2:4])
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            for entry in data:
                context = entry["context"]
                question = entry["question"]
                label = entry["label"]
                # Replace this with your actual function call
                response_part_1 = ZeroShotCoT_call1(template['zero-shot-CoT-part1'], context, question)
                print(response_part_1)
                response_part_2 = ZeroShotCoT_call2(template['zero-shot-CoT-part2'], response_part_1)
                print(response_part_2)
                csv_writer.writerow([step, response_part_2, label])