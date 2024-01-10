import pandas as pd
import openai
import requests
from openpyxl.reader.excel import load_workbook
import time

from api_key import *

headers = {
    'Content-Type': 'application/json',
    'Authorization': f"Bearer {api_key}"
}


def gpt(role, question):
    """
    :param role: <str> - role gpt takes on
    :param question: <str> - question you ask gpt
    :return: <str> - gpt response
    """
    data = {
        'messages': [
            {'role': 'system', 'content': role},
            {'role': 'user', 'content': question}
        ],
        'model': 'gpt-3.5-turbo', # gpt 4
        "temperature": 0.1
    }

    api_url = 'https://api.openai.com/v1/chat/completions'
    response = requests.post(api_url, headers=headers, json=data)

    if response.status_code == 200:
        response_data = response.json()
        reply = response_data['choices'][0]['message']['content']
        return reply
    else:
        message = 'Error:', response.status_code, response.text
        return message


def excel_generator(path, column_name):
    """
    Generator to yield rows one by one from an Excel file
    """
    df = pd.read_excel(path)
    for value in df[column_name]:
        yield value


file_path = "/Users/zihaowang/PycharmProjects/Takachar/Takachar-GPT/Hot_Tests.xlsx"
excel_gen = excel_generator(file_path, "Hot Test Text")

role = "You are a scientist with expertise in chemical engineering, physical experiments, and data science."

context = "These experiments are Combustion (Pyrolysis) experiments in presence of oxygen for biomass to biochar formation in a newly designed small-sized reactor." \
         "A hopper is used for inputting biomass, an auger rotates at a certain speed to carry the biomass in the reactor where it is burnt." \
         "Volatiles go from chimney above, and output biochar goes from other end of the auger." \
         "Thermocouples measure the temperature at different places. " \
         "Primary blowers provide air (hence oxygen) to reactor zone and secondary blowers remove volatiles through chimney. " \
         "The objective is to gain a complete understanding of the underlying combustion process and machine design by testing different hypotheses."

question = "Example Hot Test: The experiment had an input material of rice. Example Answer: rice" \
           "Example Hot Test: The experiment doesn't mention the input material. Example Answer: N/A" \
           "Example Hot Test: "

answers = []
counter = 0
for hot_test in excel_gen:
    prompt = f"{context} + {question} + {hot_test}. Example Answer: "
    gpt_response = gpt(role, prompt)
    answers.append(gpt_response)
    print(counter)
    if counter % 2 == 0:
        time.sleep(60)
    print(gpt_response)
    counter += 1
print("done with GPT")
print(answers)
workbook = load_workbook('/Users/zihaowang/PycharmProjects/Takachar/Takachar-GPT/Hot_Tests.xlsx')
worksheet = workbook['Sheet1']

for i in range(len(answers)):
    try:
        cell = f"O{i+2}"
        worksheet[cell] = answers[i]
    except:
        cell = f"O{i + 2}"
        worksheet[cell] = "N/A"

workbook.save('/Users/zihaowang/PycharmProjects/Takachar/Takachar-GPT/Hot_Tests.xlsx')
print("done with everything")

