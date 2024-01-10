import os
import json
import pandas as pd
import numpy as np
import jsonlines
import re
import matplotlib.pyplot as plt
import csv
import glob
import zhipuai
        

    
def generate_gt(vitals,label='HR'):
    #label = 'HR' or 'SpO2' or 'BVP'
    #vitals = vitals_hr or vitals_spo2 or vitals_bvp
    #both return: health state, 0: normal, 1: abnormal, 2: extreme abnormal
    #HR：return average HR and the max HR
    #SpO2：return average SpO2 and the min SpO2
    #BVP：return average HR 
    health_state = 0
    if label=='HR':
        average = np.mean(vitals)
        max_v = np.max(vitals)
        if max_v>=100:
            health_state = 1
        if max_v>130:
            health_state = 2
        return health_state,average,max_v
    elif label=='SpO2':
        average = np.mean(vitals)
        min_v = np.min(vitals)
        if min_v<=95:
            health_state = 1
        if min_v<=92:
            health_state = 2
        return health_state,average,min_v
    elif label=='BVP':
        average = np.mean(vitals)
        if average>0.5:
            health_state = 1
        return health_state,average

        


prompt_HR ={'introduction':'Here is a list of Heart Rate data of myself. Each point refers to a second.','task':'Please tell me the  average heart rate. Keep one decimal place. And give me analysis of health. If the heart rate for anytime is under 100, health state is 0. If the  heart rate for anytime is 100-130, health state is 1. If the heart rate is above 130, health state is 2. Then tell me why you have such judgment on the reasons part. Please consider the trend of the vital in time series as well. Please output as the format required: The average heart rate : XXX. The health state : XXX. Suggestions: XXX. Resons: XXX.','background':'An ideal heart rate is between 50 and 90 beats per minute (bpm). It is acceptable to continue home monitoring for 101-109. If the heart rate is 110 to 130, you would better seek advice from your GP. If the heart rate is 140 or above, you should seek medical advice immediately.','output_format':'The average heart rate : XXX. The health state : XXX. Suggestions: XXX. Resons: XXX.'}
prompt_SpO2 ={'introduction':'Here is a list of Blood Oxygen(SpO2 value) data of myself. Each point refers to a second.','task':'Please tell me the average blood oxygen. Keep one decimal place. And give me analysis of health. If SpO2 for anytim is between 96% and 99%, health state is 0. If SpO2 for anytim is between 93% and 95%, health state is 1. If SpO2 for anytim is 92 or less, health state is 2. Then tell me why you have such judgment on the reasons part. Please consider the trend of the vital in time series as well. Please output as the format required: The average blood oxygen : XXX. The health state : XXX. Suggestions: XXX. Resons: XXX.','background':'A normal blood oxygen level varies between between 96% and 99% in healthy individuals.It is acceptable to continue home monitoring for 93%-95%. If the blood oxygen level is 92% or less, it is considered low and called hypoxemia,you would better seek advice from your GP. If the blood oxygen level is below 85%, it is considered critical and called hypoxia, you should seek medical advice immediately.','output_format':'The average blood oxygen level : XXX. The health state : XXX. Suggestions: XXX. Reasons: XXX.'}
prompt_All = {'introduction':'Here is a list of Heart Rate data and a list of Blood Oxygen(SpO2 value) data  of myself. Each point refers to a second.','task':'Please tell me the average blood oxygen and the average heart rate. Keep one decimal place. And give me analysis of health. Only two vitals is normal then the health is normal and output 0. Any abnormal of any vital should be considered abnormal and output 1. Then tell me why you have such judgment on the reasons part. Please consider the trend of the vital in time series as well. Please output as the format required: The average heart rate : XXX. The average blood oxygen : XXX. The health state : XXX. Suggestions: XXX. Resons: XXX.','background':'A normal blood oxygen level varies between between 96% and 99% in healthy individuals.It is acceptable to continue home monitoring for 93%-95%. If the blood oxygen level is 92% or less, it is considered low and called hypoxemia,you would better seek advice from your GP. If the blood oxygen level is below 85%, it is considered critical and called hypoxia, you should seek medical advice immediately. If SpO2 for anytim is between 96% and 99%, health state is 0. If SpO2 for anytim is between 93% and 95%, health state is 1. If SpO2 for anytim is 92 or less, health state is 2. An ideal heart rate is between 50 and 90 beats per minute (bpm). It is acceptable to continue home monitoring for 101-109. If the heart rate is 110 to 130, you would better seek advice from your GP. If the heart rate is 140 or above, you should seek medical advice immediately. If the  heart rate for anytime is under 100, health state is 0. If the  heart rate for anytime is 100-130, health state is 1. If the  heart rate for anytime is above 130, health state is 2. ','output_format':'The average heart rate : XXX. The average blood oxygen : XXX. The health state : XXX. Suggestions: XXX. Reasons: XXX.'}
# prompt_BVP = {'introduction':'Here is a list of Blood Volume Pulse data of myself. Each second refers to 20 points.','task':'Please tell me the average blood volume pulse of this subject. And give me analysis of health of the subject.'}



def glm_api(prompt_content):
    # pip install zhipuai 
    
    zhipuai.api_key = "your-api-key"
    response = zhipuai.model_api.sse_invoke(
        model="chatglm_pro",
        prompt=[
        {"role": "user", "content":prompt_content}],
        temperature=0.9,
        top_p=0.7,
        incremental=True
    )

    response_data = ""  # Create an empty string to store event data

    for event in response.events():
        if event.event == "add":
            response_data += event.data
        elif event.event == "error" or event.event == "interrupted":
            response_data += event.data
        elif event.event == "finish":
            response_data += event.data
        else:
            response_data += event.data

    return response_data 

def gpt_api(prompt_content):
    import openai
    openai.api_key = "your-api-key"
    openai.api_base = "your-api-link"

    # create a chat completion
    chat_completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt_content}])

    # print the chat completion
    res = (chat_completion.choices[0].message.content)
    return res
    
    

def gt_text(file_list, mode='All', chunk_len=60):
    # mode: 'All'-HR+SpO2, 'HR'-HR, 'SpO2'-SpO2
    # input: file name list
    # output: ground truth and text data to LLM
    print("mode:", mode)
    # Initialize empty lists to store ground truth and text data
    ground_truth_list = []
    text_data_list = []
    
    if mode == 'HR':
        for file in file_list:
            if file.endswith('HR.csv'):
                print('file name:', file)
                vitals = HR_dict[file]
                vitals_chunks = chunk_vitals(vitals, chunk_len)
                for chunk in vitals_chunks:
                    gt = generate_gt(chunk, 'HR')
                    text = str(prompt_HR) + '\nHR data: ' + str(chunk)
                    # print('ground truth:', gt)
                    # print('text:', text)
                    # print('---------------------------')
                    # Append ground truth and text to lists
                    ground_truth_list.append(gt)
                    text_data_list.append(text)
    
    elif mode == 'SpO2':
        for file in file_list:
            if file.endswith('SpO2.csv'):
                print('file name:', file)
                vitals = SpO2_dict[file]
                vitals_chunks = chunk_vitals(vitals, chunk_len)
                for chunk in vitals_chunks:
                    gt = generate_gt(chunk, 'SpO2')
                    text = str(prompt_SpO2) + '\nSpO2 data: ' + str(chunk)
                    # print('ground truth:', gt)
                    # print('text:', text)
                    # print('---------------------------')
                    # Append ground truth and text to lists
                    ground_truth_list.append(gt)
                    text_data_list.append(text)
    
    elif mode == 'All':
        for file in file_list:
            if file.endswith('HR.csv'):
                file1 = file
                file2 = file[:-6] + 'SpO2.csv'
                print('file name:', file1, file2)
                vitals1 = HR_dict[file1]
                vitals2 = SpO2_dict[file2]
                vitals_chunks1 = chunk_vitals(vitals1, chunk_len)
                vitals_chunks2 = chunk_vitals(vitals2, chunk_len)
                
                for chunk1, chunk2 in zip(vitals_chunks1, vitals_chunks2):
                    gt1 = generate_gt(chunk1, 'HR')
                    gt2 = generate_gt(chunk2, 'SpO2')
                    gt = 'HR: ' + str(gt1) + '\n SpO2: ' + str(gt2)
                    text = str(prompt_All) + '\n HR data: ' + str(chunk1) + '\n SpO2 data: ' + str(chunk2)
                    # print('ground truth:', gt)
                    # print('text:', text)
                    # print('---------------------------')
                    # Append ground truth and text to lists
                    ground_truth_list.append(gt)
                    text_data_list.append(text)
    
    # Save ground truth to a CSV file (you need to import the appropriate library for this)
    # Example using the 'csv' module:
    # with open('ground_truth.csv', 'w', newline='') as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerow(['Ground Truth'])
    #     writer.writerows(ground_truth_list)
    
    # Return the list of text data to LLM
    return ground_truth_list, text_data_list, mode



def extract_and_save_to_csv(origin_input, text, gt, csv_filename):
    
    pattern = r'The average blood oxygen level : (\d+\.\d+). The health state : (\d+). Suggestions: (.+). Reasons: (.+)'

    
    match = re.search(pattern, text)

    if match:
        
        average_blood_oxygen = match.group(1)
        health_state = match.group(2)
        suggestions = match.group(3)
        reasons = match.group(4)

        
        data = {
            "orginal_input": origin_input,
            "Average Blood Oxygen Level": average_blood_oxygen,
            "Health State": health_state,
            "Suggestions": suggestions,
            "Reasons": reasons,
            "ground_truth":gt
        }

        
        try:
            with open(csv_filename, mode='a', newline='') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=data.keys())
                writer.writerow(data)
        except FileNotFoundError:
            with open(csv_filename, mode='w', newline='') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=data.keys())
                writer.writeheader()
                writer.writerow(data)
    else:
        print("No match found in the text.")



if __name__ == "__main__":
    vital_path = 'demo_data'
    #load all BVP,HR,SpO2
        
    def find_csv_files(directory):
        csv_files = []
        
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('BVP.csv') or file.endswith('HR.csv') or file.endswith('SpO2.csv'):
                    csv_files.append(os.path.join(root, file))
        
        return csv_files


    vital_csv_files = find_csv_files(vital_path)
    vital_csv_files.sort()
    vital_csv_files
    HR_dict = {}
    SpO2_dict = {}
    BVP_dict = {}
    csv_files = vital_csv_files
    for file in csv_files:
        #spilt the file name by BVP,HR,SpO2
        if file.endswith('HR.csv'):
            data = pd.read_csv(file)['hr']
            #save the data in int num format then to the dictionary,split with ,
            data_text = list(map(int, data))
            HR_dict[file] = data_text
        elif file.endswith('SpO2.csv'):
            data = pd.read_csv(file)['spo2']
            data_text = list(map(int, data))
            SpO2_dict[file] = data_text
        elif file.endswith('BVP.csv'):
            data = pd.read_csv(file)['bvp']
            data_text = list(map(int, data))
            BVP_dict[file] = data_text
            
            
    #chunk the vitals 
    #  30 60 120
    def chunk_vitals(vitals,length=60):
        vital_list = []
        for i in range(0, len(vitals), length):
            if i+length > len(vitals):
                break
                vital_list.append(vitals[i:])
            else:
                vital_list.append(vitals[i:i+60])
        return vital_list

    print(SpO2_dict)
    vitals_test =SpO2_dict['demo_data/light_1/v01/SpO2.csv']
    vitals_chunks = chunk_vitals(vitals_test)
    for vital in vitals_chunks:
        print(vital)
        print(len(vital))
            
    gt_list, prompt_list ,mode = gt_text(csv_files,'SpO2',60)
    for prompt,gt in zip(prompt_list,gt_list):
        print('\n\nprompt:',prompt)
        glm_ans = glm_api(prompt_content=str(prompt))
        gpt_ans = gpt_api(prompt_content=str(prompt))

        print('\n\nGLM_Ans:',glm_ans)
        print('\n\nGBT_Ans:',gpt_ans)
        extract_and_save_to_csv(origin_input = prompt, text= glm_ans,gt = gt, csv_filename= "/share/HealthLLM/glm_res.csv")
        extract_and_save_to_csv(origin_input = prompt, text= gpt_ans,gt = gt, csv_filename= "/share/HealthLLM/gpt_res.csv")
        
        
        