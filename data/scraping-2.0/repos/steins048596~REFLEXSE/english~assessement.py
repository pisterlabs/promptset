# -*- coding:utf-8 -*-
import os
import openai
import csv
from collections import Counter
import time
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from heapq import nlargest
import simplejson as json

import torch
import torch.nn as nn
from transformers import AutoTokenizer, LongformerModel
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Optional
import os
from apdia import dialogue_process
import csv
import json
os.environ['CUDA_VISIBLE_DEVICES'] = '3'


if torch.cuda.is_available():
    dev = torch.device("cuda:0") 
    print("Running on the GPU")
else:
    dev = torch.device("cpu")
    print("Running on the CPU")

tokenizer =AutoTokenizer.from_pretrained('./longformer_clinical')
tokenizer.pad_token = tokenizer.eos_token
test_data=[]




test_data=[]
def read_data(typename,file_path_chosen,file_path_reject,file_path_dia):   
    index=[]
    chosen=[]
    reject=[]
    with open(file_path_reject, 'r', encoding='utf8') as f:
        data = json.load(f)
    with open(file_path_chosen, 'r', encoding='utf8') as f:
        data2 = json.load(f)
    for i in range(len(data['data'])):
        if 'PLAN' not in data['data'][i]["asessment_and_plan"]["ASSESSMENT AND PLAN"] and 'PLAN' not in data2['data'][i]["asessment_and_plan"]["ASSESSMENT AND PLAN"]:
            index.append(data['data'][i]['encounter_id'])
    for item in data['data']:
        if item['encounter_id'] not in index:
            continue
        else:
            reject.append(item["asessment_and_plan"]["ASSESSMENT AND PLAN"])
#     print(index[0])
    for item in data2['data']:
        if item['file'][:item['file'].rfind('-')].replace('-','') in index:
            chosen.append(item["asessment_and_plan"]["ASSESSMENT AND PLAN"])
        else:               
            continue
    dialogue=dialogue_process(typename,index,reject,file_path_dia,tokenizer,300,550,1300)
    print(len(index))
    for i in range(len(index)):
        temp={'index':index[i],'prompt':dialogue[i],'chosen':chosen[i],'rejected':reject[i]}
        if typename=='test':
            test_data.append(temp)
read_data('test','data/assessment/test.json','data/assessment/test_ini.json','data/assessment/test_apdia_zhen.csv')


class LEDRM(nn.Module):
    """
    BLOOM Reward model.

    Args:
        pretrained (str): Pretrained model name or path.
        config (BloomConfig): Model config.
        checkpoint (bool): Enable gradient checkpointing.
    """

    def __init__(self,
                 pretrained: str = None,
                 checkpoint: bool = False) -> None:
        super().__init__()
        self.model =LongformerModel.from_pretrained('./longformer_clinical')
        self.value_head = nn.Linear(self.model.config.hidden_size, 1, 1)
        self.value_head.weight.data.normal_(mean=0.0, std=1 / (self.model.config.hidden_size + 1))
#        print(self.model.config)
    def forward(self, sequences: torch.LongTensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        outputs = self.model(sequences, attention_mask=attention_mask)
        last_hidden_states = outputs['last_hidden_state']
        values = self.value_head(last_hidden_states)[:, :-1]
        value =values.mean(dim=1).squeeze(1)    # ensure shape is (B)
        return value

model = LEDRM().to(dev)
load_layer_state_dict = torch.load("littlemodel/jianyi_real_xray_real.pdparams")
model.load_state_dict(load_layer_state_dict)


def modelinput(prompt,chosen):
    model.eval()
    chosen = prompt + chosen + tokenizer.eos_token
    
    chosen_token = tokenizer(chosen,
                     max_length=1700,
                     padding="max_length",
                     truncation=True,
                     return_tensors="pt")
    chosen_ids = chosen_token['input_ids'].squeeze(1).to(dev)
    c_mask = chosen_token['attention_mask'].squeeze(1).to(dev)
    results = model(chosen_ids,c_mask).item()
    return results

def topn_dict(d, n):
    return nlargest(n, d, key=lambda k: d[k])

def save_csv(total,i,rname):
    with open(rname, 'a', newline='') as file:
        writer = csv.writer(file)
        if i==0:
            writer.writerow(["id", "text",])
        writer.writerow([test_data[i]['index'],total])
rname='results/asse.csv'

#rname3='hpi_test_record.csv'
print("a")

data_output=[]
present_total=[]
text_total=[]

    
temp=[]

@retry(wait=wait_random_exponential(min=1, max=30), stop=stop_after_attempt(2))
def completion_with_backoff(prompt1,prompt2,prompt3,icl1,icl2,input0,temperature):
    result=openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a professional doctor, please help me point out the ERROR in the following medical records and REVISE."},
                    {"role": "user", "content": prompt1+icl1+icl2+prompt2+prompt3+input0}
                    # {"role": "assistant", "content": "https://www.cnblogs.com/botai/"},
                ],
                temperature=temperature,
                max_tokens=800
            )
    return result
#i=121
i=0
k=0
t=0
w=0
xr=0
tt=0
temptext=[]
black_list=[]
temptext.append([])
dict_temp={}
count0=0
flag=0
tempmax=''
flag_text=0

while(i<=len(test_data)):
    if flag_text==0:
        text00=str(test_data[i]['rejected'])
    temperature=0.1
    if flag_text==0:
        temptext[-1].append(text00)
        dict_temp.setdefault(text00)
        dict_temp[text00]=modelinput(str(test_data[i]['prompt']),str(test_data[i]['rejected']))
    icl2='''EXAMPLE NOTE：
1. Uncontrolled Hypertension.
- Medical Reasoning: Multifactorial cause, possibly related to lifestyle and diet. Additional investigations needed to rule out renal artery or adrenal gland issues.
- Patient Education and Counseling: Discussed lifestyle modifications, including reducing alcohol and salt intake, and referred to a nutritionist.
- Medical Treatment: Ordered renal artery ultrasound, urine collection, morning aldosterone and renin levels, and 24-hour urine. Prescribed Cardura 4 mg once daily. Advised the cessation of NSAIDs.
- Follow-Up: Return in 3 weeks with blood pressure records.
Patient Agreements: The patient understands and agrees with the recommended medical treatment plan.

EXAMPLE ERROR:
Follow-up should be documented in Patient Education and Counseling. And the NOTE missed the Counseling that she should stop taking anti-inflammatories and use Tylenol as needed for pain.

REVISE: 
1. Hypertension, uncontrolled.
- Medical Reasoning: The patient's elevated blood pressure is consistent with uncontrolled hypertension.
- Patient Education and Counseling: We discussed the nature of the diagnosis and that this is typically multifactorial. She was encouraged to reduce her intake of alcohol as well as her salt intake. I recommended that she stop taking anti-inflammatories and use Tylenol as needed for pain. We also discussed the importance of home blood pressure monitoring of the next 3 weeks to see if the medication is beneficial.
- Medical Treatment: Renal artery ultrasound ordered. Urine collection, morning aldosterone levels, renal levels, and a 24-hour urine were also ordered. Referral to nutritionist provided. Prescription for Cardura 4 mg once a day provided as well.
Patient Agreements: The patient understands and agrees with the recommended medical treatment plan.

'''
    icl1="EXAMPLE NOTE：\nMr. Bryan is a 55-year-old male with a past medical history significant for prior discectomy, who presents today with low back pain.\nLumbar strain.\n Medical Reasoning: He injured his lower back while moving a refrigerator. His recent x-ray was unremarkable.\n Medical Treatment: Order MRI to be sure of the diagnosis, initiate meloxicam 15 mg once a day and ultram 50 mg every four hours as needed.\n Specialist Referrals: Referral to physical therapy once MRI results are back.\n Patient Education and Counseling: He was advised to stop ibuprofen and can continue Tylenol if desired. The patient understands the treatment plan.\nPatient Agreements: The patient understands and agrees with the recommended medical treatment plan.\n\nEXAMPLE ERROR:\nCompared to the DIALOGUE, in the NOTE, there are many inaccurate descriptions,such as the NOTE did not describe '5 days ago' and 'right-sided' low back,and also missed that the patient's symptom is related to his previous discectomy. Ordering a MRI should be documented in 'Additional Testing'.\n\nREVISE:\nBryan is a 55-year-old male with a past medical history significant for prior discectomy, who presents with back pain.\nLumbar strain.\n  Medical Reasoning: He reports right-sided low back after moving a refrigerator approximately 5 days ago. X-ray of his lumbar spine is unremarkable. I do not believe this is related to his previous discectomy.\n  Additional Testing: We will order a MRI of the lumbar spine for further evaluation.\n  Medical Treatment: Initiate meloxicam 15 mg once daily, as well as Ultram 50 mg every 4 hours as needed.\n  Specialist Referrals: We will refer him to physical therapy to be started after we get his MRI results back.\n  Patient Education and Counseling: I advised the patient to discontinue the use of ibuprofen, but he may continue using Tylenol if he wishes.\nPatient Agreements: The patient understands and agrees with the recommended medical treatment plan.\n\n"
    prompt1="GUIDE:\nThe assessment and plan should include:\nPatient specific differential diagnosis and discussion that is NOT a summary or list from textbook or UPTODATE.\nA commitment to a leading diagnosis.\nA weighted list of 3-5 active alternative diagnoses with emphasis on the diagnostic imperative. (a diagnosis that because of its probability/morbidity/treatability must not be missed).\nSupport for the diagnostic possibilities that attempts to explicitly link the diagnostic possibilities to the patient’s clinical findings and integrates pathophysiology where appropriate; compare and contrast the diagnostic possibilities.\nTo evaluate the problems.\nTo treat the problems.\nTo provide patient education.\nIt is necessary to comprehensively consider  RESULT(the results of x-ray and so on), ASSESSMENT.\n\n"
    prompt2="----------------------------------------------------------------------------------------------\nDIALOGUE：\n"+str(test_data[i]['prompt'])+'\n'
    prompt3="Now, recorrect the NOTE with the DIALOGUE below and use the EXAMPLE above. In REVISE, DO NOT delete the information in the NOTE. In REVISE, information that is not in the NOTE or DIALOGUE is PROHIBITED.  Compare the NOTE to the DIALOGUE, and point the error in ERROR, and correct them in REVISE. Patient Agreements Please be brief.\n\n"
    input0="NOTE：\n"+text00+"\n\nERROR:\n"
    response = completion_with_backoff(prompt1,prompt2,prompt3,icl2,icl1,input0,temperature)

    text0=str(response["choices"][0]["message"]["content"])
    flag_text=1

    if "REVISE:" not in text0:
        print('bbbb')
        if "no error" not in text0:
            t=t+1
            continue
        else:
            flag_text=2
#     w=0
    if "REVISE:" in text0:
        text=text0[text0.rfind('REVISE:'):][:text0.rfind('。')].replace('REVISE:','')
        index=0
        while(text[index]=='\n'):
            index=index+1
        text=text[index:]           
        if "Note:" in text:
            text=text[:text.rfind('Note:')].replace('Note:','')
    
    elif flag_text==2:
        text=text00
        flag_text=1
    print(temptext[-1])
    print('小模型输入',text)
    if 'Bryan' in text00:
        ff=1
    else:
        if 'Bryan' in text:
            print('cccc')
            t=t+1
            continue
    if text in black_list:
        flag=flag^1
        continue
    if text not in dict_temp.keys() and text not in black_list:
        dict_temp.setdefault(text) 
        dict_temp[text]=modelinput(str(test_data[i]['prompt']),text)
    
    temptext[-1].append(text)
    print('结果',text)
    temp.append(response["choices"][0]["message"]["content"])   
    number = Counter(temptext[-1])
    result = number.most_common()
    print("result01",result[0][1])
    print("result00",result[0][0])
    tempmax=result[0][0]
    text00=text
    print(dict_temp)
    print(temptext[-1])
    if len(temptext[-1])>=14:
        if temptext[-1].count(max(dict_temp, key=lambda x: dict_temp[x]))>=2 or max(dict_temp, key=lambda x: dict_temp[x])==temptext[-1][-1]:
            save_csv(max(dict_temp, key=lambda x: dict_temp[x]),i,rname)
            data_output.append(max(dict_temp, key=lambda x: dict_temp[x]))
            dict_temp={}
            count0=0
            flag=0
            flag_text=0
            temptext.append([])
            i=i+1
            xr=0
            t=0
            tempmax=''
            continue
        else:
            save_csv(result[0][0],i,rname)
            data_output.append(result[0][0])
            dict_temp={}
            count0=0
            flag=0
            flag_text=0
            temptext.append([])
            i=i+1
            xr=0
            t=0
            tempmax=''
            continue

            
            
            
    if "no error" in text0:
        res=topn_dict(dict_temp,2)
        print('最大值',res)
        if result[0][0] not in res:
            flag=flag^1
            black_list.append(result[0][0])
            while result[0][0] in temptext[-1]:
                temptext[-1].remove(result[0][0])
            text00=max(dict_temp, key=lambda x: dict_temp[x])
            continue
        else:
            save_csv(result[0][0],i,rname)
            data_output.append(result[0][0])
            dict_temp={}
            count0=0
            flag=0
            flag_text=0
            temptext.append([])            
            black_list=[]
            i=i+1
            xr=0
            tempmax=''
            t=0
            continue



    elif result[0][1]>=2:
        res=topn_dict(dict_temp,2)
        print('最大值',res[0])
        if len(dict_temp)<3:
            flag=flag^1
            continue
        if result[0][0] not in res:
            flag=flag^1
            black_list.append(result[0][0])
            while result[0][0] in temptext[-1]:
                temptext[-1].remove(result[0][0])
            text00=max(dict_temp, key=lambda x: dict_temp[x])
            continue
        else:
            save_csv(result[0][0],i,rname)
            data_output.append(result[0][0])
            dict_temp={}
            count0=0
            flag=0
            flag_text=0
            black_list=[]
            temptext.append([])            
            i=i+1
            xr=0
            tempmax=''
            t=0
            continue

