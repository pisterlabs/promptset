# -*- coding:utf-8 -*-
import paddle
import os
#os.environ['http_proxy']="http://127.0.0.1:7892"
#os.environ['https_proxy']="http://127.0.0.1:7892"
paddle.device.set_device('gpu:3')
import openai
openai.api_key = "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"
import json
import csv
from collections import Counter
import time
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from heapq import nlargest
from paddlenlp.datasets import load_dataset
import simplejson as json

import os
import paddlenlp
import numpy as np
import paddle.optimizer as opt
from paddlenlp.transformers import ErnieModel, ErnieTokenizer
import paddle.nn.functional as F
from paddle import nn
from paddlenlp.transformers import ErniePretrainedModel
from paddlenlp.transformers import AutoModel, AutoTokenizer
class ValueErnie(ErniePretrainedModel):
    def __init__(self, config,dropout=None):
        super(ValueErnie, self).__init__(config)        
        self.ernie = ErnieModel(config)
        self.value_head = nn.Linear(self.ernie.config['hidden_size'], 1)
        self.dropout=nn.Dropout(dropout if dropout is not None else
                                  self.ernie.config["hidden_dropout_prob"])

    def forward(self, chosen_ids, chosen_mask):
        chosen_ids=chosen_ids.squeeze(1)
        chosen = self.ernie(chosen_ids,attention_mask=chosen_mask)
        chosen_output = self.dropout(chosen[0])
        chosen_logits = self.value_head(chosen_output)
        chosen_reward=chosen_logits.mean(axis=1).squeeze(1)
        return chosen_reward
    
    
MODEL_NAME2='ernie-health-chinese'
tokenizer = paddlenlp.transformers.AutoTokenizer.from_pretrained(MODEL_NAME2)
model = ValueErnie.from_pretrained('little_model/real_model/ernie-health-chinese/',dropout=0.1)
load_layer_state_dict = paddle.load("little_model/real_model/jianyi.pdparams")
model.set_state_dict(load_layer_state_dict)


def modelinput(text):
    model.eval()
    input_ids = tokenizer(text=text, max_seq_len=512)['input_ids']
    input_ids=paddle.to_tensor(input_ids, dtype='int64')
    input_ids=paddle.unsqueeze(input_ids,axis=0)
    input_mask = np.ones(len(input_ids[0]),dtype='int64')
    input_mask=paddle.to_tensor(input_mask, dtype='int64')
    results = model(input_ids,input_mask).item()
    return results

def topn_dict(d, n):
    return nlargest(n, d, key=lambda k: d[k])

def save_csv(total,i,rname):
    with open(rname, 'a', newline='') as file:
        writer = csv.writer(file)
        if i==0:
            writer.writerow(["id", "text",])
        writer.writerow(["dev_"+str(i),total])
rname='new_experiment/sug_chatgpt_turbo_fewshot.csv'
print("a")

data_output=[]
present_total=[]
text_total=[]
filename='data/test_index.csv'
#filename='../new_experiment/16ktest/index_test.csv'
with open(filename,encoding='gbk') as f:
        i=0    
        f_csv=csv.reader(f)
        headers = next(f_csv)
        for row in f_csv:
            text_total.append(str(row[19]))
            i=i+1
temp=[]
@retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(10))
def completion_with_backoff(prompt1,prompt2,prompt3,icl1,icl2,input0,temperature):
    result=openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "你是一位专业医生，请根据具体的规则和要求修改电子病历,生成ErrorPrompt和reference"},
                    {"role": "user", "content": prompt1+icl1+icl2+prompt2+prompt3+input0}
                    # {"role": "assistant", "content": "https://www.cnblogs.com/botai/"},
                ],
                temperature=temperature
            )
#    print(prompt1+icl1+icl2+prompt2+prompt3+input0)
    return result
i=0
k=0
xr=0
temptext=[]
black_list=[]
temptext.append([])
dict_temp={}
count0=0
flag=0
t=0
flag_text=0
#while(i<=127):
#while(i<=316):
#list1=[53,80, 82, 83, 93, 100, 115, 129, 135, 139, 140, 142, 143, 151, 154, 161, 165, 170]
list1=[53,80,82,83, 100, 115, 129, 135, 139, 140, 151, 154, 165, 170]
#list1=[151, 154, 161, 165, 170]
#while(i<=5):
while(i<=len(text_total)):
    print("I:",i)
#     if i not in list1:
#         i=i+1
#         continue
# 设置 API 密钥
    if flag_text==0:
        text00=str(text_total[i]).replace('(6)建议：','')
    temperature=0.2
    print(text00)
#    if text00.count('。')>=3：
        
    if flag_text==0:
        temptext[-1].append(text00)
        dict_temp.setdefault(text00)
        dict_temp[text00]=modelinput(str(text_total[i]).replace('(6)建议：',''))
#    print(present)
    icl1="例子1建议：建议进行肺部听诊，检查血常规和支原体检测以明确病因。目前继续口服蒲地蓝消炎口服液和止咳药治疗，勤喝水，注意休息和保持室内空气流通，如咳嗽仍然频繁，建议及时就医。饮食应以清淡易消化为主，避免过多过杂食物，无需特别忌口。\n\nErrorPrompt：语言风格要精练，尽量使用小短句，只写2-4条“指南”内的建议\n\nreference：肺部听诊，支原体，血常规，勤喝水。继续当前药物治疗。密切观察，必要时及时就医。\n\n"
    icl2="例子2建议：保暖，多喝水。如有其他问题，随时留言。\n\nErrorPrompt：建议内容逐条参考指南，不要记录指南以外的信息。\n\nreference：注意保暖，多喝水。\n\n"
    prompt1="指南：建议可包含如下内容：1、进一步的“检查措施”、“所用药品”的名称、“剂量用法”、疗程等。2、向患者交代的注意事项：包括生活、饮食注意事项，休息方式与期限，复诊时间，随访要求。3、尽量减少口语化,尽量使用小短句，只写2-4条重要建议。\n参考上述要求\nErrorPrompt一定要简练。\n"
#    if len(str(text_total[i]).replace('(1)主诉：',''))>10:
    prompt2="“提示”："+str(text_total[i]).replace('(6)建议：','')+'\n'
#    else:
#        prompt2="“提示”："+present+'\n'
    prompt3="参考指南，根据下方“建议”生成ErrorPrompt与reference，并对比“提示”和下方“建议”，“建议”中禁止生成“提示”以外的信息\n\n"
   # 调用 ChatGPT
    if flag==0:
        input0="建议："+text00+"\n\nErrorPrompt："
    else:
        input0="建议："+text00+"\n\n"
    if flag==0:
        response = completion_with_backoff(prompt1,prompt2,prompt3,icl1,icl2,input0,temperature)
    if flag==1:
        response = completion_with_backoff(prompt1+'\n',prompt2,prompt3,icl2,icl1,input0,temperature)
    k=0
    text0=str(response["choices"][0]["message"]["content"])
    print(text0)
    flag_text=1
    if "ErrorPrompt：" not in text0 and "ErrorPrompt:" not in text0 and "ErrorPrompt：" not in input0:
        print('a')
#         t=t+1
#         if t>2:
        flag=flag^1
#             t=0
        continue
    if "reference" not in text0 and "改为" not in text0:
        if "无错误" not in text0 and "无需修改" not in text0 and "无误" not in text0:
            flag=flag^1
            t=t+1
            continue
        else:
            flag_text=2
    if '肺部听诊，支原体，血常规' in text0:
        flag=flag^1
        continue
    if '观察病情，如有其他问题可随时联系或留言' in text0:
        flag=flag^1
        continue
    if '注意保暖，多喝水' in text0:
        flag=flag^1
        continue
    if '口服蒲地蓝消炎口服液和止咳药治疗，勤喝水' in text0:
        flag=flag^1
        continue
    if "reference" in text0:
        t=0
        text=text0[text0.rfind('reference'):].replace(' ','')
        if "描述" in text:
            print("b")
            text00=max(dict_temp, key=lambda x: dict_temp[x])
            flag=flag^1
            continue
        if '。' in text:
            text=text[:text.rfind('。')].replace('reference：','').replace('reference:','').replace('。','')
        else:
            text=text.replace('reference：','').replace('reference:','').replace('reference:\n','').replace('reference：\n','').replace('。','')
        if '\n' in text:
            if text.split('\n')[0]!='':
                text=text.split('\n')[0]
            else:
                text=text.replace('\n','')
        if "ErrorPrompt：" in text:
            text=text[:text.rfind('ErrorPrompt：')].replace('ErrorPrompt：','')
        if "提示：" in text:
            text=text[:text.rfind('提示：')].replace('提示：','')
    elif "改为" in text0:
        t=0
        text=text0[text0.rfind('改为'):][:text0.rfind('\n')].replace('“','').replace('\n','').replace('改为','').replace('”','')
        if "ErrorPrompt：" in text:
            text=text[:text.rfind('ErrorPrompt：')].replace('ErrorPrompt：','')
        if "提示：" in text:
            text=text[:text.rfind('提示：')].replace('提示：','')
    elif flag_text==2:
        text=text00
        flag_text=1
#    print(temptext[-1])
    if text==text00:
        flag=1
    if len(text)<len(text00)*0.35:
        flag=flag^1
        continue
    print(temptext[-1])
    print('小模型输入',text)
    if text in black_list:
        flag=flag^1
        continue
    if text not in temptext[-1] and text not in black_list:
        dict_temp.setdefault(text) 
        dict_temp[text]=modelinput(text)
    temptext[-1].append(text)
    print('结果',text)
    temp.append(response["choices"][0]["message"]["content"])   
    number = Counter(temptext[-1])
    result = number.most_common()
    print("result01",result[0][1])
    print("result00",result[0][0])
    text00=text
    print(dict_temp)
    print(temptext[-1])
    if len(temptext[-1])>=16:
        if temptext[-1].count(max(dict_temp, key=lambda x: dict_temp[x]))>=2:
            save_csv(max(dict_temp, key=lambda x: dict_temp[x]),i,rname)
            data_output.append(max(dict_temp, key=lambda x: dict_temp[x]))
            dict_temp={}
            count0=0
            flag=0
            t=0
            flag_text=0
            temptext.append([])
            xr=0
            i=i+1        
            continue
        else:
            save_csv(result[0][0],i,rname)
            data_output.append(result[0][0])
            dict_temp={}
            count0=0
            t=0
            flag=0
            flag_text=0
            temptext.append([])
            xr=0
            i=i+1            
            continue
    if "无错误" in text0 or "无需修改" in text0 or "无误" in text0:
        if temptext[-1].count(max(dict_temp, key=lambda x: dict_temp[x]))<5:
            if len(dict_temp)<2:
                flag=flag^1
                continue
        res=topn_dict(dict_temp,2)
        print('最大值',res)
        if result[0][0] not in res:
            flag=flag^1
            black_list.append(result[0][0])
            if result[0][0] in dict_temp:
                dict_temp.pop(result[0][0])
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
            t=0
            continue
            
    elif result[0][1]>2:
        res=topn_dict(dict_temp,2)
        print('最大值',res[0])
        if result[0][0] not in res:
            flag=flag^1
            black_list.append(result[0][0])
            if result[0][0] in dict_temp:
                dict_temp.pop(result[0][0])
            while text in temptext[-1]:
                temptext[-1].remove(text)
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
            t=0
            continue
