# Note: you need to be using OpenAI Python v0.27.0 for the code below to work
import openai
import datasets
from datasets import load_dataset,load_from_disk
import json
import os
from tqdm import tqdm
from multiprocessing import Pool
from random import randint
openai.api_key="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"

def call_gpt(row):
  # print(row)
  context = row["context"]
  candidate = row["candidate"]
  label = row["label"]
  try:
    response = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=[
        {"role": "system", "content": "You are a retrieval-based dialogue system."},
        {"role": "user", "content": f'There is a dialog context:{context}.This is the candidate responses:{candidate}.Please choose the most appropriate response and output its corresponding serial number.'},
    ])
    gpt_gen=response["choices"][0]["message"]["content"]
  except:
    gpt_gen="null"
  data = {"gpt_gen":gpt_gen,"label":label}
  data = json.dumps(data,ensure_ascii=False)
  data = data+"\n"
  return data

def make_data(path):
    with open(path,"r") as f:
        data = f.readlines()
    groups = []
    candidates = []
    candidates_labels = []
    json_list = []
    for index,row in enumerate(data):
        row_split = row.split("\t")
        groups.append(row_split)
        label,string = row_split[0], row_split[1:]
        label = int(label)
        context, response = string[0:-1], string[-1]
        if index%10==0:
            context_string = ""
            for j,c in enumerate(context):
                role ="A:" if j%2==0 else "B:"
                context_string+=f"{role}{c} "
        candidates_labels.append(label)
        candidates.append(response)
        if (index+1)%10 == 0:
            t_index = candidates_labels.pop(1)
            g_index = randint(0, 9)
            swap = candidates[g_index]
            candidates[g_index]=candidates[t_index]
            candidates[t_index] = swap
            candidates_string = ""
            for j,c in enumerate(candidates):
                c= c.strip()
                candidates_string+=f"({j}):{c} "
            # print("a")
            candidates_labels,candidates = [],[]
            proc_data = {"context":context_string,"candidate":candidates_string,"label":g_index}
            json_list.append(json.dumps(proc_data)+"\n")
    save_path = "/mmu_nlp/wuxing/suzhenpeng/HC3/ubuntu/save.txt"
    with open(save_path,"w") as f:
        f.writelines(json_list)
                
def main(path):

    save="gpt.json"
    ours_data = []
    with open(path,"r") as f:
        data = f.readlines()
        for i in data:
            ours_data.append(json.loads(i))

    # ours_data=ours_data[0:16]
    with Pool(processes=64) as p:
        proc_data = p.map(call_gpt, tqdm(ours_data))
        with open(save,"w") as f:
            f.writelines(proc_data)

    # all_proc_data = []
    # for i in tqdm(ours_data):
    #     proc_data = call_gpt(i)
    #     all_proc_data.append(proc_data)
    # with open(save,"w") as f:
    #     f.writelines(all_proc_data)


def is_true(path):
    corrrect = 0
    with open(path,"r") as f:
        data  = f.readlines()
        for i in data:
            i=json.loads(i)
            label = i["label"]
            gpt = i["gpt_gen"]
            p="("+str(label)+")"
            if p in gpt:
                corrrect+=1
    print(corrrect/2000)
if __name__ == "__main__":
    # path = "/mmu_nlp/wuxing/suzhenpeng/HC3/ubuntu/sub_test.txt"
    # make_data(path)

    # path = "/mmu_nlp/wuxing/suzhenpeng/HC3/ubuntu/save.txt"
    # main(path)

    path="ubuntu/gpt.json"
    is_true(path)