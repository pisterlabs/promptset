# This is for SCOTUS dataset on GPT3.5-Turbo
from datasets import load_dataset
import openai
import os
from dotenv import load_dotenv
import time
import re
from sklearn.metrics import f1_score
import math

# Dataset loading and API
load_dotenv()
api_key = os.getenv('NEXT_PUBLIC_OPENAI_API_KEY')
openai.api_key = api_key
dataset = load_dataset('coastalcph/fairlex', 'scotus', split='train')

# Example data
text = dataset[0]['text']

decision_dir = {"0": ([0], [0], [], []), "1": ([0], [0], [], [])} #Dictionary for decision direction. Tuple: (0: conservative, 1: liberal){inside tuple: Total, correct, truelable, reslabel}
res_type = {"0": ([0], [0], [], []), "1": ([0], [0], [], []), "2": ([0], [0], [], []), "3": ([0], [0], [], []), "4": ([0], [0], [], [])}
#similar to decision direction: total, correct, true, predicted.

# Numbers
total = 0
total_right = 0
buffer = 0
is_first = True
# Loop now
for example in dataset:
    if(is_first is True):
       is_first = False
       continue # Check for the first time, and will never be checked again
    else:
      if(total == 1000):
        break
      
      input_text = example['text']
      input_ans = example['label']
      input_direction = example['decision_direction']
      input_res = example['respondent_type']

      completion = openai.ChatCompletion.create(
        temperature=0,
        model="gpt-3.5-turbo", 
        messages = [{"role": "system", "content" : "As a legal advisor, I specialize in providing guidance on various legal situations. Please describe the specific legal situation you need help with, and I will select the most appropriate label from the following options: (0, Criminal Procedure), (1, Civil Rights), (2, First Amendment), (3, Due Process), (4, Privacy), (5, Attorneys), (6, Unions), (7, Economic Activity), (8, Judicial Power), (9, Federalism), (10, Interstate Relations), (11, Federal Taxation), (12, Miscellaneous), (13, Private Action). It's important to include all relevant details related to the situation to ensure accurate advice."},
        #messages = [{"role": "system", "content" : "I want you to think as a legal advisor. I will describe a legal situation, and then you will select the best corresponding label from the followings: (0, Criminal Procedure), (1, Civil Rights), (2, First Amendment), (3, Due Process), (4, Privacy), (5, Attorneys), (6, Unions), (7, Economic Activity), (8, Judicial Power), (9, Federalism), (10, Interstate Relations), (11, Federal Taxation), (12, Miscellaneous), (13, Private Action)."},
        {"role": "user", "content" : "What would be the best corresponding label of the legal situation" + text[:4000] + "will be? You should only reply with the index number (range from 0 to 13)"},
        {"role": "assistant", "content" : "9"},
        {"role": "user", "content" : "What would be the best corresponding label of the legal situation" + input_text[:4000] + "will be? You should only reply with the index number (range from 0 to 13)"}]
      )

      if(completion['choices'][0]['message']['content'] == str(input_ans)): # Check if the predicted label is equal to actual label.
          total_right += 1
          decision_dir[str(input_direction)][1][0] += 1
          res_type[str(input_res)][1][0] += 1
          #appending correct label
      
      else: #A safe layer to check if the result is correct but format issue causing it to receive wrong answer
        if(len(completion['choices'][0]['message']['content']) > 1):
            match = re.search(r'\d+', completion['choices'][0]['message']['content']) #Regular expression to make sure there is only one item here.
            if match:
                completion['choices'][0]['message']['content'] = str(match.group())
                if completion['choices'][0]['message']['content'] == str(input_ans): #check if it is the correct label
                  total_right += 1 #Total correct append
                  decision_dir[str(input_direction)][1][0] += 1
                  res_type[str(input_res)][1][0] += 1

      #If the result is wrong then it goes here.
      decision_dir[str(input_direction)][2].append(str(input_ans))
      decision_dir[str(input_direction)][3].append(completion['choices'][0]['message']['content'])
      res_type[str(input_res)][2].append(str(input_ans))
      res_type[str(input_res)][3].append(completion['choices'][0]['message']['content'])
      # total++
      decision_dir[str(input_direction)][0][0] += 1
      res_type[str(input_res)][0][0] += 1
      
      #Add 1 to the total number
      total += 1
      print(total, " out of 1000 complete")
      buffer += 1
      if(buffer % 10 == 0):
        time.sleep(10)
      if(buffer % 200 == 0):
         time.sleep(120)

print("Using GPT3.5 turbo")
print(total_right)
print(total)
print(total_right / total * 100)

print("Real answer from dataset for lib: ", decision_dir["1"][2])
print("GPT's response for lib: ", decision_dir["1"][3])
print("Real answer from dataset for con: ", decision_dir["0"][2])
print("GPT's response for con: ", decision_dir["0"][3])
print("For conservative this is the total and total correct ", decision_dir["0"][0][0], " ----", decision_dir["0"][1][0])
print("For liberal this is the total and total correct ", decision_dir["1"][0][0], " ----", decision_dir["1"][1][0])

f1_scores_lib = f1_score(decision_dir["1"][2], decision_dir["1"][3], average="macro")
f1_scores_con = f1_score(decision_dir["0"][2], decision_dir["0"][3], average="macro")

print("mF1 Score for liberal:", f1_scores_lib)
print("mF1 Score for conservative:", f1_scores_con)

ave_f1_scores_decision_dir = (f1_scores_con + f1_scores_lib) / 2

GD = math.sqrt(0.5 * math.pow(f1_scores_lib - ave_f1_scores_decision_dir, 2) * math.pow(f1_scores_con - ave_f1_scores_decision_dir, 2))
print("The mf1 average is:", ave_f1_scores_decision_dir)
print("The GD score is:", GD)
print("The worst mf1 score is:", min(f1_scores_con, f1_scores_lib))


print("Real answer from dataset for other: ", res_type["0"][2])
print("GPT's response for other: ", res_type["0"][3])
print("Real answer from dataset for person: ", res_type["1"][2])
print("GPT's response for person: ", res_type["1"][3])
print("Real answer from dataset for organization: ", res_type["2"][2])
print("GPT's response for organization: ", res_type["2"][3])
print("Real answer from dataset for public entity: ", res_type["3"][2])
print("GPT's response for public entity: ", res_type["3"][3])
print("Real answer from dataset for facility: ", res_type["4"][2])
print("GPT's response for facility: ", res_type["4"][3])
print("For other this is the total and total correct ", res_type["0"][0][0], " ----", res_type["0"][1][0])
print("For person this is the total and total correct ", res_type["1"][0][0], " ----", res_type["1"][1][0])
print("For organization this is the total and total correct ", res_type["2"][0][0], " ----", res_type["2"][1][0])
print("For public entity this is the total and total correct ", res_type["3"][0][0], " ----", res_type["3"][1][0])
print("For facility this is the total and total correct ", res_type["4"][0][0], " ----", res_type["4"][1][0])

f1_scores_other = f1_score(res_type["0"][2], res_type["0"][3], average="macro")
f1_scores_person = f1_score(res_type["1"][2], res_type["1"][3], average="macro")
f1_scores_org = f1_score(res_type["2"][2], res_type["2"][3], average="macro")
f1_scores_pe = f1_score(res_type["3"][2], res_type["3"][3], average="macro")
f1_scores_facil = f1_score(res_type["4"][2], res_type["4"][3], average="macro")

print("mF1 Score for other:", f1_scores_other)
print("mF1 Score for person:", f1_scores_person)
print("mF1 Score for organization:", f1_scores_org)
print("mF1 Score for public entity:", f1_scores_pe)
print("mF1 Score for facility:", f1_scores_facil)

ave_f1_scores_res_type = (f1_scores_other + f1_scores_person + f1_scores_org + f1_scores_pe + f1_scores_facil) / 5

GD_res = math.sqrt(0.2 * math.pow(f1_scores_other - ave_f1_scores_res_type, 2) * math.pow(f1_scores_person - ave_f1_scores_res_type, 2) * math.pow(f1_scores_org - ave_f1_scores_res_type, 2) * math.pow(f1_scores_pe - ave_f1_scores_res_type, 2) * math.pow(f1_scores_facil - ave_f1_scores_res_type, 2))
print("The mf1 average is:", ave_f1_scores_res_type)
print("The GD score is:", GD_res)
print("The worst mf1 score is:", min(f1_scores_other, f1_scores_person, f1_scores_org, f1_scores_pe, f1_scores_facil))
