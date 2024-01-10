# This is for CAIL dataset on GPT3
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
dataset = load_dataset('coastalcph/fairlex', 'cail', split='train')

# Example data
text = dataset[0]['text']

gender = {"0": ([0], [0], [], []), "1": ([0], [0], [], [])} #Dictionary for decision direction. Gender: (0: male, 1: female){inside tuple: Total, correct, truelable, reslabel}
region = {"0": ([0], [0], [], []), "1": ([0], [0], [], []), "2": ([0], [0], [], []), "3": ([0], [0], [], []), "4": ([0], [0], [], []), "5": ([0], [0], [], []), "6": ([0], [0], [], [])}
#similar to decision direction: total, correct, true, predicted. But it is for seven province regions

# Numbers
total = 0
total_right = 0
buffer = 0
is_first = True
found = 0
# Loop now
for example in dataset:
    if(is_first is True):
       is_first = False
       continue # Check for the first time, and will never be checked again

    if(found == 100):
       break
    
    if(str(example['court_region']) == "6"):
      found += 1
      input_text = example['text']
      input_ans = example['label']
      input_gender = example['defendant_gender']
      input_reg = example['court_region']

      prompt1 = f"你是中国的一名法律顾问,请阅读案件并根据中国的法律预测被告的刑期有多长。如果被告免于监禁,请回答'0';如果被告的刑期小于或等于一年,请回答'1';如果被告的刑期小于等于三年但大于一年,请回答'2';如果被告的刑期小于等于五年但大于三年,请回答'3';如果被告的刑期小于等于十年但大于五年,请回答'4';如果被告的刑期大于十年,请回答'5'. 请阅读以下事件: {input_text[:1500]} 以上事件中被告的刑期有多长?你的回答只能包含从0到5的一个数字"
      completion = openai.Completion.create(
        engine="text-davinci-003",  # Change this to the Davinci 3 model
        prompt=prompt1,
        temperature=0
      )

      if(completion['choices'][0]['text'] == str(input_ans)): # Check if the predicted label is equal to actual label.
          total_right += 1
          gender[str(input_gender)][1][0] += 1
          region[str(input_reg)][1][0] += 1
          #appending correct label
      
      else: #A safe layer to check if the result is correct but format issue causing it to receive wrong answer
        if(len(completion['choices'][0]['text']) > 1):
            match = re.search(r'\d+', completion['choices'][0]['text']) #Regular expression to make sure there is only one item here.
            if match:
                completion['choices'][0]['text'] = str(match.group())
                if completion['choices'][0]['text'] == str(input_ans): #check if it is the correct label
                  total_right += 1 #Total correct append
                  gender[str(input_gender)][1][0] += 1
                  region[str(input_reg)][1][0] += 1

      #If the result is wrong then it goes here.
      gender[str(input_gender)][2].append(str(input_ans))
      gender[str(input_gender)][3].append(completion['choices'][0]['text'])
      region[str(input_reg)][2].append(str(input_ans))
      region[str(input_reg)][3].append(completion['choices'][0]['text'])
      # total++
      gender[str(input_gender)][0][0] += 1
      region[str(input_reg)][0][0] += 1
      
      #Add 1 to the total number
      total += 1
      print(total, " out of 1000 complete")
      buffer += 1
      if(buffer % 10 == 0):
        time.sleep(10)
      if(buffer % 200 == 0):
         time.sleep(120)

print("Using GPT3.5 turbo")


f1_scores_ZJ = f1_score(region["6"][2], region["6"][3], average="macro")

print("mF1 Score for ZJ:", f1_scores_ZJ)

ave_f1_scores_reg = (0.148113134743015 + 0.116211484593837 + 0.0238095238095238 + 0.20016339869281 + 0.104353741496598 + 0.177157314538718 + f1_scores_ZJ) / 7

GD_res = math.sqrt(1/7 * math.pow(0.148113134743015 - ave_f1_scores_reg, 2) * math.pow(0.116211484593837 - ave_f1_scores_reg, 2) * math.pow(0.0238095238095238 - ave_f1_scores_reg, 2) * math.pow(0.20016339869281 - ave_f1_scores_reg, 2) * math.pow(0.104353741496598 - ave_f1_scores_reg, 2) * math.pow(0.177157314538718 - ave_f1_scores_reg, 2) * math.pow(f1_scores_ZJ - ave_f1_scores_reg, 2))
print("The mf1 average is:", ave_f1_scores_reg)
print("The GD score is:", GD_res)
print("The worst mf1 score is:", min(0.148113134743015, 0.116211484593837, 0.0238095238095238, 0.20016339869281, 0.104353741496598, 0.177157314538718, f1_scores_ZJ))
