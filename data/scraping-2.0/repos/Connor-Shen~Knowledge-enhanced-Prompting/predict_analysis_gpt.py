"""
set the environment of chatglm

!git clone https://github.com/THUDM/ChatGLM2-6B
!cd ChatGLM2-6B

import os
os.chdir("/content/ChatGLM2-6B")
!ls

!pip install -r requirements.txt
!pip install fastapi uvicorn

source /etc/network_turbo
"""

from Prompts_analysis import meta_prompt, scorer_prompt
import pandas as pd
import os
import re
import openai

os.environ['CURL_CA_BUNDLE'] = ''

# 设置您的API密钥
openai.api_key = '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'

# calculate accuracy, recall, precision, and fb score
def calculate_metrics(predictions, labels):

    TP = sum([(p == 1) and (l == 1) for p, l in zip(predictions, labels)])
    TN = sum([(p == 0) and (l == 0) for p, l in zip(predictions, labels)])
    FP = sum([(p == 1) and (l == 0) for p, l in zip(predictions, labels)])
    FN = sum([(p == 0) and (l == 1) for p, l in zip(predictions, labels)])

    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0

    # calculate the weighted f1 score
    beta = 1   # emphasize recall
    fb = ((beta**2 + 1) * precision * recall) / ((beta**2 * precision) + recall) if (precision != 0 or recall != 0) else 0

    return accuracy, recall, precision, fb

# 编写一个函数来保存sample_answers_with_index到Excel文件
def save_sample_answers_with_index_to_excel(sample_answers_with_index, file_count):
    # 创建一个DataFrame来存储数据
    df = pd.DataFrame(sample_answers_with_index, columns=['index', 'analysis', 'prediction'])

    # 将数据保存到Excel文件
    excel_file_name = f"checkpoint_opt/dm0_gpt4_sample_answers_batch_{file_count}.xlsx"
    df.to_excel(excel_file_name, index=False)


import time
# modify the score_prompts function
def score_evaluates(prompts, training_examples, performance_df):

    # 初始化一个空的列表来保存sample_answer和索引
    sample_answers_with_index = []
    # 初始化一个计数器来跟踪保存的文件数量
    file_count = 1
    
    for prompt in prompts:
        predictions = []
        labels = []
        raw_prompt = prompt
        for _, example in training_examples.iterrows():
            index = example["index"]
            question = example["text"]
            raw_label = example["label"]
            input_prompt = scorer_prompt.format(question=question, prompt=raw_prompt)
            start_time = time.time()  # 记录开始时间
            try:
                response = openai.ChatCompletion.create(
    model="gpt-4-1106-preview",
    messages=[
        {"role": "user", "content": input_prompt}
    ]
)
                sample_answer = response['choices'][0]['message']['content']
                time.sleep(3)
            except Exception as e:
                # 处理其他异常情况，可以选择跳过这条数据
                print(f"处理数据时发生异常: {str(e)}")
                continue

            elapsed_time = time.time() - start_time  # 计算执行时间
            if elapsed_time > 15:  # 设置超时时间为10秒
                # 超时情况，跳过这条数据
                print(f"处理数据时超时: {raw_prompt}")
                continue
            
            print("Index: ", index, flush=True)
            print(sample_answer, flush=True)
            match = re.search(r'### ?(?:<Label>|Label):\s*(.*?)(?=\n###|$)', sample_answer, re.DOTALL)
            if match:
                label_value = match.group(1).strip()
                if "0" in label_value or "ontrol" in label_value:
                    label = 0
                elif "1" in label_value or "epress" in label_value:
                    label = 1
                else:
                    label = 0
            else:
                print("Label not found!")
                label = None
                continue
                
            print(label, flush=True)
            predictions.append(label)
            labels.append(raw_label)

            # 将sample_answer和索引添加到列表中
            sample_answers_with_index.append((index, sample_answer, label))
            
        sample_answers_with_index.append((0,0,raw_prompt))
        save_sample_answers_with_index_to_excel(sample_answers_with_index, file_count)
        sample_answers_with_index = []
        file_count += 1
        
        
        accuracy, recall, precision, fb_score = calculate_metrics(predictions, labels)

        new_row = {
            'text': prompt,
            'accuracy': accuracy,
            'recall': recall,
            'precision': precision,
            'f1_score': fb_score,
        }
        performance_df = pd.concat([performance_df, pd.DataFrame([new_row])], ignore_index=True)
        # save the checkpoint
        performance_df.to_excel("checkpoint_opt/dm0_performance_gpt4.xlsx", index=False)

    return performance_df

def predict_eval(performance_df, training_examples, prompts):
    """
    return evaluation scores of different prompts
    Args:
        scorer_chain (LLMChain): Scorer language model chain.
        performance_df (pd.DataFrame): DataFrame containing text and scores.
        training_examples (pd.DataFrame): DataFrame containing training exemplars.
        prompts: list of prompts
    Returns:
        pd.DataFrame: Updated performance DataFrame.
    """
    performance_df = score_evaluates(prompts, training_examples, performance_df)
    return performance_df

if  __name__ == "__main__":

    os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
    os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
    
    performance_df = pd.read_excel("data/performance.xlsx")
    training_data_df = pd.read_excel("data/training_data.xlsx").iloc[:200]

    prompts = ["""You are an expert in identifying depression. Next you need to help me diagnose depression based on user tweets. I want you to think step by step.
Here are some key words of depression that may help you identify the user's mental state:
'depressed, sad, unhappy, miserable, sorrowful, dejected, downcast, downhearted, despondent, disconsolate, wretched, glum, gloomy, dismal, melancholy, woebegone, forlorn, crestfallen, heartbroken, inconsolable, grief-stricken, broken-hearted, heavy-hearted, heavy, low-spirited, morose, despairing, disheartened, discouraged, demoralized, desolate, despairing, desperate, anguished, distressed, fraught, distraught, suffering, wretched, traumatized, tormented, tortured, racked, haunted, hunted, persecuted, haunted, cursed, luckless, ill-fated, ill-starred, jinxed, unhappy, sad, miserable, sorrowful, dejected, depressed, down, downhearted, downcast, despondent, disconsolate, desolate, wretched, glum, gloomy, dismal'
"""]
    # prompt_df = pd.read_excel("/data/performance_final_dm2.xlsx")
    # prompts = prompt_df["text"].tolist()[8:]
    # print(len(prompts))
    
    evaluation_df = predict_eval(performance_df, training_data_df, prompts)
    print(evaluation_df)
    evaluation_df.to_excel("data/gpt4_dm0_performance_final_dm1.xlsx", index=False)


   
 