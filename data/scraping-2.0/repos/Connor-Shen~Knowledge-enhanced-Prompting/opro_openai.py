from Prompts_openai import meta_prompt, scorer_prompt, system_prompt
import pandas as pd
import openai
import re
import os
import time 


def build_text_and_scores(performance_df):
    """
    Build a string containing text and scores from a DataFrame.
    Args:
        performance_df (pd.DataFrame): DataFrame containing text and scores.
    Returns:
        str: A formatted string containing text and scores.
    """
    return ''.join([f"text:\n{performance_df.iloc[i]['text']}\nrecall:\n{performance_df.iloc[i]['recall']}\nprecision:\n{performance_df.iloc[i]['precision']}\naccuracy:\n{performance_df.iloc[i]['accuracy']}\nf1_score:\n{performance_df.iloc[i]['f1_score']}" for i in range(len(performance_df))])

# sort the values by recall
def rank_instructions(performance_df, num_scores):
    """
    Rank instructions based on scores and limit the number of instructions.
    Args:
        performance_df (pd.DataFrame): DataFrame containing text and scores.
        num_scores (int): Number of top instructions to keep.
    Returns:
        pd.DataFrame: A DataFrame containing the top-ranked instructions.
    """
    performance_df = performance_df.sort_values(by='f1_score')
    if len(performance_df) > num_scores:
        performance_df = performance_df.tail(num_scores)
    return performance_df

def sample_exemplars(training_exemplar_df, num_exemplars=3):
    """
    Sample exemplars from a DataFrame.
    Args:
        training_exemplar_df (pd.DataFrame): DataFrame containing training exemplars.
        num_exemplars (int): Number of exemplars to sample.
    Returns:
        str: A string containing sampled exemplars.
    """
    exemplar_sample = training_exemplar_df.sample(num_exemplars)
    return ''.join([f"<prompt>\n<question>: {exemplar_sample.iloc[i]['text'][:150]}\noutput:\n{exemplar_sample.iloc[i]['label']}\n" for i in range(num_exemplars)])


def generate_prompts(system_prompt, meta_prompt, texts_and_scores, exemplars, n_prompts=3):
    """
    Generate prompts using an optimizer chain.
    Args:
        optimizer_chain (LLMChain): Optimizer language model chain.
        texts_and_scores (str): Texts and scores string.
        exemplars (str): Exemplars string.
        n_prompts (int): Number of prompts to generate.
    Returns:
        list: A list of generated prompts.
    """
    system = system_prompt
    user = meta_prompt.format(texts_and_scores=texts_and_scores, exemplars=exemplars)
    openai.api_key = "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"

    generation_list = []
    for i in range(n_prompts):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-1106",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ]
        )
        generation = response['choices'][0]['message']['content'].replace("[", "").replace("]", "")
        time.sleep(1.5)
        generation_list.append(generation)
        print(f"the new generated prompt is: {generation}")

    return generation_list

    # calculate accuracy, recall, precision
def calculate_metrics(predictions, labels):
    TP = sum([(p == 1) and (l == 1) for p, l in zip(predictions, labels)])
    TN = sum([(p == 0) and (l == 0) for p, l in zip(predictions, labels)])
    FP = sum([(p == 1) and (l == 0) for p, l in zip(predictions, labels)])
    FN = sum([(p == 0) and (l == 1) for p, l in zip(predictions, labels)])

    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0

    # calculate the weighted f1 score
    f1 = 2*(precision * recall) / (precision + recall) if (precision != 0 or recall != 0) else 0

    return accuracy, recall, precision, f1

# def score_prompts(scorer_chain, prompts, training_examples, performance_df):
#     """
#     Score prompts based on training examples.
#     Args:
#         scorer_chain (LLMChain): Scorer language model chain.
#         prompts (list): List of prompts to score.
#         training_examples (pd.DataFrame): DataFrame containing training examples.
#         performance_df (pd.DataFrame): DataFrame containing text and scores.
#     Returns:
#         pd.DataFrame: Updated performance DataFrame.
#     """
#     for prompt in prompts:
#         scores = []
#         for index, example in training_examples.iterrows():
#             question = example['question']
#             answer = example['raw_answer']
#             # in the form of 0 and 1 
#             sample_answer = scorer_chain.predict(question=question, instruction=prompt)
#             # 
#             scores.append(are_numbers_the_same(answer, sample_answer))
#         score = int(100 * sum(scores) / len(scores))
#         performance_df = performance_df.append({'text': prompt, 'score': score}, ignore_index=True)
#     return performance_df

# 编写一个函数来保存sample_answers_with_index到Excel文件
def save_sample_answers_with_index_to_excel(sample_answers_with_index, file_count):
    # 创建一个DataFrame来存储数据
    df = pd.DataFrame(sample_answers_with_index, columns=['index', 'analysis', 'prediction'])
    
    # 将数据保存到Excel文件
    excel_file_name = f"checkpoint_opt/opro_dm0_sample_answers_batch_{file_count}.xlsx"
    df.to_excel(excel_file_name, index=False)

def score_evaluates(prompts, training_examples, performance_df):

    # 初始化一个空的列表来保存sample_answer和索引
    sample_answers_with_index = []
    # 初始化一个计数器来跟踪保存的文件数量
    file_count = 2
    
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
    model="gpt-3.5-turbo-1106",
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
            if elapsed_time > 10:  # 设置超时时间为10秒
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

    return performance_df


def opro(performance_df, training_exemplar_df, system_prompt, meta_prompt, n_scores=20, n_exemplars=3, n_prompts=8, n_training_samples=10, max_iterations=3):
    """
    Optimize prompts using an optimizer chain and scorer chain.
    Args:
        optimizer_chain (LLMChain): Optimizer language model chain.
        scorer_chain (LLMChain): Scorer language model chain.
        performance_df (pd.DataFrame): DataFrame containing text and scores.
        training_exemplar_df (pd.DataFrame): DataFrame containing training exemplars.
        n_scores (int): Number of top scores to consider.
        n_exemplars (int): Number of exemplars to sample.
        n_prompts (int): Number of prompts to generate.
        n_training_samples (int): Number of training samples to use.
        max_iterations (int): Maximum number of optimization iterations.
    Returns:
        pd.DataFrame: Updated performance DataFrame.
    """
    performance_df = rank_instructions(performance_df, n_scores)
    for i in range(max_iterations):
        texts_and_scores = build_text_and_scores(performance_df)
        exemplars = sample_exemplars(training_exemplar_df, n_exemplars)
        prompts = generate_prompts(system_prompt, meta_prompt, texts_and_scores, exemplars, n_prompts)
        print(f"New Generated prompted List is: {prompts} ")
        training_examples = training_exemplar_df.sample(n_training_samples)
        performance_df = score_evaluates(prompts, training_examples, performance_df)
        # save the new generated prompts
        performance_df.to_excel(f"performance_tmp/newprompt_dm0_performance_iter{i}.xlsx", index=False)
        performance_df = rank_instructions(performance_df, n_scores)
        # save the checkpoint
        performance_df.to_excel(f"checkpoint_opt/oprp_performance_dm0_iter{i}.xlsx", index=False)
    return performance_df


if __name__ == '__main__':
    
    os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
    os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
    


    performance_df = pd.read_excel("checkpoint_opro/dm0_baselines.xlsx")

    training_exemplar_df = pd.read_excel("data/training_data.xlsx")

    output = opro(performance_df, training_exemplar_df,  system_prompt, meta_prompt, n_scores=10, n_exemplars=3, n_prompts=3, n_training_samples=100, max_iterations=2)
    print(output)
    output.to_excel("performance_tmp/performance_final_dm0.xlsx")

