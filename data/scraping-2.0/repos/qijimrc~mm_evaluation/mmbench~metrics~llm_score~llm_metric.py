import os
import time
import pandas as pd
import jsonlines
import openai
import numpy as np

from tqdm import tqdm
from sat.helpers import print_rank0
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

from mmbench.common.registry import Registry
from mmbench.metrics.base_metric import BaseMetric
from mmbench.metrics.llm_score.chat_api import ChatAPI

prompt_dic_en = {"system_prompt": "You are a helpful and precise assistant for checking the quality of the answer.",
              "prompt_template": "[Detailed Image Description]\n{human_annotation}\n\n[Question]\n{question}\n\n[The Start of Assistant 1's Answer]\n{answer_1}\n\n[The End of Assistant 1's Answer]\n\n[The Start of Assistant 2's Answer]\n{answer_2}\n\n[The End of Assistant 2's Answer]\n\n[System]\n{prompt}\n\n",
              "defaults": {
                  "prompt": "We would like to request your feedback on the performance of two AI assistants in response to the user question and image description displayed above. AI assistants are provided with detailed image description and questions.\nPlease rate the helpfulness, relevance, accuracy, comprehensiveness of their responses. Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.\nPlease output a single line containing only two values indicating the scores for Assistant 1 and 2, respectively. The two scores are separated by a space."},
              "description": "Prompt for general questions", "category": "general"}

prompt_dic_zh = {"system_prompt": "你是一个帮助检查答案质量的有用而准确的助手。",
              "prompt_template": "[Detailed Image Description]\n{human_annotation}\n\n[Question]\n{question}\n\n[The Start of Assistant 1's Answer]\n{answer_1}\n\n[The End of Assistant 1's Answer]\n\n[The Start of Assistant 2's Answer]\n{answer_2}\n\n[The End of Assistant 2's Answer]\n\n[System]\n{prompt}\n\n",
              "defaults": {
                  "prompt": "我们希望请您就上述用户问题和图像描述的两位AI助手的表现提供反馈。AI助手们已被提供详细的图像描述和问题。\n请您评价他们的回答在帮助性、相关性、准确性和全面性方面的表现。每位助手将在1到10的评分尺度上获得综合得分，其中较高的分数表示整体表现更好。\n请输出一行文本，其中只包含两个数值，分别表示助手1和助手2的分数。这两个分数之间用一个空格隔开。"},
              "description": "Prompt for general questions", "category": "general"}

prompt_dic = {
    "english": prompt_dic_en,
    "chinese": prompt_dic_zh
}

@Registry.register_metric('llm_score')
class LLMScore(BaseMetric):
    def __init__(self, args):
        self.chatapi = ChatAPI()
        self.server_name = "chatglm2-66b"
        self.max_thread_nums = args.max_thread_nums if hasattr(args, "max_thread_nums") else 8
        self.MAX_API_RETRY = 3

    def process_reply(self, reply):
        # Process the response to the scores
        rets = reply.strip().split(" ")
        if len(rets) >= 2:
            score1, score2 = rets[:2]
        score1, score2 = float(score1), float(score2)
        return score1, score2, "ok"

    def evaluate(self, query, language):
        """
        Evaluate the response using the GPT-4 model.
        Args:
        - query: str, the user's query

        Returns:
        - result: tuple(int, int, str), the evaluation result in the format (score1, score2, response)
        - score1: int, the first score
        - score2: int, the second score
        - response: str, the status from the model
        """
        # Retry the API call for a maximum of MAX_API_RETRY times
        for i in range(self.MAX_API_RETRY):
            try:
                # Make API call to the Chat Completion API
                prompt = f'{prompt_dic[language]["system_prompt"]}\n{query}'
                status, content = self.chatapi.get_response(api_server=self.server_name, prompt=prompt)
                if status != "SUCCESS":
                    continue
                # Process the response
                reply = self.process_reply(content)
                if len(reply) == 3:
                    return reply[0], reply[1], reply[2]
                else:
                    print_rank0(f"error in reply: {reply}")
                    return 0, 0, "error"
            except Exception as e:
                print_rank0(f"error in content")
                return 0, 0, "error"
        return 0, 0, "error"
    
    def process(self, single_data, language, name="TestModel"):
        # Load data from file
        row = single_data
        # Process each answer
        ques = row['question']
        human_annotation = row['human_annotation']
        ans1 = row['answer']
        ans2 = row['preds']
        
        # Check if ans2 is a list and convert it to a string
        if isinstance(ans2, list):
            ans2 = ans2[0]
            
        # Create prompt using the provided template
        prompt = prompt_dic[language]["defaults"]["prompt"]
        query1 = prompt_dic[language]["prompt_template"].format(human_annotation=human_annotation, question=ques, answer_1=ans1,
                                                    answer_2=ans2, prompt=prompt)
        query2 = prompt_dic[language]["prompt_template"].format(human_annotation=human_annotation, question=ques, answer_1=ans2,
                                                    answer_2=ans1, prompt=prompt)
        
        # Evaluate the answer with position balancing
        round1_score1, round1_score2, round1_reason = self.evaluate(query1, language)
        round2_score1, round2_score2, round2_reason = self.evaluate(query2, language)
        
        # Create a dictionary to store the results
        results = [{
            'ques_id': row["question_id"],
            'human_annotation': row['human_annotation'],
            'category': row['category'],
            'task_name': row['task_name'],
            'round1': {
                'model1': 'gpt4-ha',
                'model2': name,
                'answer1': ans1,
                'answer2': ans2,
                'score1': round1_score1,
                'score2': round1_score2,
                'reason': round1_reason
            },
            'round2': {
                'model1': name,
                'model2': 'gpt4-ha',
                'answer1': ans2,
                'answer2': ans1,
                'score1': round2_score1,
                'score2': round2_score2,
                'reason': round2_reason
            }
        }]
        return results

    def multiple_process(self, data_df, language):
        results = []
        max_threads = min(os.cpu_count(), max(1, self.max_thread_nums), len(data_df))
        print_rank0(f'Using {max_threads} threads...')
        try:
            with ThreadPoolExecutor(max_workers=max_threads) as executor:
                futures = {executor.submit(self.process, *(data, language)): data for idx, data in data_df.iterrows()}
                with tqdm(total=len(data_df)) as progress_bar:
                    for future in as_completed(futures):
                        try:
                            result = future.result()
                            results.extend(result)
                        except Exception as e:
                            print(f"Error occurred while processing data: {e}")
                        progress_bar.update(1)
        except Exception as e:
            print_rank0(f'Process: {e}')
        return results

    def calc_scores(self, results_df, language="english") -> Dict:
        # get chat model responses
        results = self.multiple_process(results_df, language) 
        # Dictionary to store scores for each model
        scores = {}
        # Dictionary to store scores categorized by category and model
        scores_cate_wise = {}
        # Open the JSONL file
        for idx, info in enumerate(results):
            # Get round1 and round2 information from each line
            round1 = info["round1"]
            round2 = info["round2"]
            rounds = [round1, round2]

            # Iterate over both rounds
            for round_ in rounds:
                # Check if the reason is not 'error'
                if round_['reason'] != 'error':
                    model1 = round_["model1"]
                    model2 = round_["model2"]

                    # Check if the model1 is not in scores dictionary
                    if model1 not in scores:
                        scores[model1] = []
                    # Check if the model2 is not in scores dictionary
                    if model2 not in scores:
                        scores[model2] = []

                    # Append the scores for model1 and model2
                    scores[model1].append(round_["score1"])
                    scores[model2].append(round_["score2"])

                    # Split the category string and iterate over each category
                    for cate in info["category"].split(','):
                        # Check if the category is not in scores_cate_wise dictionary
                        if cate not in scores_cate_wise:
                            scores_cate_wise[cate] = {}
                        # Check if the model1 is not in scores_cate_wise dictionary for the category
                        if model1 not in scores_cate_wise[cate]:
                            scores_cate_wise[cate][model1] = []
                        # Check if the model2 is not in scores_cate_wise dictionary for the category
                        if model2 not in scores_cate_wise[cate]:
                            scores_cate_wise[cate][model2] = []

                        # Append the scores for model1 and model2 categorized by category
                        scores_cate_wise[cate][model1].append(round_["score1"])
                        scores_cate_wise[cate][model2].append(round_["score2"])

        print(' -------------- TouchStone Overall Score ------------- ')
        final_ret = {}
        # Calculate and print the mean scores for each model
        for model_name in scores:
            tmp_score = np.mean(scores[model_name]) * 100
            final_ret[f"{model_name}-Overall"] = tmp_score
            print(model_name, tmp_score)

        # Print the scores categorized by category
        for cate in scores_cate_wise:
            print(' -------------- {} ------------- '.format(cate))
            for model_name in scores_cate_wise[cate]:
                scores_cate_wise[cate][model_name] = np.mean(scores_cate_wise[cate][model_name]) * 100
                # Calculate and print the mean scores for each model in each category
                print(model_name, scores_cate_wise[cate][model_name])
                final_ret[f"{model_name}-{cate}"] = scores_cate_wise[cate][model_name]
        return final_ret

if __name__ == "__main__":
    import argparse
    def parse_args():
        parser = argparse.ArgumentParser(description="TouchStone evaluation.")
        parser.add_argument("--max_thread_nums", type=int, default=1, help="submitted tsv file")
        args = parser.parse_args()
        return args
    args = parse_args()
    
    llm_score = LLMScore(args)
    df = pd.read_csv("/mnt/shared/img_datasets/mmbench_datasets/raw/TouchStone/CogVLM.csv")
    score = llm_score.calc_scores(df, "english")
    print(score)