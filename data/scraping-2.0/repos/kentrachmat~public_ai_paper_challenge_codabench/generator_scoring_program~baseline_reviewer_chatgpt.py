import openai
import json
import os
import random
from tqdm.auto import tqdm
from collections import defaultdict
from utils import *

from config import TRUNCATE

class BaselineReviewer:
    def __init__(self):
        current_real_dir = os.path.dirname(os.path.realpath(__file__))
        with open(os.path.join(current_real_dir, 'scoring_program_chatgpt_api_key.json'), 'rb') as f:
            self.api_key = json.load(f)['key']

        openai.api_key = self.api_key

    def compare_papers(self, good_papers, bad_papers, prediction_paper, prediction_prompt, criteria):
        ''' Function to compare the solution and prediction papers
        Args:
            good_papers: list of good papers
            bad_papers: json file of bad papers
            prediction_paper: prediction paper
            prediction_prompt: prediction prompt
            criteria: criteria for comparison in json format 
                (for example: {'Clarity': 
                                    {'Correct grammar': '[0 if paper 1 is better, 1 if paper 2 is better]'}, 
                                    'Organization': '[0 if paper 1 is better, 1 if paper 2 is better]',
                                    'Explanation': '[0 if paper 1 is better, 1 if paper 2 is better]'},
                                    ...
                                'Contributions':
                                    {'Coverage': '[0 if paper 1 is better, 1 if paper 2 is better]',
                                    ...
                                    }
                                ...
                                }
                )
        Returns:
            comparison_result: comparison result in json format
                (for example: {'Clarity':
                                    {'Correct grammar': 0.0},
                                    'Organization': 0.0,
                                    'Explanation': 0.0},
                                    ...
                                'Contributions':
                                    {'Coverage': 0.0,
                                    ...
                                    }
                                ...
                                }
                )
            '''
        # If the prediction paper is {"Title: '', "Abstract": '', "Citations": '', "Introduction": '', "Related work": '', "Method": '', "Experiments": '', "Results": '', "Conclusion": ''}, or similar. Basically no values for the keys. 
        # Then we return the default values for the comparison result. which is 0 for all the criteria. and no comments.
        empty_string = True
        for pred in custom_json_loads(prediction_paper):
            if pred['text'] != '':
                empty_string = False
                break

        if empty_string:
            comparison_result = {}
            comparison_comment = {}
            for category, sub_criteria in criteria.items():
                if isinstance(sub_criteria, str):
                    comparison_result[category] = 0.0
                    comparison_comment[category] = "This is a test response. Please ignore."
                else:
                    comparison_result[category] = {}
                    comparison_comment[category] = {}
                    for sub_criterion, value in sub_criteria.items():
                        comparison_result[category][sub_criterion] = 0.0
                        comparison_comment[category][sub_criterion] = "This is a test response. Please ignore."

            # add soundness
            comparison_result['soundness'] = {'c1': 0.0, 'c2': 0.0, 'c3': 0.0}
            comparison_comment['soundness'] = {'c1': "This is a test response. Please ignore.", 'c2': "This is a test response. Please ignore.", 'c3': "This is a test response. Please ignore."}

            # add relevance
            comparison_result['relevance'] = {'title': 0.0, 'abstract': 0.0, 'citations': 0.0}
            comparison_comment['relevance'] = {'title': "This is a test response. Please ignore.", 'abstract': "This is a test response. Please ignore.", 'citations': "This is a test response. Please ignore."}
            return [comparison_result, comparison_comment]
            
        print("\nComparing good papers with prediction paper...")
        good_results = []
        for good_paper in tqdm(good_papers):
            good_result = self.compare_two_paper(good_paper, prediction_paper, prediction_prompt, json.dumps(criteria))
            good_results.append(good_result)
        # Combine good results
        combined_good_result = defaultdict(dict)
        for super_category, super_value in good_results[0].items():
            if isinstance(super_value, str) or isinstance(super_value, float) or isinstance(super_value, int):
                for good_result in good_results:
                    if super_category in combined_good_result:
                        combined_good_result[super_category] += int(good_result[super_category]) / len(good_results)
                    else:
                        combined_good_result[super_category] = int(good_result[super_category]) / len(good_results)  
            else:
                for sub_category, sub_value in super_value.items():
                    for good_result in good_results:
                        try:
                            if super_category in combined_good_result and sub_category in combined_good_result[super_category]:
                                combined_good_result[super_category][sub_category] += int(good_result[super_category][sub_category]) / len(good_results)
                            else:
                                combined_good_result[super_category][sub_category] = int(good_result[super_category][sub_category]) / len(good_results)
                        except:
                            print("Error: ", good_result)
                            print("super_category: ", super_category)
                            print("sub_category: ", sub_category)
                            print("good_results: ", good_results)
                            raise Exception
        
        print("\nComparing bad papers with prediction paper...")
        combined_bad_result = defaultdict(dict)
        for super_category, super_value in tqdm(bad_papers.items()):
            if isinstance(super_value, list):
                for bad_paper in super_value:
                    result = self.compare_two_paper(bad_paper, prediction_paper, prediction_prompt, json.dumps({super_category: "0 if paper 1 is better, 1 if paper 2 is better"}))
                    if super_category in combined_bad_result:
                        combined_bad_result[super_category] += int(result[super_category]) / len(super_value)
                    else:
                        combined_bad_result[super_category] = int(result[super_category]) / len(super_value)
            else:
                for sub_category, sub_value in super_value.items():
                    for bad_paper in sub_value:
                        result = self.compare_two_paper(bad_paper, prediction_paper, prediction_prompt, json.dumps({super_category: {sub_category: "0 if paper 1 is better, 1 if paper 2 is better"}}))
                        if super_category in combined_bad_result and sub_category in combined_bad_result[super_category]:
                            combined_bad_result[super_category][sub_category] += int(result[super_category][sub_category]) / len(sub_value)
                        else:
                            combined_bad_result[super_category][sub_category] = int(result[super_category][sub_category]) / len(sub_value)

        # Combine good and bad results
        combined_result = defaultdict(dict)

        for super_category, super_value in combined_good_result.items():
            if isinstance(super_value, float):
                combined_result[super_category] = 0.5*super_value + 0.5*combined_bad_result[super_category]
            else:
                for sub_category, sub_value in super_value.items():
                    combined_result[super_category][sub_category] = 0.5*sub_value + 0.5*combined_bad_result[super_category][sub_category]

        print("\nCalculate soundness")
        combined_result['soundness']['c1'] = self.compare_one_paper(prediction_paper,'Evaluate the soundness of the paper (whether the references are legitimate)','[float number between [0,1] where 0.0 is the lowest and 1.0 is the highest]')
        combined_result['soundness']['c2'] = self.compare_one_paper(prediction_paper,'Evaluate the soundness of the paper (whether the references employed by the paper align well with the content it generates)','[float number between [0,1] where 0.0 is the lowest and 1.0 is the highest]')
        combined_result['soundness']['c3'] = self.compare_one_paper(prediction_paper,'Evaluate the soundness of the paper (whether the paper does not randomly cite unrelated papers, maintaining a coherent academic narrative)','[float number between [0,1] where 0.0 is the lowest and 1.0 is the highest]')
        print("Done calculating soundness")

        combined_comments = self.get_html_comments(combined_result, prediction_paper)

        return [combined_result, combined_comments]
     
    def get_html_comments(self, generator_score, prediction_paper):
        combined_html = defaultdict(dict)

        print("Get html comments for each criterion...")
        for super_category, super_value in tqdm(generator_score.items()):
            if isinstance(super_value, float) or isinstance(super_value, int):
                    combined_html[super_category] = self.ask_chat_gpt_reason(prediction_paper, super_category, super_value)
            else:
                sub_criteria_gpt = self.ask_chat_gpt_reason(prediction_paper, str(super_value), None, "multiple")
                for sub_value in sub_criteria_gpt:
                    combined_html[super_category][sub_value['criterion']] = sub_value['reason']
        return dict(combined_html)
    
    def ask_chat_gpt_reason(self, prediction_paper, criterion, score, types="single"):
        conversation = [{"role": "system", "content": "You are a helpful assistant who will help me evaluate a paper. You have given a score and now you need to provide a reason for the score."}]
        if types == "multiple":
            temp = '[{"criterion":..., "reason":...}, {"criterion":..., "reason":...}, ...]'

            conversation.append({"role": "user", "content":
            f"Please provide a short reason for each score where 0.0 is the lowest and 1.0 is the highest score for each criterion: {criterion}.\n\n" + \
            "Paper :\n" + prediction_paper[:(1000 if TRUNCATE else len(prediction_paper))] + \
            f"\n\nOutput the result in this JSON format {temp}"})

            success = False
            num_trials = 0
            while not success and num_trials < 5:
                try:
                    result = ask_chat_gpt(conversation)["choices"][0]["message"]["content"]
                    return custom_json_loads(result)

                except Exception as e:
                    print("Error: ", e)
                    print("Retrying...")
                    num_trials += 1
                    success = False
        else:
            conversation.append({"role": "user", "content":
            f"Please provide a short reason for the score {score:.2f} for the criterion: {criterion}:\n" + \
            "Paper :\n" + prediction_paper[:(1000 if TRUNCATE else len(prediction_paper))]})

            result = ask_chat_gpt(conversation)["choices"][0]["message"]["content"]
            return str(result)
    
    def compare_one_paper(self, prediction_paper, description, criterion):
        conversation = [{"role": "system", "content": "You are a helpful assistant who will help me evaluate a paper."}]
        conversation.append({"role": "user", "content":
        f"{description} the output of the template must be a single float value, no explanation:\nThe template:\n" + \
        criterion + \
        "Paper :\n" + prediction_paper[:(1000 if TRUNCATE else len(prediction_paper))]})

        success = False
        num_trials = 0
        while not success and num_trials < 5:
            try:
                result = ask_chat_gpt(conversation)["choices"][0]["message"]["content"]
                return float(result)
                success = True
            except Exception as e:
                print("Error: ", e)
                print("Retrying...")
                num_trials += 1
                success = False

    def compare_two_paper(self, paraphrased_paper, prediction_paper, prediction_prompt, criterion):
        """ Function to compare two papers
        Args:
            paraphrased_paper: paraphrased paper
            prediction_paper: prediction paper
            criterion: criterion for comparison in json format
                (for example: {'Clarity':
                                    {'Correct grammar': '[0 if paper 1 is better, 1 if paper 2 is better]',
                                    ...
                                    }
                                'Contributions':
                                    {'Coverage': '[0 if paper 1 is better, 1 if paper 2 is better]',
                                    ...
                                    }
                                ...
                                }
                )
        Returns:
            comparison_result: comparison result in json format
        """
        # 50% chance to swap the order of the papers
        if random.random() < 0.5:
            flipped = True
            paraphrased_paper, prediction_paper = prediction_paper, paraphrased_paper
        else:
            flipped = False

        paraphrased_paper_without_references = custom_json_loads(paraphrased_paper)
        paraphrased_paper_without_references = [x for x in paraphrased_paper_without_references if x['heading'] != 'References']
        paraphrased_paper_without_references = json.dumps(paraphrased_paper_without_references)

        prediction_paper_without_references = custom_json_loads(prediction_paper)
        prediction_paper_without_references = [x for x in prediction_paper_without_references if x['heading'] != 'References']
        prediction_paper_without_references = json.dumps(prediction_paper_without_references)

        conversation = [{"role": "system", "content": "You are a helpful assistant who will help me compare papers."}]
        conversation.append({"role": "user", "content": "Compare these 2 papers below and return the result using the template, no explanation:\nThe template:\n" + \
        criterion + \
        """Below are the details of the subcriteria:
        Relevance (Correctness): Does the answer address the prompt?
        Clarity (Correct language): Is the paper written in good English, with correct grammar, and precise vocabulary?
        Clarity (Organization): Is the paper well organized in meaningful sections and subsections?
        Clarity (Explanation): Are the concepts clearly explained, with short sentences?
        Contributions (Coverage): Does the answer provide a comprehensive overview, comparing and contrasting a plurality of viewpoints?
        Contributions (Title): Does the title of the paper accurately address the prompt?
        Contributions (Abstract): Does the abstract of the paper accurately reflect the content of the paper?
        Contributions (Introduction): Does the introduction of the paper accurately reflect the content of the paper?
        Contributions (Conclusion): Does the conclusion of the paper highlight the main findings of the paper?
        Soundness (Factuality/ Attribution): Does the answer present accurate facts, supported by citations of authoritative references?
        Responsibility: Does the paper address potential risks or ethical issues and is respectful of human moral values, including fairness, and privacy, and is free of libelous or unlawful statements, does not infringe upon the rights of others, or contain material or instructions that might cause harm or injury?
        """ + \
        f"These 2 papers below are generated using this prompt: {prediction_prompt}. \n" + \
        "The papers:\n'Paper 1':\n" + paraphrased_paper_without_references[:(1000 if TRUNCATE else len(paraphrased_paper))] + ",\n'Paper 2':\n" + prediction_paper_without_references[:(1000 if TRUNCATE else len(prediction_paper))] + '\nRemember to return the result in JSON format'})

        # In case when the length of the prompt is too long, we will create a shorter prompt
        while num_tokens_from_messages(conversation) > 8_000:
            shorter_paraphrased_paper = custom_json_loads(paraphrased_paper_without_references)
            shorter_paraphrased_paper = shorter_paraphrased_paper[:-1]
            shorter_paraphrased_paper = json.dumps(shorter_paraphrased_paper)

            shorter_prediction_paper = custom_json_loads(prediction_paper_without_references)
            shorter_prediction_paper = shorter_prediction_paper[:-1]
            shorter_prediction_paper = json.dumps(shorter_prediction_paper)

            conversation = [{"role": "system", "content": "You are a helpful assistant who will help me compare papers."}]
            conversation.append({"role": "user", "content":
            "Compare these 2 papers below and return the result using the template, no explanation:\nThe template:\n" + \
            criterion + \
            """Below are the details of the subcriteria:
            Relevance (Correctness): Does the answer address the prompt?
            Clarity (Correct language): Is the paper written in good English, with correct grammar, and precise vocabulary?
            Clarity (Organization): Is the paper well organized in meaningful sections and subsections?
            Clarity (Explanation): Are the concepts clearly explained, with short sentences?
            Contributions (Coverage): Does the answer provide a comprehensive overview, comparing and contrasting a plurality of viewpoints?
            Contributions (Title): Does the title of the paper accurately address the prompt?
            Contributions (Abstract): Does the abstract of the paper accurately reflect the content of the paper?
            Contributions (Introduction): Does the introduction of the paper accurately reflect the content of the paper?
            Contributions (Conclusion): Does the conclusion of the paper highlight the main findings of the paper?
            Soundness (Factuality/ Attribution): Does the answer present accurate facts, supported by citations of authoritative references?
            Responsibility: Does the paper address potential risks or ethical issues and is respectful of human moral values, including fairness, and privacy, and is free of libelous or unlawful statements, does not infringe upon the rights of others, or contain material or instructions that might cause harm or injury?
            """ + \
            f"These 2 papers below are generated using this prompt: {prediction_prompt}. \n" + \
            "The papers:\n{'Paper 1':\n" + shorter_paraphrased_paper + ",\n'Paper 2':\n" + shorter_prediction_paper+ '\n}Remember to return the result in JSON format'})

        success = False
        num_trials = 0
        while not success and num_trials < 5:
            try:
                result = ask_chat_gpt(conversation, temperature=0.2*num_trials)["choices"][0]["message"]["content"]
                json_result = custom_json_loads(result)

                # If the order of the papers is swapped, flip the result
                if flipped:
                    for super_category, super_value in json_result.items():
                        if isinstance(super_value, str) or isinstance(super_value, float) or isinstance(super_value, int):
                            json_result[super_category] = str(1 - int(super_value))
                        else:
                            for sub_category, sub_value in super_value.items():
                                json_result[super_category][sub_category] = str(1 - int(sub_value))
                                
                success = True
            except Exception as e:
                print("Error: ", e)
                print("Retrying...")
                num_trials += 1
                success = False

        return json_result 