import openai
import json5 as json
import re
import os
import random
import numpy as np
from collections import defaultdict

from metacriteria.utils import custom_json_loads

from metacriteria.utils import *
from metacriteria.a_rating import Rating
from metacriteria.b_precision import Precision
from metacriteria.c_correctness import Correctness
from metacriteria.d_recommendation import Recommendation
from metacriteria.e_respectfulness import Respectfulness

from tqdm.auto import tqdm

from config import DEBUG

class MetaTextReviewer:
    def __init__(self):
        current_real_dir = os.path.dirname(os.path.realpath(__file__))
        with open(os.path.join(current_real_dir, 'scoring_program_chatgpt_api_key.json'), 'rb') as f:
            self.api_key = json.load(f)['key1']

        openai.api_key = self.api_key

        self.rating_reviewer = Rating()
        self.precision_reviewer = Precision()
        self.correctness_reviewer = Correctness()
        self.recommendation_reviewer = Recommendation()
        self.respectfulness_reviewer = Respectfulness()

    def set_api_key(self, name_api_key):
        current_real_dir = os.path.dirname(os.path.realpath(__file__))
        with open(os.path.join(current_real_dir, 'scoring_program_chatgpt_api_key.json'), 'rb') as f:
            self.api_key = json.load(f)[name_api_key]
        openai.api_key = self.api_key
        
    def get_meta_review_scores(self, scores_and_comments, criteria_to_review):
        meta_reviews = []
        for score_and_comment in tqdm(scores_and_comments):
            meta_review = {}
            for criterion in criteria_to_review:
                # it will return a list of 5 scores:
                # - A - Score: Is the score consistent with the text feed-back?
                # - B - Precision (clarity): Is the text feed-back precise (does it point to a specific reason of praise of criticism)?
                # - C - Correctness (soundness): Is the praise or criticism correct and well substantiated?
                # - D - Recommendation (contribution): Does the text feed-back provide detailed and actionable recommendations for improvement?
                # - E - Respectfulness (responsibility): Is the language polite and non discriminatory?
                meta_review[criterion] = self.get_meta_review_score(score_and_comment[criterion], criterion)
            meta_reviews.append(meta_review)
        return meta_reviews

    def get_meta_review_score(self, score_and_comment, criterion):
        if DEBUG:
            return np.array([random.random() for _ in range(5)])
        else:    
            score = score_and_comment['score']
            comment = score_and_comment['comment']

            rating_prompt = self.rating_reviewer.get_prompt_for_evaluation(score, comment)
            precision_prompt = self.precision_reviewer.get_prompt_for_evaluation(score, comment)
            correctness_prompt = self.correctness_reviewer.get_prompt_for_evaluation(score, comment)
            recommendation_prompt = self.recommendation_reviewer.get_prompt_for_evaluation(score, comment)


            scores = []
            for i, metacriteria_prompt in enumerate([rating_prompt, precision_prompt, correctness_prompt, recommendation_prompt]):
                success = False
                num_trials = 0
                while not success and num_trials < 5:
                    try:
                        prompt = [
                            {"role": "system", "content":"You are a meta-reviewer who will help me evaluate a paper. You have given a score and a comment. Now you need to provide a grading for the score."},
                            {"role": "user", "content": \
                                "You are a meta-reviewer who will help me evaluate a paper. You have given a score and a comment. Now you need to provide a single number for your assessment in Likert scale from 1 to 3: [1 for No, 2 for More-or-less, 3 for Yes]}. No explaination needed.\n" + \
                                metacriteria_prompt}
                        ]
                        
                        answer = ask_chat_gpt(prompt)["choices"][0]["message"]["content"]
                        try:
                            score = (float(answer) - 1) / 2
                        except:
                            pattern = r"Assessment: (\d+(\.\d+)?)(/(\d+(\.\d+)?))?"
                            match = re.search(pattern, answer)
                            matched = match.group(1, 4)
                            if matched[1] is None:
                                score = (float(matched[0]) - 1) / 2
                            else:
                                score = float(matched[0]) / float(matched[1])
                        scores.append(score)
                        success = True
                    except Exception as e:
                        print("Error: ", e)
                        print("Retrying...")
                        num_trials += 1
                        success = False
            scores.append(self.respectfulness_reviewer.evaluate(score, comment))
            return np.array(scores)

    def get_meta_review_reasons(self, scores_and_comments, meta_review_scores, criteria_to_review):
        meta_review_reasons = {}
        # print("Getting meta-review reasons...")
        for criterion in tqdm(criteria_to_review):
            meta_review_reasons[criterion] = self.get_meta_review_reason(scores_and_comments[criterion], meta_review_scores[criterion], criterion)

        return meta_review_reasons
    
    def get_meta_review_reason(self, score_and_comment, meta_review_scores, criterion):
        """ Get the reason for the meta-review score 
        Args:
            score_and_comment: a dictionary of the form {score: score, comment: comment}
            meta_review_scores: a list of 5 scores: [A, B, C, D, E]
            criterion: the criterion to review
        Returns:
            a list of string of reasons for each score
        """
        if DEBUG:
            # return random string
            return ["".join([random.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(100)]) for _ in range(5)]
        else:
            score = score_and_comment['score']
            comment = score_and_comment['comment']

            rating_prompt = self.rating_reviewer.get_prompt_for_reason(score, comment, meta_review_scores[0])
            precision_prompt = self.precision_reviewer.get_prompt_for_reason(score, comment, meta_review_scores[1])
            correctness_prompt = self.correctness_reviewer.get_prompt_for_reason(score, comment, meta_review_scores[2])
            recommendation_prompt = self.recommendation_reviewer.get_prompt_for_reason(score, comment, meta_review_scores[3])
            respectfulness_prompt = self.respectfulness_reviewer.get_prompt_for_reason(score, comment, meta_review_scores[4])

            prompt = [
                {"role": "system", "content":"You are a meta-reviewer who will help me evaluate a paper. You have given a score and a comment. Now you need to provide a reason for the score."},
                {"role": "user", "content": \
                    "You are a meta-reviewer. One of your reviewers turned in feed-back in the form of text. You need to continue the assessment of the review. Answer each question below and return the answer in the corresponding JSON format:\n" + \
                    "\{\n" + \
                        "\"rating_reason\" :" + f"{rating_prompt}," + \
                        "\"precision_reason\" :" + f"{precision_prompt}," + \
                        "\"correctness_reason\" :" + f"{correctness_prompt}," + \
                        "\"recommendation_reason\" :" + f"{recommendation_prompt}," + \
                        "\"respectfulness_reason\" :" + f"{respectfulness_prompt}" + \
                    "\n\}"
                    },
            ]

            success = False
            num_trials = 0
            while not success and num_trials < 5:
                try:
                    answer = ask_chat_gpt(prompt, temperature=0.2*num_trials)["choices"][0]["message"]["content"]
                    answer = custom_json_loads(answer)
                    success = True
                except Exception as e:
                    print("Error: ", e)
                    try:
                        print("Retrying to reformat the results with chatGPT...")
                        reformat_prompt = [
                            {"role": "system", "content":"You are an assitant who will help me reformat the results to JSON format."},
                            {"role": "user", "content":"Reformat the preliminary results to this JSON format:\n" + \
                                "\{\n" + \
                                    "\"rating_reason\" : ...," + \
                                    "\"precision_reason\" : ...," + \
                                    "\"correctness_reason\" : ...," + \
                                    "\"recommendation_reason\" : ...," + \
                                    "\"respectfulness_reason\" : ..." + \
                                "\n\}\nPreliminary results: " + str(answer)}
                        ]
                        answer = ask_chat_gpt(reformat_prompt)["choices"][0]["message"]["content"]
                        answer = custom_json_loads(answer)
                        success = True
                    except:
                        print("Retrying to reformat the results with chatGPT failed...")
                        print("Retrying...")
                        num_trials += 1
                        success = False

            rating_reason = answer["rating_reason"]
            precision_reason = answer["precision_reason"]
            correctness_reason = answer["correctness_reason"]
            recommendation_reason = answer["recommendation_reason"]
            respectfulness_reason = answer["respectfulness_reason"]

            return [rating_reason, precision_reason, correctness_reason, recommendation_reason, respectfulness_reason]



                
        