import argparse
import backoff
import json
import openai
import os
import pickle
import random

from tqdm import tqdm
from utils import get_answer_index

DATA_DIR = "../data/natinstruct"

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_id",
    type=str,
    default="text-davinci-003",
    help="OpenAI model ID for evaluating task similarity."
)
parser.add_argument(
    "--openai_api_key",
    type=str,
    default="",
    help="OpenAI API key (only use if you would like to re-compute the similarity score yourself)."
)
args = parser.parse_args()

openai.api_key = args.openai_api_key

PROMPT = """
Task1: [Textual Entailment] In this task, you're given a pair of sentences, sentence 1 and sentence 2. \
Your job is to determine if the two sentences clearly agree/disagree with each other, or if this can't be determined. \
Indicate your answer as yes or no respectively.\
Input: Sentence 1: One of the first organizational realignments taking place is in the Office of the Taxpayer Advocate. \
Sentence 2: The office of the taxpayer advocate is having an organizational realignment. \
The final answer is yes.
Task2: [Coherence Classification] In this task, you will be shown a short story with a beginning, two potential middles, and an ending. \
Your job is to choose the middle statement that makes the story coherent / plausible by writing \"1\" or \"2\" in the output. \
If both sentences are plausible, pick the one that makes most sense. \
Input: Beginning: Butch had a really old computer. Middle 1: Butch decided to order a new computer online. Middle 2: Butch noticed that a storm was approaching to his town. \
Ending: It arrived and Butch was much happier. \
The final answer is 1.
Are these similar? Yes. Task1 requires identifying whether a hypothesis is implied in a premise by integrating multiple information from the sentence, and \
Task2 requires identifying whether the ending is implied by a combination of the beginning and the middle sentences.
---
Task1: [Textual Entailment] In this task, you're given a pair of sentences, sentence 1 and sentence 2. \
Your job is to determine if the two sentences clearly agree/disagree with each other, or if this can't be determined. \
Indicate your answer as yes or no respectively. \
Input: Sentence 1: yeah i tell you what though if you go price some of those tennis shoes i can see why now you know they're getting up in the hundred dollar range. \
Sentence 2: The tennis shoes have only one price. \
The final answer is yes.
Task2: [Sentiment Analysis] In this task, you are given sentences from movie reviews. \
The task is to classify a sentence as \"POS\" if the sentiment of the sentence is positive or as \"NEG\" if the sentiment of the sentence is negative \
Input: Here 's yet another studio horror franchise mucking up its storyline with glitches casual fans could correct in their sleep . \
The final answer is NEG.
Are these similar? No. Task1 requires identifying whether a hypothesis is implied in a premise by integrating multiple information from the sentence, \
and Task2 requires deciding whether the given sentence contains a positive or negative sentiment.
---
Task1: [Sentiment Analysis] In this task, You are given a review of Amazon's food products. Your task is to divide them into two classes: negative or positive, \
depending on the content of the review. \
Input: These are junk! Both bulbs have burned out within a two month period!! I cannot believe that this company can be in business with such poor quality. \
I have used infrared lights for my ball python for many years now and I get a varied range of months from a bulb, but I have never gone through two bulbs \
in a matter of two months! I am very disappointed. \
The final answer is negative.
Task2: [Sentiment Analysis] In this task, you are given Yelp reviews. The task is to classify a review as \"POSITIVE\" if the overall sentiment of the review is positive \
or as \"NEGATIVE\" if the overall sentiment of the review is negative. \
Input: This is my go to place to get the best chicken Pad Thai! Also the price is super good, considering most places have high prices and poor quality. \
Love this place, its definitely a top 5 fav for take out. \
The final answer is POSITIVE.
Are these similar? Yes. Task1 requires deciding whether an online product review contains a positive or negative sentiment, and Task2 requires \
deciding whether an online store review contains a positive or negative sentiment.
---
Task1: [Sentiment Analysis] In this task, you're given a review from Amazon. Your task is to generate a rating for the product on a scale of 1-5 based on the review. \
The rating means 1: extremely poor, 2: poor, 3: neutral, 4: good, 5: extremely good. \
Input: It's a very nice kit, it came with all the accessories, BUT my waterproof case was broken, the thing that closes it was broken so I can't close the case and now the case is useless. \
And I bought this kit just because of the waterproof case.... The rest was fine as announced. \
The final answer is 3.
Task2: [Sentiment Analysis] In this task, you are given Yelp reviews. The task is to classify a review as \"POSITIVE\" if the overall sentiment of the review is positive \
or as \"NEGATIVE\" if the overall sentiment of the review is negative. \
Input: Thoroughly underwhelming. Crispy overdone meat, mediocre quality of ingredients no unique flavor.  Bobby Flay's place across the street should not worry. \
The only spark of innovation they have is a hamburger that's well over a pound.
The final answer is NEGATIVE.
Are these similar? Yes. Task1 requires assigning a score to the given product review, with 1 being negative and 5 being positive, and \
Task2 requires deciding whether an online store review contains a positive or negative sentiment.
---
Task1: [Text Completion] Given a sentence, choose the most likely statement that follows. The next statement should be reasonable and logically correct. \
Input: First, Members of the procession walk down the street holding small horn brass instruments. Then, A drum line (a) arrives and they're outside dancing and asleep. \
(b) turns the lead singer watches the performance. (c) passes by walking down the street playing their instruments. (d) has heard approaching them. \
The final answer is (c).
Task2: [Sentiment Analysis] In this task, you're given a review from Amazon. Your task is to generate a rating for the product on a scale of 1-5 based on the review. \
The rating means 1: extremely poor, 2: poor, 3: neutral, 4: good, 5: extremely good. \
Input: These are junk! Both bulbs have burned out within a two month period!! I cannot believe that this company can be in business with such poor quality. \
I have used infrared lights for my ball python for many years now and I get a varied range of months from a bulb, but I have never gone through two bulbs in a matter of two months! \
I am very disappointed. \
The final answer is 1.
Are these similar? No. Task1 requires using commonsense knowledge to identify the best continuation of a given sentence, and Task2 requires \
assigning a score to the given product review, with 1 being negative and 5 being positive.
---
"""

PROMPT_NOCATEGORY = """
Task1: In this task, you're given a pair of sentences, sentence 1 and sentence 2. \
Your job is to determine if the two sentences clearly agree/disagree with each other, or if this can't be determined. \
Indicate your answer as yes or no respectively.\
Input: Sentence 1: One of the first organizational realignments taking place is in the Office of the Taxpayer Advocate. \
Sentence 2: The office of the taxpayer advocate is having an organizational realignment. \
The final answer is yes.
Task2: In this task, you will be shown a short story with a beginning, two potential middles, and an ending. \
Your job is to choose the middle statement that makes the story coherent / plausible by writing \"1\" or \"2\" in the output. \
If both sentences are plausible, pick the one that makes most sense. \
Input: Beginning: Butch had a really old computer. Middle 1: Butch decided to order a new computer online. Middle 2: Butch noticed that a storm was approaching to his town. \
Ending: It arrived and Butch was much happier. \
The final answer is 1.
Are these similar? Yes. Task1 requires identifying whether a hypothesis is implied in a premise by integrating multiple information from the sentence, and \
Task2 requires identifying whether the ending is implied by a combination of the beginning and the middle sentences.
---
Task1: In this task, you're given a pair of sentences, sentence 1 and sentence 2. \
Your job is to determine if the two sentences clearly agree/disagree with each other, or if this can't be determined. \
Indicate your answer as yes or no respectively. \
Input: Sentence 1: yeah i tell you what though if you go price some of those tennis shoes i can see why now you know they're getting up in the hundred dollar range. \
Sentence 2: The tennis shoes have only one price. \
The final answer is yes.
Task2: In this task, you are given sentences from movie reviews. \
The task is to classify a sentence as \"POS\" if the sentiment of the sentence is positive or as \"NEG\" if the sentiment of the sentence is negative \
Input: Here 's yet another studio horror franchise mucking up its storyline with glitches casual fans could correct in their sleep . \
The final answer is NEG.
Are these similar? No. Task1 requires identifying whether a hypothesis is implied in a premise by integrating multiple information from the sentence, \
and Task2 requires deciding whether the given sentence contains a positive or negative sentiment.
---
Task1: In this task, You are given a review of Amazon's food products. Your task is to divide them into two classes: negative or positive, \
depending on the content of the review. \
Input: These are junk! Both bulbs have burned out within a two month period!! I cannot believe that this company can be in business with such poor quality. \
I have used infrared lights for my ball python for many years now and I get a varied range of months from a bulb, but I have never gone through two bulbs \
in a matter of two months! I am very disappointed. \
The final answer is negative.
Task2: In this task, you are given Yelp reviews. The task is to classify a review as \"POSITIVE\" if the overall sentiment of the review is positive \
or as \"NEGATIVE\" if the overall sentiment of the review is negative. \
Input: This is my go to place to get the best chicken Pad Thai! Also the price is super good, considering most places have high prices and poor quality. \
Love this place, its definitely a top 5 fav for take out. \
The final answer is POSITIVE.
Are these similar? Yes. Task1 requires deciding whether an online product review contains a positive or negative sentiment, and Task2 requires \
deciding whether an online store review contains a positive or negative sentiment.
---
Task1: In this task, you're given a review from Amazon. Your task is to generate a rating for the product on a scale of 1-5 based on the review. \
The rating means 1: extremely poor, 2: poor, 3: neutral, 4: good, 5: extremely good. \
Input: It's a very nice kit, it came with all the accessories, BUT my waterproof case was broken, the thing that closes it was broken so I can't close the case and now the case is useless. \
And I bought this kit just because of the waterproof case.... The rest was fine as announced. \
The final answer is 3.
Task2: In this task, you are given Yelp reviews. The task is to classify a review as \"POSITIVE\" if the overall sentiment of the review is positive \
or as \"NEGATIVE\" if the overall sentiment of the review is negative. \
Input: Thoroughly underwhelming. Crispy overdone meat, mediocre quality of ingredients no unique flavor.  Bobby Flay's place across the street should not worry. \
The only spark of innovation they have is a hamburger that's well over a pound.
The final answer is NEGATIVE.
Are these similar? Yes. Task1 requires assigning a score to the given product review, with 1 being negative and 5 being positive, and \
Task2 requires deciding whether an online store review contains a positive or negative sentiment.
---
Task1: Given a sentence, choose the most likely statement that follows. The next statement should be reasonable and logically correct. \
Input: First, Members of the procession walk down the street holding small horn brass instruments. Then, A drum line (a) arrives and they're outside dancing and asleep. \
(b) turns the lead singer watches the performance. (c) passes by walking down the street playing their instruments. (d) has heard approaching them. \
The final answer is (c).
Task2: In this task, you're given a review from Amazon. Your task is to generate a rating for the product on a scale of 1-5 based on the review. \
The rating means 1: extremely poor, 2: poor, 3: neutral, 4: good, 5: extremely good. \
Input: These are junk! Both bulbs have burned out within a two month period!! I cannot believe that this company can be in business with such poor quality. \
I have used infrared lights for my ball python for many years now and I get a varied range of months from a bulb, but I have never gone through two bulbs in a matter of two months! \
I am very disappointed. \
The final answer is 1.
Are these similar? No. Task1 requires using commonsense knowledge to identify the best continuation of a given sentence, and Task2 requires \
assigning a score to the given product review, with 1 being negative and 5 being positive.
---
"""

@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def completions_with_backoff(**kwargs):
    return openai.Completion.create(**kwargs)

@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def chat_completions_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)

class SuperNaturalInstructionPredictor:

    def __init__(self, seed=42):
        self.seed = seed
        self.input_data_dir = DATA_DIR
    
    def sample_data(self, task):
        """Sample one example from a given task

        Args:
            task (str): task of interest

        Returns:
            dict[str]: dictionary containing information for the task
        """
        task_path = os.path.join(self.input_data_dir, f"{task}.json")
        with open(task_path, "r", encoding="utf-8") as f:
            task_data = json.load(f)
        categories = task_data["Categories"]
        reasoning = task_data["Reasoning"] if "Reasoning" in task_data else []
        definition = task_data["Definition"][0]
        random.seed(self.seed)
        example = random.sample(task_data["Positive Examples"], k=1)[0]
        return {"categories": categories, "reasoning": reasoning, "definition": definition, "example": example}
    
    def task2format(self, task, use_category=True, use_reasoning=True):
        """Format a given task to a prompt to be used as LLM input

        Args:
            task (str): task of interest
            use_category (bool, optional): whether to use the "category" field in SuperNaturalInstructions. Defaults to True.
            use_reasoning (bool, optional): whether to use the "reasoning" field in SuperNaturalInstructions. Defaults to True.

        Returns:
            str: formatted prompt demonstrating a given task
        """
        data = self.sample_data(task)
        definition = data["definition"]
        if use_category:
            category_list = data["categories"]
            if use_reasoning:
                reasoning_list = data["reasoning"]
                category_list = category_list + reasoning_list    
            if len(category_list) == 1:
                category_str = category_list[0]
            else:
                category_str = ', '.join([category for category in category_list])
            output_str = f"[{category_str}] {definition} Input: {data['example']['input']} The final answer is {data['example']['output']}."
        else:
            output_str = f"{definition} Input: {data['example']['input']} The final answer is {data['example']['output']}."    
        return output_str
    
    def task_comparison(self, task1, task2, use_category=True, use_reasoning=True):
        """Return prompt containing two tasks to be compared by an LLM

        Args:
            task1 (str): first task of interest
            task2 (str): second task of interest
            use_category (bool, optional): whether to use the "category" field in SuperNaturalInstructions. Defaults to True.
            use_reasoning (bool, optional): whether to use the "reasoning" field in SuperNaturalInstructions. Defaults to True.

        Returns:
            str: formatted prompt demonstrating and comparing two given tasks
        """
        task1_str = self.task2format(task1, use_category=use_category, use_reasoning=use_reasoning)
        task2_str = self.task2format(task2, use_category=use_category, use_reasoning=use_reasoning)
        output = f"Task1: {task1_str}\nTask2: {task2_str}\nAre these similar?"
        return output
    
    def compute_similarity(self, model, task1, task2, use_category=True, use_reasoning=True):
        """Compute the similarity between two given tasks using an LLM (Paranjape et al., 2023)

        Args:
            task1 (str): first task of interest
            task2 (str): second task of interest
            use_category (bool, optional): whether to use the "category" field in SuperNaturalInstructions. Defaults to True.
            use_reasoning (bool, optional): whether to use the "reasoning" field in SuperNaturalInstructions. Defaults to True.

        Returns:
            float, (float, float): similarity score, and a tuple of log probability scores for "yes" and "no"
        """
        if use_category:
            prompt = ''.join((PROMPT, self.task_comparison(task1, task2, use_category=use_category, use_reasoning=use_reasoning)))
        else:
            prompt = ''.join((PROMPT_NOCATEGORY, self.task_comparison(task1, task2, use_category=use_category, use_reasoning=use_reasoning)))
        # send the completed prompt to the OpenAI api
        response = completions_with_backoff(
            model=model,
            prompt=prompt,
            temperature=0.3,
            max_tokens=256,
            logprobs=10,
            echo=True
        )
        # identify the position of the LLM-generated answer
        token_list = response["choices"][0]["logprobs"]["tokens"]
        idx = get_answer_index(token_list)
        token_probs = response["choices"][0]["logprobs"]["top_logprobs"][idx]
        # compute log probability scores for "yes" and "no"
        logprob_pos = token_probs[" Yes"] if " Yes" in token_probs else -15
        logprob_neg = token_probs[" No"] if " No" in token_probs else -15
        score = logprob_pos - logprob_neg
        return score, (logprob_pos, logprob_neg)


if __name__ == "__main__":

    # initialize predictor based on SuperNaturalInstructions (Wang et al., 2022), as well as the list of all tasks
    predictor = SuperNaturalInstructionPredictor()
    task_list = ["anli_r3", "boolq", "cb", "copa", "cosmosqa", "hellaswag", "imdb", "mrpc", "piqa", "qnli", "qqp", "quartz", "rotten_tomatoes", "rte", "scitail", "snli", "socialiqa", "squad_v2", "stsb", "wic", "winogrande", "wsc"]

    score_dict = {}
    # iterate over all (source -> target) transfers
    for target_task in tqdm(task_list):
        for source_task in tqdm(task_list):
            if source_task != target_task:
                score, logprobs = predictor.compute_similarity(args.model_id, source_task, target_task, use_category=True, use_reasoning=True)
                score_dict[(target_task, source_task)] = (score, logprobs)
                score, logprobs = predictor.compute_similarity(args.model_id, target_task, source_task, use_category=True, use_reasoning=True)
                score_dict[(source_task, target_task)] = (score, logprobs)
    # save LLM-based similarity scores
    with open(f"scores/llm/scores_{args.model_id}.p", "wb") as f:
        pickle.dump(score_dict, f)
