import unittest
import os
from dotenv import load_dotenv
from langdiversity.models import OpenAIModel
from langdiversity.utils import DiversityCalculator, PromptSelection
from langdiversity.parser import extract_last_letters

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

class TestPromptSelection(unittest.TestCase):

    def setUp(self):
        self.openai_model = OpenAIModel(openai_api_key=openai_api_key, model="gpt-3.5-turbo", extractor=extract_last_letters)
        self.diversity_calculator = DiversityCalculator()
        self.prompt_selector = PromptSelection()

    def test_prompt_selection(self):
        ll_question = "Take the last letter of each word in \"Tal Evan Lesley Sidney\" and concatenate them."
        prompt = f"At the end, say 'the answer is [put the concatenated word here]'.\nQuestion: {ll_question}.\n "
        
        all_responses = []
        all_scores = []
        for _ in range(10):  # Repeating the process 10 times
            responses = self.openai_model.generate(prompt, count=5)
            diversity_scores = self.diversity_calculator.calculate(responses, measures=['entropy', 'gini'])
            all_responses.append(responses)
            all_scores.append(diversity_scores)
        
        selected_prompts_max = self.prompt_selector.find_extreme_scores(all_scores, measures=['entropy', 'gini'], selection_method='max')
        selected_prompts_min = self.prompt_selector.find_extreme_scores(all_scores, measures=['entropy', 'gini'], selection_method='min')
        
        # Shows all responses and scores
        for i, (responses, scores) in enumerate(zip(all_responses, all_scores)):
            print(f"Run {i+1}:")
            print("Responses:", responses)
            print("Scores:", scores)
            print("----------")
        
        print("Selected Prompts (Max):", selected_prompts_max)
        print("Selected Prompts (Min):", selected_prompts_min)
