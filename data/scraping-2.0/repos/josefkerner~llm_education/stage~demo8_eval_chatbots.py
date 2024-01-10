from typing import List, Dict, Tuple
from agents.vectara_agent import VectaraAgent
from model.anthropic.llm_model import AnthropicModel
import json
from prompts.prompt_templates import EVALUATE_LLM_ANSWER_PROMPT
class ChatbotEvaluator:
    def __init__(self):
        self.manager = VectaraAgent()
        self.qa_pairs = self.load_question_answer_pairs(file_path='data/qa_pairs.jsonl')

    def evaluate(self, qa_pairs: List[Tuple[str, str]]):
        '''
        Evaluates the chatbot
        :param qa_pairs:
        :return:
        '''
        self.agent = self.manager.get_qa_agent(corpus_id=str(8))
        scores = self.get_rouge_score(qa_pairs=qa_pairs)
        print(scores)

    def get_rouge_score(self, qa_pairs: List[Tuple[str, str]]):
        '''
        Gets ROUGE score
        :param qa_pairs:
        :return:
        '''
        scores = []
        for question, answer in qa_pairs:
            response = self.agent({"query": question})
            #parse response json
            actual_answer = json.loads(response['result'])['answer']
            score = self.get_rouge_score_for_pair(actual_answer, answer)
            print(score)
            response = self.eval_answer_llm(question=question,
                                            model_answer=actual_answer,
                                            expected_answer=answer)
            print(response)
            scores.append(score)
        #average the scores
        avg_scores = {}
        for key in scores[0].keys():
            avg_scores[key] = sum([score[key] for score in scores])/len(scores)
        return scores

    def eval_answer_llm(self,question: str, model_answer: str, expected_answer: str):
        '''
        Evaluates the answer with the LLM Anthropic model
        :param model_answer:
        :param expected_answer:
        :return:
        '''

        model = AnthropicModel(cfg={
            'model_name': 'claude-2.1',
            'max_output_tokens': 300
        })
        prompt= EVALUATE_LLM_ANSWER_PROMPT.format(question=question,
                                                  answer=model_answer,
                                                  expected_answer=expected_answer)

        answer = model.generate(prompts=[prompt])
        return answer[0]



    def get_rouge_score_for_pair(self, actual_answer: str, expected_answer: str) -> Dict[str, float]:
        '''
        Gets ROUGE score for a pair of answers
        :param actual_answer:
        :param expected_answer:
        :return:
        '''
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
        print('actual answer', actual_answer)
        print('expected answer', expected_answer)
        scores = scorer.score(actual_answer, expected_answer)
        #score object to dict
        scores = {key: value.fmeasure for key, value in scores.items()}
        return scores


    def load_question_answer_pairs(self, file_path: str) -> List[Tuple[str, str]]:
        '''
        Loads question answer pairs from a JSONL file
        :param file_path:
        :return:
        '''
        qa_pairs = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                qa_pair = json.loads(line)
                qa_pairs.append((qa_pair['question'], qa_pair['answer']))
        return qa_pairs

if __name__ == "__main__":
    evaluator = ChatbotEvaluator()
    evaluator.evaluate(qa_pairs=evaluator.qa_pairs)