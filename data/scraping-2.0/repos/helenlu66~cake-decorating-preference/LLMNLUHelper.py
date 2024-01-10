from pprint import pprint
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from ConfigUtil import get_args, load_experiment_config
from prompts import *
import os


class LLMNLUHelper:
    def __init__(self, prompts_setup:dict, task_setup:dict, api_key=os.environ['OPENAI_API_KEY'] if 'OPENAI_API_KEY' in os.environ else None) -> None:
        self.input_dict = {**prompts_setup, **task_setup}
        self.classifier = LLMChain(llm = OpenAI(openai_api_key=api_key, temperature=0), prompt=classification_prompt, output_key='classification')
        self.human_accept_classifier = LLMChain(llm = OpenAI(openai_api_key=api_key, temperature=0), prompt=classify_human_accept, output_key='classification')
        self.redirector = LLMChain(llm = OpenAI(openai_api_key=api_key, temperature=0), prompt=redirect_prompt, output_key='robot_response')
        self.constraints_extractor = LLMChain(llm = OpenAI(openai_api_key=api_key, temperature=0), prompt=constraints_extraction_prompt, output_key='constraints')
    
    def classify(self, robot_question:str, human_answer:str) -> bool:
        """takes in a human response and classify whether the ConstraintExtractor can handle this response

        Args:
            robot_question (str): the robot's question
            human_response (str): a human response to the robot's question

        Returns:
            either true or false
        """
        self.input_dict['robot_question'] = robot_question
        self.input_dict['human_answer'] = human_answer
        outputs = self.classifier(inputs=self.input_dict)
       
        if outputs['classification'].strip().lower() == 'yes':
            return True
        return False
    
    def classify_human_accept(self, robot_question:str, human_answer:str):
        """classify whether the human accepted the robot's proposal

        Args:
            robot_question (str): proposing a location
            human_answer (str): yes, left, right, up, down

        Returns:
            str
        """
        self.input_dict['robot_question'] = robot_question
        self.input_dict['human_answer'] = human_answer
        outputs = self.human_accept_classifier(inputs=self.input_dict)
       
        return outputs['classification'].strip().lower()
    
    def redirect(self, robot_question:str, human_answer:str) -> bool:
        """given a human answer that does not answer the robot question, respond in a way to redirect the human back to the task

        Args:
            robot_question (str): the robot's question
            human_response (str): a human response to the robot's question

        Returns:
            robot's response text
        """
        self.input_dict['robot_question'] = robot_question
        self.input_dict['human_answer'] = human_answer
        outputs = self.redirector(inputs=self.input_dict)
        return outputs['robot_response']
    
    def extract_constraints(self, robot_question:str, human_answer:str) -> list[str]:
        """takes in a human response and extracts a list of constraints

        Args:
            robot_question (str): the robot's question
            human_response (str): a human response to the robot's question

        Returns:
            list of constraints
        """
        self.input_dict['robot_question'] = robot_question
        self.input_dict['human_answer'] = human_answer
        outputs = self.constraints_extractor(inputs=self.input_dict)
        constraints_list = outputs['constraints'].split('\n')
        return constraints_list
        

if __name__=="__main__":
    args = get_args()
    exp_config = load_experiment_config('experiment_config.yaml')
    constraint_extractor = LLMNLUHelper(prompts_setup=prompts_setup, task_setup=exp_config['task_setup'], api_key=args.api_key if args.api_key else os.environ['OPENAI_API_KEY'])
    # test responses to general candle placements
    robot_question = "Where should I place the second candle?" + '\n'

    human_answer = input(robot_question)
    related = constraint_extractor.classify(robot_question=robot_question, human_answer=human_answer)
    while not related:
        robot_question = constraint_extractor.redirect(robot_question=robot_question, human_answer=human_answer)
        human_answer = input(robot_question)
        related = constraint_extractor.classify(robot_question=robot_question, human_answer=human_answer)
    
    human_intent = constraint_extractor.classify_human_accept(robot_question=robot_question, human_answer=human_answer)
    while not human_intent == 'accept':
        robot_question = "Is this a good location (You can say either yes, no, or move to the left, to the right, move up or move down)?" 
        human_answer = input(robot_question)
        human_intent = constraint_extractor.classify_human_accept(robot_question=robot_question, human_answer=human_answer)
            
    pprint(constraint_extractor.extract_constraints(robot_question=robot_question, human_answer=human_answer))
    
    pprint(constraint_extractor.redirect(robot_question='Where should I put the second candle?', human_answer='Um, oh.'))
    
    # test interpretation of "directly below"
    pprint(constraint_extractor.extract_constraints(robot_question='Where should I put the second candle?', human_answer='Put it directly below the first'))
    
    
    pprint(constraint_extractor.classify(robot_question='Where should I put the first candle?', human_answer='I do not know. Can you give me some example locations?'))
    pprint(constraint_extractor.redirect(robot_question='Where should I put the first candle?', human_answer='I do not know. Can you give me some example locations?'))
    
    pprint(constraint_extractor.classify(robot_question='Is this a good location?', human_answer='I do not know. Can you give me some example locations?'))
    pprint(constraint_extractor.redirect(robot_question='Is this a good location? You can say yes or to the left, right, up, down.', human_answer='I do not know. Can you give me some example locations?'))
    
    pprint(constraint_extractor.classify_human_accept(robot_question='Is this a good location?', human_answer='No, to the left.'))
    pprint(constraint_extractor.classify_human_accept(robot_question='Is this a good location?', human_answer='Yes.'))

    pprint(constraint_extractor.extract_constraints(robot_question='Where should I put the first candle?', human_answer='Put it on the left side of the cake.'))

