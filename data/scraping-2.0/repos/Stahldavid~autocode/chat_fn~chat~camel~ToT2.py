import logging
import os
import re
from queue import Queue
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

# Load environment variables from .env file
load_dotenv()

# Access the API key from the environment variable
openai_api_key = os.getenv('OPENAI_API_KEY')

# Set up logging
logging.basicConfig(level=logging.INFO)

# Define constants for evaluation criteria
MAX_EVALUATIONS = 11
EVALUATION_SCORE_THRESHOLD = 7

decomposition_system_template = SystemMessagePromptTemplate.from_template(
    "You are an AI that decomposes code generation tasks. Please output the plan starting with the header 'Plan:' "
    "and then followed by a numbered list of steps. "
)
decomposition_human_template = HumanMessagePromptTemplate.from_template(
    "Decompose the code generation task in a list: {task}"
)

generation_system_template = SystemMessagePromptTemplate.from_template(
    "You are an AI that generates possible next steps for code."
)
generation_human_template = HumanMessagePromptTemplate.from_template(
    "Generate a possible next step for the code: {code}"
)

evaluation_system_template = SystemMessagePromptTemplate.from_template(
    "You are an AI that evaluates the quality of proposed code on a scale from 0 to 10. Just responde score : x"
)
evaluation_human_template = HumanMessagePromptTemplate.from_template(
    "Evaluate the quality of the proposed code on a scale from 0 to 10.: {code}"
)

search_system_template = SystemMessagePromptTemplate.from_template(
    "You are an AI that chooses the next step to take from proposed next steps."
)
search_human_template = HumanMessagePromptTemplate.from_template(
    "From the proposed next steps, choose the next step to take: {proposals}"
)

# Define a regular expression pattern for evaluation score
evaluation_pattern = r"score\s*:\s*(\d+)"

class EvaluationChain(LLMChain):
    def __call__(self, *args, **kwargs):
        try:
            result = super().__call__(*args, **kwargs)
            match = re.search(bytes(evaluation_pattern, 'utf-8'), bytes(result["text"], 'utf-8'))

            if match:
                score_value = match.group(1)
                int_result = int(score_value)
            else:
                print(f'No score value found in result: {result["text"]}')
                raise ValueError("No score value found in result!")
            return int_result

        except Exception as e:
            logging.error(f"Error during evaluation: {str(e)}")
            return None

decomposition_llm = ChatOpenAI(temperature=0.3, model_name='gpt-3.5-turbo')
generation_llm = ChatOpenAI(temperature=0.7, model_name='gpt-3.5-turbo')
evaluation_llm = ChatOpenAI(temperature=0.5, model_name='gpt-3.5-turbo')
search_llm = ChatOpenAI(temperature=0.2, model_name='gpt-3.5-turbo')

decomposition_chain = LLMChain(llm=decomposition_llm, prompt=ChatPromptTemplate.from_messages([decomposition_system_template, decomposition_human_template]))
generation_chain = LLMChain(llm=generation_llm, prompt=ChatPromptTemplate.from_messages([generation_system_template, generation_human_template]))
evaluation_chain = EvaluationChain(llm=evaluation_llm, prompt=ChatPromptTemplate.from_messages([evaluation_system_template, evaluation_human_template]))
search_chain = LLMChain(llm=search_llm, prompt=ChatPromptTemplate.from_messages([search_system_template, search_human_template]))

def validate_task(task):
    if not task or not isinstance(task, str) or len(task.strip()) == 0:
        logging.error("Invalid task. Please provide a non-empty string.")
        return False
    return True

def process_task(task, queue, stack, code, evaluation_counter):
    try:
        print(f"Decomposing task: {task}")
        decomposition = decomposition_chain(task)
        print(decomposition)

        # split the decomposition text by newline characters
        steps = decomposition["text"].split("\n")

        for step in steps:
            print(f"Generating next step for part: {step}")
            generation = generation_chain(step)

            if not generation["text"].strip():  # If no code is generated, raise ValueError
                raise ValueError("No code generated!")
            
            print(f"Evaluating generated code: {generation}")
            evaluation_score = evaluation_chain({"code": generation["text"]})  # Pass the generated code as an argument

            evaluation_counter += 1
            
            if evaluation_score >= EVALUATION_SCORE_THRESHOLD:
                print(f"Generated code meets the threshold. Added to queue and stack.")
                queue.put(generation)
                stack.append(generation)
            elif stack:
                print(f"Generated code doesn't meet the threshold. Reverting to last state in stack.")
                last_state = stack.pop()
                queue.put(last_state)
        
        if not queue.empty():
            print(f"Choosing next step from the proposed steps in the queue.")
            search = search_chain(queue.queue)
            code += search
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")

    queue.queue.clear()
    stack.clear()

    return code, evaluation_counter

def main():
    task = 'Write a variable impedance control code for force feedback using ros2, webots and webots_ros2'
    
    if not validate_task(task):
        return

    queue = Queue()
    stack = []
    
    queue.put(task)
    stack.append(task)

    code = """"""
    
    evaluation_counter = 0

    while not queue.empty():
        task = queue.get()
        
        code, evaluation_counter = process_task(task, queue, stack, code, evaluation_counter)

        if  evaluation_counter >= MAX_EVALUATIONS:
            break
    
    logging.info(f"Final code: {code}")

if __name__ == "__main__":
    main()
