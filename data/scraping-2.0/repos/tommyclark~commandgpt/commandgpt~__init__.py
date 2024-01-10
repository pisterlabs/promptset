from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import argparse
import subprocess
import sys

def get_input(text):
    return input(text)

def parse_arguments(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("-q", "--question", help="Command to look for")
    parser.add_argument("-r", "--run", help="Run the previously found command", action='store_true')
    return parser.parse_args(args)

def getPromptedLLMChain():
    llm = OpenAI(temperature=.7)
    template = """You will return a bash command which meets the criteria set out in the question.
        You will return only the command and no other text.
        Question: {text}
        Answer:
        """
    prompt_template = PromptTemplate(input_variables=["text"], template=template)
    return LLMChain(llm, prompt_template)

def main(args):
    answer = ''
    if args.question:
        chain = getPromptedLLMChain()
        answer = chain.run(args.question)
        if not args.run:
            print(answer)
        if args.run:
            isSure = get_input('Are you sure you want to run the following command? (y/n)\n' + answer + '\n').lower().strip() == 'y'
            if isSure:
                return_code = subprocess.call(answer, shell=True)

def runApplication():
    args = parse_arguments(sys.argv[1:])
    main(args)
