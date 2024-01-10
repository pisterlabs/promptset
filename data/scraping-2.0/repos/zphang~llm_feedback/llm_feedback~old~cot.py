import os
import argparse
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
import warnings
from datasets import load_dataset
from evaluate import load
import numpy as np
from wasabi import color
import pyutils.io as io
import tqdm.auto as tqdm
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
import warnings
from datasets import load_dataset
from evaluate import load
import numpy as np
from wasabi import color
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
import datasets


def create_math_qa_chain(initial_llm, feedback_llm, refinement_llm):
    initial_solution_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="You are a math question-answering assistant."),
        HumanMessagePromptTemplate.from_template("""
The following is a math problem. Reason through the problem step-by-step, putting each separate reasoning step on a new numbered line (e.g. "Step 1. ") and finally respond with the right answer. Put the final answer letter on a single line.

Question:
{text}
Options:
{options}
    """.strip(), input_variables=["text", "options"])
    ])
    initial_solution_chain = LLMChain(llm=initial_llm, prompt=initial_solution_prompt, output_key="initial_solution")

    ilf_feedback_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="You are a math question-answering assistant."),
        HumanMessagePromptTemplate.from_template("""
The following is a proposed solution to a math question. There may be an error with the solution, or it may be correct. Go through each line and indicate if that line has an error (and explain what the error is) or no error ("OK."). After that, print "REFINE" one a single line if there are errors identified, or if there are no errors, print "CORRECT".

The output should look like:

    Step X: (Description of error)
    
    or 
    
    Step X: OK.

for each line.

Question:
{text}
Options:
{options}

Proposed solution:
{initial_solution}
    """.strip(), input_variables=["text", "options", "initial_solution"])
    ])
    feedback_chain = LLMChain(llm=feedback_llm, prompt=ilf_feedback_prompt, output_key="feedback")

    ilf_refinement_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="You are a math question-answering assistant."),
        HumanMessagePromptTemplate.from_template("""
You will be given a math problem with multiple-choice answers, and a proposed answer from a student. You will also be provided feedback a teacher provided on that initial solution. Based on the feedback, reason through the problem step-by-step, and finally respond with the letter corresponding to the right answer choice.
    
Instruction:
{text}
Options:
{options}
Student's answer:
{initial_solution}
Teacher's feedback:
{feedback}
    """.strip(), input_variables=["text", "options", "initial_solution", "feedback"])
    ])
    refinement_chain = LLMChain(llm=refinement_llm, prompt=ilf_refinement_prompt, output_key="refinement")

    ilf_chain = SequentialChain(
        chains=[initial_solution_chain, feedback_chain, refinement_chain],
        input_variables=["text", "options"],
        output_variables=["initial_solution", "feedback", "refinement"],
    )
    return ilf_chain


def create_mbpp_chain(initial_llm, feedback_llm, refinement_llm):
    initial_solution_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="You are a Python coding assistant."),
        HumanMessagePromptTemplate.from_template("""
You will be given a Python programming task and one unit test. Write a function that satisfies the specification in task description and passes the unit test. Prepend every line of code with a comment with the line number and a description of what the code does. Important: Do not include the test case in your solution!
For example:

# Line 1: set x to 0
x = 0
# Line 2: add 1 to x
x += 1

Instruction:
{text}
Unit test:
{test}
    """.strip(), input_variables=["text", "test"])
    ])
    initial_solution_chain = LLMChain(llm=initial_llm, prompt=initial_solution_prompt, output_key="initial_solution")

    ilf_feedback_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="You are a Python coding assistant."),
        HumanMessagePromptTemplate.from_template("""
The following is a proposed solution to a programming test. There may be an error with the solution, or it may be correct. Go through each line and indicate if that line has an error (and explain what the error is) or no error ("OK."), referring to the line numbers in the code comments. After that, print "REFINE" one a single line if there are errors identified, or if there are no errors, print "CORRECT".

The output should look like:

Line X: (Description of error)

or 

Line X: OK.

for each line.

Question:
{text}
Unit test:
{test}

Proposed solution:
{initial_solution}
    """.strip(), input_variables=["text", "test", "initial_solution"])
    ])
    feedback_chain = LLMChain(llm=feedback_llm, prompt=ilf_feedback_prompt, output_key="feedback")

    ilf_refinement_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="You are a Python coding assistant."),
        HumanMessagePromptTemplate.from_template("""
You will be given a Python programming task, one unit test, an initial solution and feedback an expert provided on that initial solution. Your job is to rewrite the initial solution based on the feedback.

Instruction:
{text}
Unit test:
{test}
Initial solution:
{initial_solution}
Expert feedback:
{feedback}
    """.strip(), input_variables=["text", "test", "initial_solution", "feedback"])
    ])
    refinement_chain = LLMChain(llm=refinement_llm, prompt=ilf_refinement_prompt, output_key="refinement")

    ilf_chain = SequentialChain(
        chains=[initial_solution_chain, feedback_chain, refinement_chain],
        input_variables=["text", "test"],
        output_variables=["initial_solution", "feedback", "refinement"],
    )
    return ilf_chain


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial_llm")
    parser.add_argument("--feedback_llm")
    parser.add_argument("--refinement_llm")
    parser.add_argument("--dataset", choices=["math_qa", "mbpp"])
    parser.add_argument("--output_path")
    parser.add_argument("--phase", choices=["train", "validation"], default="train")
    parser.add_argument("--num_examples", type=int, default=100)
    args = parser.parse_args()
    initial_llm = ChatOpenAI(model_name=args.initial_llm)
    feedback_llm = ChatOpenAI(model_name=args.feedback_llm)
    refinement_llm = ChatOpenAI(model_name=args.refinement_llm)
    if args.dataset == "math_qa":
        ilf_chain = create_math_qa_chain(
            initial_llm=initial_llm,
            feedback_llm=feedback_llm,
            refinement_llm=refinement_llm,
        )
    elif args.dataset == "mbpp":
        ilf_chain = create_mbpp_chain(
            initial_llm=initial_llm,
            feedback_llm=feedback_llm,
            refinement_llm=refinement_llm,
        )
    else:
        raise KeyError(args.dataset)
    os.makedirs(os.path.split(args.output_path)[0], exist_ok=True)
    ds = load_dataset(args.dataset, split=args.phase)
    outputs_list = []
    for i in tqdm.trange(args.num_examples):
        example = ds[i]
        if args.dataset == "math_qa":
            outputs = ilf_chain({"text": example["Problem"], "options": example["options"]})
        elif args.dataset == "mbpp":
            outputs = ilf_chain({"text": example["text"], "test": example["test_list"][0]})
        else:
            raise KeyError(args.dataset)
        outputs_list.append(outputs)
    io.write_jsonl(outputs_list, args.output_path)


if __name__ == "__main__":
    main()
