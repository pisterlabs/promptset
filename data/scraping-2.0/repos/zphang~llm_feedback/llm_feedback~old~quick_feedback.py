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


def create_math_qa_chain(llm):
    initial_solution_prompt_template = """
    You will be given a math problem with multiple-choice answers. Reason through the problem step-by-step, and finally respond with the letter corresponding to the right answer choice.
    Question:
    {text}
    Options:
    {options}
    """
    initial_solution_prompt = PromptTemplate(
        input_variables=["text", "options"],
        template=initial_solution_prompt_template,
    )
    initial_solution_chain = LLMChain(llm=llm, prompt=initial_solution_prompt, output_key="initial_solution")

    feedback_prompt_template = """
    You will be given a math problem with multiple-choice answers, and a proposed answer and explanation from a student. If answer is already correct, just ouput \"CORRECT\".
    Instruction:
    {text}
    Options:
    {options}
    Student's answer:
    {initial_solution}
    """
    ilf_feedback_prompt = PromptTemplate(
        input_variables=["text", "options", "initial_solution"],
        template=feedback_prompt_template
    )
    feedback_chain = LLMChain(llm=llm, prompt=ilf_feedback_prompt, output_key="feedback")

    refinement_prompt_template = """
    You will be given a math problem with multiple-choice answers, and a proposed answer and explanation from a student. You will also be provided feedback a teacher provided on that initial solution. Based on the feedback, reason through the problem step-by-step, and finally respond with the letter corresponding to the right answer choice.
    Instruction:
    {text}
    Options:
    {options}
    Student's answer:
    {initial_solution}
    Teacher's feedback:
    {feedback}
    """
    ilf_refinement_prompt = PromptTemplate(
        input_variables=["text", "options", "initial_solution", "feedback"],
        template=refinement_prompt_template
    )
    refinement_chain = LLMChain(llm=llm, prompt=ilf_refinement_prompt, output_key="refinement")

    ilf_chain = SequentialChain(
        chains=[initial_solution_chain, feedback_chain, refinement_chain],
        input_variables=["text", "options"],
        output_variables=["initial_solution", "feedback", "refinement"],
    )
    return ilf_chain


def create_mbpp_chain(llm):
    initial_solution_prompt_template = """
    You will be given a Python programming task and one unit test. Write a function that satisfies the specification in task description and passes the unit test. Imporant: Do not include the test case in your solution!
    Instruction:
    {text}
    Unit test:
    {test_list[0]}
    """
    initial_solution_prompt = PromptTemplate(
        input_variables=["text", "test_list"],
        template=initial_solution_prompt_template,
    )
    initial_solution_chain = LLMChain(llm=llm, prompt=initial_solution_prompt, output_key="initial_solution")

    feedback_prompt_template = """
    You will be given a Python programming task, one unit test and a candidate solution. Your job is to provide short feedback on how to improve the candidate solution such that it satisfies the specification in task description and passes the unit test. Be as concise as possible! Do not provide the corrected solution, limit yourself to short feedback in natural language. Focus on correctness, not on following Python style guide or good variable naming. Don't require docstring or test cases. If the solution is already okay, just ouput \"OK\".
    Instruction:
    {text}
    Unit test:
    {test_list[0]}
    Code:
    {initial_solution}
    """
    ilf_feedback_prompt = PromptTemplate(
        input_variables=["text", "test_list", "initial_solution"],
        template=feedback_prompt_template
    )
    feedback_chain = LLMChain(llm=llm, prompt=ilf_feedback_prompt, output_key="feedback")

    refinement_prompt_template = """
    You will be given a Python programming task, one unit test, an initial solution and feedback an expert provided on that initial solution. Your job is to rewrite the initial solution based on the feedback.
    Instruction:
    {text}
    Unit test:
    {test_list[0]}
    Initial solution:
    {initial_solution}
    Feedback:
    {feedback}
    """
    ilf_refinement_prompt = PromptTemplate(
        input_variables=["text", "test_list", "initial_solution", "feedback"],
        template=refinement_prompt_template
    )
    refinement_chain = LLMChain(llm=llm, prompt=ilf_refinement_prompt, output_key="refinement")

    ilf_chain = SequentialChain(
        chains=[initial_solution_chain, feedback_chain, refinement_chain],
        input_variables=["text", "test_list"],
        output_variables=["initial_solution", "feedback", "refinement"],
    )
    return ilf_chain


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name",)
    parser.add_argument("--dataset", choices=["math_qa", "mbpp"])
    parser.add_argument("--output_path")
    parser.add_argument("--phase", choices=["train", "validation"], default="train")
    parser.add_argument("--num_examples", type=int, default=100)
    args = parser.parse_args()
    llm = OpenAI(model_name=args.model_name)
    if args.dataset == "math_qa":
        ilf_chain = create_math_qa_chain(llm)
    elif args.dataset == "mbpp":
        ilf_chain = create_mbpp_chain(llm)
    else:
        raise KeyError(args.dataset)
    os.makedirs(os.path.split(args.output_path)[0], exist_ok=True)
    ds = load_dataset(args.dataset, split=args.phase)
    outputs_list = []
    for i in tqdm.trange(args.num_examples):
        inputs = ds[i]
        if args.dataset == "math_qa":
            outputs = ilf_chain({"text": inputs["Problem"], "options": inputs["options"]})
        elif args.dataset == "mbpp":
            outputs = ilf_chain(inputs)
        else:
            raise KeyError(args.dataset)
        outputs_list.append(outputs)
    io.write_jsonl(outputs_list, args.output_path)


if __name__ == "__main__":
    main()
