import wandb
import os
import re
import math
import numexpr
import argparse
from tqdm import tqdm
from typing import List, Union
from pydantic import BaseModel, Field, validator

from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.output_parsers import OutputFixingParser

from langchain.callbacks import get_openai_callback
from wandb.integration.langchain import WandbTracer
from langchain.callbacks import wandb_tracing_enabled

# TODO: remove this
from dotenv import load_dotenv
load_dotenv("/Users/ayushthakur/integrations/llm-eval/apis.env")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


def get_args():
    parser = argparse.ArgumentParser(description="Train image classification model.")
    parser.add_argument(
        "--sweep_file", type=str, default="configs/maths_sweeps.yaml", help="sweep yaml configuration"
    )
    parser.add_argument(
        "--prompt_template_file", type=str, help="prompt template for the LLM"
    )
    parser.add_argument(
        "--temperature", type=float, default=0, help="temperature for sampling"
    )
    parser.add_argument(
        "--llm_model_name", type=str, default="gpt-4", help="LLM model name",
    )
    parser.add_argument(
        "--wandb_project_name",
        type=str,
        default="llm-eval-sweep",
        help="wandb project name",
    )

    return parser.parse_args()


def is_valid_expression(expression):
    # Remove all white spaces from the expression
    expression = re.sub(r'\s', '', expression)

    # Check if the expression contains invalid characters
    if not re.match(r'^[\d+\-*/().\[\]{}^√]+$', expression):
        return False

    # Validate brackets using a stack
    stack = []
    opening_brackets = {'(', '[', '{'}
    closing_brackets = {')', ']', '}'}
    bracket_pairs = {'(': ')', '[': ']', '{': '}'}

    for char in expression:
        if char in opening_brackets:
            stack.append(char)
        elif char in closing_brackets:
            if not stack or bracket_pairs[stack.pop()] != char:
                return False

    if stack:
        return False

    return True


def correct_expression(expr: str) -> str:
    expr = expr.replace(" ", "")
    expr = expr.replace("[", "(")
    expr = expr.replace("]", ")")
    expr = expr.replace("{", "(")
    expr = expr.replace("}", ")")
    expr = expr.replace("^", "**")
    
    return expr


def evaluate_expr(expr: str) -> str:
    local_dict = {"pi": math.pi, "e": math.e}

    if is_valid_expression(expr):
        try:
            expr = correct_expression(expr)
            output = str(
                numexpr.evaluate(
                    expr.strip(),
                    global_dict={},  # restrict access to globals
                    local_dict=local_dict,  # add common mathematical functions
                )
            )
            return float(output)
        except:
            return None
    else:
        return None

expression_path = "data/maths/expressions.txt"

def load_chain(llm, prompt_template):
    chain = LLMChain(
        llm=llm,
        prompt=prompt_template,
    )
    return chain


# Define your desired data structure.
class Result(BaseModel):
    result: float = Field(description="Computation result returned by the LLM")
        
parser = PydanticOutputParser(pydantic_object=Result)
fixing_parser = OutputFixingParser.from_llm(parser=parser, llm=ChatOpenAI())


def accuracy_score(true_predictions: List[str], pred_predictions: List[str]):
    total = len(true_predictions)
    correct = 0
    for true, pred in zip(true_predictions, pred_predictions):
        if true == pred:
            correct += 1
    return correct / total


def main(args: argparse.Namespace):
    # Initialize wandb
    wandb.init(project=args.wandb_project_name, config=vars(args))

    # Load the expressions to evaluate
    with open(expression_path, "r") as f:
        expressions = f.readlines()
        print("Total expressions loaded for evaluation:", len(expressions))

    # Load the prompt template
    # The template file should contain the same input variables.
    PROMPT = PromptTemplate.from_file(
        template_file=args.prompt_template_file,
        input_variables=["question"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    # Load LLM
    if args.llm_model_name == "text-davinci-003":
        llm = OpenAI(temperature=args.temperature, model_name=args.llm_model_name)
    else:
        llm = ChatOpenAI(temperature=args.temperature, model_name=args.llm_model_name)
    # Load chain
    chain = load_chain(llm=llm, prompt_template=PROMPT)

    # Initialize W&B Table
    expression_table = wandb.Table(columns=[
        "expression", "true_result", "pred_result",
        "LLM Prompt Tokens", "LLM Completion Tokens", "LLM Total Tokens", "LLM Total Cost (USD)",
        "Parsing Prompt Tokens", "Parsing Compeltion Tokens", "Parsing Total Tokens", "Parsing Total Cost (USD)"
    ])

    total_tokens = 0
    total_cost = 0
    total_parsing_tokens = 0
    total_parsing_cost = 0

    pred_predictions = []
    true_predictions = []

    # Get answer for each expression
    for expression in tqdm(expressions):
        with wandb_tracing_enabled():
            with get_openai_callback() as output_cb:
                output = chain.run(expression)

            # Parse the output
            with get_openai_callback() as parsing_cb:
                pred_result = fixing_parser.parse(output)
                pred_result = pred_result.result

        # Evaluate the expression
        true_result = evaluate_expr(expression)

        # Log the results to W&B Table
        expression_table.add_data(
            expression, true_result, pred_result,
            output_cb.prompt_tokens, output_cb.completion_tokens, output_cb.total_tokens, output_cb.total_cost,
            parsing_cb.prompt_tokens, parsing_cb.completion_tokens, parsing_cb.total_tokens, parsing_cb.total_cost,
        )

        # Update the total tokens and cost
        total_tokens += output_cb.total_tokens
        total_cost += output_cb.total_cost
        total_parsing_tokens += parsing_cb.total_tokens
        total_parsing_cost += parsing_cb.total_cost

        # Append the predictions
        pred_predictions.append(pred_result)
        true_predictions.append(true_result)

    # Evaluate the predictions
    accuracy = accuracy_score(true_predictions, pred_predictions)
    print("Accuracy:", accuracy)

    # Log the results to W&B
    wandb.log({
        "accuracy": accuracy*100,
        "total_tokens": total_tokens,
        "total_cost": total_cost,
        "total_parsing_tokens": total_parsing_tokens,
        "total_parsing_cost": total_parsing_cost,
    }, commit=False)

    # Log the table to W&B
    wandb.log({"expression_table": expression_table})


if __name__ == "__main__":
    args = get_args()
    print(args)

    main(args)
