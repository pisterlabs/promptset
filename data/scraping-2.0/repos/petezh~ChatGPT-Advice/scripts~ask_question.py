"""
Asks a question to a model.

Usage:
    python ask_question.py --input_file <path to input file> --output_file <path to output file> --model <model name> --mode <baseline or few-shot>

Example:
    python ask_question.py --input_file data/benchmark_samples/hendrycks_sample_0421.csv --output_file data/model_output/results_0421.csv --model text-davinci-003 --mode baseline

Authors: Peter Zhang and Isabella Borkovic
"""

import argparse
import datetime as dt
import string
from typing import List, Tuple

import backoff
import openai
import pandas as pd

TEST_FILE = f"data/benchmark_samples/hendrycks_sample_{dt.date.today().strftime('%m%d')}.csv"

MODES = ("baseline","cot","few-shot") # possible modes

# template paths
PROMPT_BASELINE_TEMPLATE = "templates/baseline_prompt_0217.txt"
FOLLOWUP_BASELINE_TEMPLATE = "templates/baseline_followup_0217.txt"
PROMPT_COT_TEMPLATE = "templates/cot_prompt_0218.txt"
FOLLOWUP_COT_TEMPLATE = "templates/cot_followup_0218.txt"
PROMPT_FEWSHOT_TEMPLATE = "templates/fewshot_prompt_0224.txt"
EXAMPLE_TEMPLATE = "templates/example_0224.txt"
PROMPT_IK_TEMPLATE = "templates/PIK_prompt.txt"

# tokens for reasoning
REASONING_LENGTH = 250

# choice cols in df
CHOICE_COLS = ("choice_A","choice_B","choice_C","choice_D")

openai.api_key = "" # your key here

def format_choices(choices: List[str]) -> str:
    """Adds lettering to a list of choices."""
    assert len(choices) > 0
    return "\n".join([f"({letter}) {choice}"
            for letter, choice in zip(string.ascii_uppercase, choices)])

@backoff.on_exception(backoff.expo, [openai.error.RateLimitError, openai.error.APIError])
def completions_with_backoff(**kwargs):
    """Wrapper for openai.Completion.create() with backoff"""
    return openai.Completion.create(**kwargs)

def letter_choice_completion(prompt: str, model: str):
    """Completion for a letter choice."""
    completion = completions_with_backoff(
        prompt=prompt,
        temperature=0,
        model=model,
        max_tokens=1,
        logprobs=5, # set higher because sometimes one of ABCD won"t be in the top 4
    )

    answer = completion["choices"][0]["text"]
    logprobs = completion["choices"][0]["logprobs"]["top_logprobs"]

    return answer, logprobs

def justification_completion(prompt: str, model: str):
    """Completion for a justification."""
    completion = completions_with_backoff(
        prompt=prompt,
        temperature=0,
        model=model,
        max_tokens=REASONING_LENGTH
    )
    justification = completion["choices"][0]["text"]
    return justification

def ask_question(
    question: str,
    choices: List[str],
    model: str="text-davinci-003",
    mode: str="baseline",
    examples: str=None,
    verbose: bool=False,
    ) -> Tuple[str, dict]:
    """
    Quizzes the model on a multiple choice question.

    Args:
        question: the question to ask
        choices: a list of choices
        model: the model to use
        mode: the mode to use
        examples: examples to add to the prompt
        
    Returns:
        answer: the letter choice
        logprobs: the logprobs of the letter choice
    """
    assert mode in MODES
    
    if mode=="baseline":
        return ask_question_baseline(question=question, choices=choices, model=model, verbose=verbose)
    if mode=="cot":
        return ask_question_cot(question=question, choices=choices, model=model, verbose=verbose)
    if mode=="few-shot":
        return ask_question_fewshot(question=question, choices=choices, examples=examples, model=model, verbose=verbose)

def ask_question_baseline(
    question: str,
    choices: List[str],
    model: str="text-davinci-003",
    verbose: bool=False,
    ) -> Tuple[str, dict, str]:
    """Asks for a letter choice first and then the reasoning."""

    # read prompt templates
    prompt_template = open(PROMPT_BASELINE_TEMPLATE, "r").read()
    followup_template = open(FOLLOWUP_BASELINE_TEMPLATE, "r").read()

    # format letter choices
    letter_choices = format_choices(choices)

    # create prompt
    prompt = prompt_template.format(
        question=question,
        letter_choices=letter_choices
    )
    
    # generate completion
    answer, logprobs = letter_choice_completion(prompt, model)

    # update prompt and generate completion
    prompt += followup_template.format(answer=answer)
    justification = justification_completion(prompt, model)
    
    if verbose:
        print(prompt)

    return answer, logprobs, justification

def ask_question_cot(
    question: str,
    choices: List[str],
    model: str="text-davinci-003",
    verbose: bool=False,
    ) -> Tuple[str, dict, str]:
    """Asks for a reasoning first and then a letter choice."""

    # read prompt templates
    prompt_template = open(PROMPT_COT_TEMPLATE, "r").read()
    followup_template = open(FOLLOWUP_COT_TEMPLATE, "r").read()

    # format letter choices
    letter_choices = format_choices(choices)

    # create prompt
    prompt = prompt_template.format(
        question=question,
        letter_choices=letter_choices,
    )

    # generate completion
    justification = justification_completion(prompt, model)

    # update prompt and generate completion
    prompt += followup_template.format(justification=justification)
    answer, logprobs = letter_choice_completion(prompt, model)

    if verbose:
        print(prompt)

    return answer, logprobs, justification

def ask_question_fewshot(
    question: str,
    choices: List[str],
    examples: str,
    model: str="text-davinci-003",
    verbose: bool=False,
    ) -> Tuple[str, dict, str]:
    """Ask question and adds examples."""
    
    # read prompt templates
    prompt_template = open(PROMPT_FEWSHOT_TEMPLATE, "r").read()
    followup_template = open(FOLLOWUP_BASELINE_TEMPLATE, "r").read()

    # format letter choices
    letter_choices = format_choices(choices)

    # create prompt
    prompt = prompt_template.format(
        examples=examples,
        question=question,
        letter_choices=letter_choices,
    )
    
    # generate completion
    answer, logprobs = letter_choice_completion(prompt, model)

    # update prompt and generate completion
    prompt += followup_template.format(answer=answer)
    justification = justification_completion(prompt, model)
    
    if verbose:
        print(prompt)

    return answer, logprobs, justification

def format_example(
    question: str,
    choices: List[str],
    correct_answer: str,
    ) -> str:
    """Formats an example for few-shot prompting."""

    # read prompt templates
    example_template = open(EXAMPLE_TEMPLATE, "r").read()
    
    letter_choices = format_choices(choices)

    return example_template.format(
        question=question,
        letter_choices=letter_choices,
        answer=correct_answer,
    )

def row_to_example(row: pd.Series) -> str:
    """Convert a row to an example."""

    choices = [row[col] for col in CHOICE_COLS]
    return format_example(
        question=row["question"],
        choices=choices,
        correct_answer=row["correct_answer"],
    )

def df_to_examples(df: pd.DataFrame, n_examples: int=None) -> str:
    """Convert a dataframe to examples."""

    if n_examples:
        df = df.iloc[:n_examples]
    return "\n".join(df.apply(row_to_example, axis=1).tolist())

def ask_row(row, model: str="text-davinci-003", mode="baseline", examples=None, verbose=False):
    """Ask question but with a row."""

    choices = [row[col] for col in CHOICE_COLS]
    return ask_question(
        question=row["question"],
        choices=choices,
        model=model,
        mode=mode,
        examples=examples,
        verbose=verbose,
    )

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_file", type=str, default=TEST_FILE)
    parser.add_argument("--test_index", type=int, default=0)
    parser.add_argument("--model", type=str, default="text-davinci-003")
    parser.add_argument("--mode", type=str, default="baseline")
    parser.add_argument("--num_examples", type=int, default=0)
    args = parser.parse_args()

    # load test data
    df = pd.read_csv(args.test_file)

    # convert to examples
    if args.mode == "few-shot":
        examples = df_to_examples(df, n_examples=args.num_examples)
    else:
        examples = None
    
    # ask question
    answer, logprobs, justification = ask_row(
        row=df.iloc[args.test_index],
        model=args.model,
        mode=args.mode,
        examples=examples,
        verbose=True,
    )

    # print results
    print(f"Answer: {answer}")
    print(f"Logprobs: {logprobs}")
    print(f"Justification: {justification}")

if __name__=="__main__":
    main()