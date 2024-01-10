import os

from langchain.evaluation import load_evaluator, EvaluatorType
from langchain_community.chat_models.huggingface import ChatHuggingFace
from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint

"""
This lab will explore using built-in evaluator criteria, creating custom criteria, and using evaluators.

https://python.langchain.com/docs/guides/evaluation/string/criteria_eval_chain
"""

"""
Defining LLM, templates, chat_model, and prompt. No need to edit these.
"""
llm = HuggingFaceEndpoint(
    endpoint_url=os.environ['LLM_ENDPOINT'],
    task="text2text-generation",
    model_kwargs={
        "max_new_tokens": 200,
        "huggingfacehub_api_token": os.environ['HF_TOKEN'],
    }
)
chat_model = ChatHuggingFace(llm=llm)

textInput = """
<|system|>
You are a helpful AI that responds to questions concisely, if possible.</s>
<|user|>
{userInput}</s>
<|assistant|>
"""

"""
Your tasks for lab completion are below: 

-Define a prompt that will be evaluated by the built-in "depth" evaluator.
-Define your own custom criteria that will evaluate whether a response returns a mathematical value.
-Instantiate an evaluator that utilizes your custom criteria.
"""

# TODO: Write an input that will pass the built-in "depth" criteria.
"""In other words, write a question that gets an insightful or "deep" answer from the llm"""
depth_criteria_passing_query = ""

# This is a sample custom criteria that will evaluate whether a response contains spanish words.
# Do not edit this, use it as a template for the task below
sample_custom_criteria = {
    "spanish": "Does the output contain words from the spanish language?"
}

# TODO: create your own custom criteria that evaluates whether a response returns a mathematical value
your_custom_criteria = {
    "TODO"
}

"""The first 2 functions are responsible for evaluating llm responses. DO NOT EDIT THEM.
DO read through the functions along with their prints to the console to get an idea of what each function is doing.

You will be responsible for instantiating the evaluator in the 3rd function (custom_mathematical_evaluator).

The evaluator will return a VALUE of Y and SCORE of 1 if the answer meets the criteria."""
def built_in_depth_evaluator(query: str):

    # Initial response from LLM
    prediction = chat_model.invoke(textInput.format(userInput=query))

    # Instantiating an evaluator and using it to evaluate the response based on the built-in "depth" criteria
    evaluator = load_evaluator(EvaluatorType.CRITERIA, llm=chat_model, criteria="depth")
    eval_result = evaluator.evaluate_strings(prediction=prediction.content, input=query)

    # Print results
    print_results("DEPTH", query, prediction, eval_result)


def custom_spanish_evaluator(query: str):

    # Initial response from LLM
    prediction = chat_model.invoke(textInput.format(userInput=query))

    # Instantiating an evaluator and using it to evaluate the response based on the sample custom criteria
    evaluator = load_evaluator(EvaluatorType.CRITERIA, criteria=sample_custom_criteria, llm=chat_model)
    eval_result = evaluator.evaluate_strings(prediction=prediction, input=query)

    # Print results
    print_results("SPANISH", query, prediction, eval_result)


def custom_mathematical_evaluator(query: str):

    # Initial response from LLM
    prediction = chat_model.invoke(textInput.format(userInput=query))

    # TODO: instantiate an evaluator that uses your custom criteria, and evaluate the response
    """ Remember the examples of this in the methods above """
    evaluator = "TODO"
    eval_result = evaluator.evaluate_strings(prediction=prediction, input=query)

    # Print results
    print_results("MATHEMATICAL", query, prediction, eval_result)

    # This is just for testing
    return eval_result


"""This function will print the results of evaluations"""
def print_results(prompt: str, query: str, prediction, eval_result):

    print("\nPROMPT : ", prompt)
    print("INPUT: ", query)
    print("OUTPUT:", prediction.content)

    result = (eval_result["value"].replace(" ", ""))
    print("RESULT: ", result)
    score = (eval_result["score"])
    print("SCORE: ", score)

    if result == "Y" and score == 1:
        print("The output passes the " + prompt + " criteria")
    else:
        print("The output does not pass the " + prompt + " criteria")