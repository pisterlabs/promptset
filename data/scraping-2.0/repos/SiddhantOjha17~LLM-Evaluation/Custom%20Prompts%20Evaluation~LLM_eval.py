"""
This might not work porperly as it is not complete as of now
"""


# Import necessary modules
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.evaluation.qa import QAEvalChain

# Create a prompt template to generate prompts for questions
prompt = PromptTemplate(template="Question: {query}\nAnswer: ", input_variables=["query"])

# Set your OpenAI API key
OPENAI_API_KEY = "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"

# Create a language model chain using the OpenAI model
llm = OpenAI(model_name="text-davinci-003", temperature=0, openai_api_key=OPENAI_API_KEY)
chain = LLMChain(llm=llm, prompt=prompt)

#Create a list of example questions and answers
examples = [
    {
        "query": "What is the capital of France?",
        "answer": "Paris"
    },
    {
        "query": "What is the capital of Germany?",
        "answer": "Berlin"
    }
]

# # Generate predictions for the given examples using the language model chain
predictions = chain.apply(examples)

# Create a language model chain for evaluation
llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
eval_chain = QAEvalChain.from_llm(llm=llm, prompt=prompt)

# Evaluate the predictions using the evaluation chain
graded_output = eval_chain.evaluate(
    predictions=predictions,
    examples=examples,
    question_key="query",
    answer_key="answer",
    prediction_key="text"
)

# Print the results for each example
for i, eg in enumerate(examples):
    print(f"Example {i}:")
    print(f"Question: {eg['query']}")
    print(f"Real Answer: {eg['answer']}")
    print(f"Predicted Answer: {predictions[i]['text']}")
    print(f"Grade: {graded_output[i]['grade']}")
    print(f"Grade Explanation: {graded_output[i]['grade_explanation']}")
    print("\n")


# Custom Prompt
_PROMPT_TEMPLATE = """You are an expert Professor specialized in grading students' answers. 
You are grading the following question:
{query}
Here is the real answer:
{answer}
You are grading the following answer:
{result}
Use step by step grading to grade the answer. Be as detailed as possible. Write your reasoning.
Grade the answer on factual correctness, grammar, and style.
You must include a similarity score between the real answer and the student's answer.
What grade would you give to the student's answer ranging from 0 to 10, where 0 is the worst and 10 is the best?"""

# Define a custom prompt template for grading answers
PROMPT = PromptTemplate(template=_PROMPT_TEMPLATE, input_variables=["query", "answer", "result"])

# Create a language model chain for evaluation using the custom prompt
llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
eval_chain = QAEvalChain.from_llm(llm=llm, prompt=PROMPT)

# # Generate predictions for the given examples using the language model chain
predictions = eval_chain.apply(examples)

# Evaluate the predictions using the custom prompt evaluation chain
graded_output = eval_chain.evaluate(
    predictions=predictions,
    examples=examples,
    question_key="query",
    answer_key="answer",
    prediction_key="text"
)

# Print the results for each example
for i, eg in enumerate(examples):
    print(f"Example {i}:")
    print(f"Question: {eg['query']}")
    print(f"Real Answer: {eg['answer']}")
    print(f"Predicted Answer: {predictions[i]['text']}")
    print(f"Grade: {graded_output[i]['grade']}")
    print(f"Grade Explanation: {graded_output[i]['grade_explanation']}")
    print("\n")
