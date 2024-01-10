from langchain.output_parsers import RegexParser
from langchain.prompts import PromptTemplate
from class_resolver import ClassResolver
import types

def default_map_rerank():
    template="""Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

In addition to giving an answer, also return a score of how fully it answered the user's question. This should be in the following format:

Question: [question here]
Helpful Answer: [answer here]
Score: [score between 0 and 100]

How to determine the score:
- Higher is a better answer
- Better responds fully to the asked question, with sufficient level of detail
- If you do not know the answer based on the context, that should be a score of 0
- Don't be overconfident!

Example #1

Context:
---------
Apples are red
---------
Question: what color are apples?
Helpful Answer: red
Score: 100

Example #2

Context:
---------
it was night and the witness forgot his glasses. he was not sure if it was a sports car or an suv
---------
Question: what type was the car?
Helpful Answer: a sports car or an suv
Score: 60

Example #3

Context:
---------
Pears are either red or orange
---------
Question: what color are apples?
Helpful Answer: This document does not answer the question
Score: 0

Begin!

Context:
---------
{context}
---------
Question: {question}
Helpful Answer:"""
    template = prompt_template(template, output_parser_v1())
    return template

def custom_map_rerank(template):
    template_="""
Begin!

Context:
---------
{context}
---------
Question: {question} Think through step by step.
Helpful Answer:"""
    template = template + template_
    template = prompt_template(template, output_parser_v1())
    return template

def template_resolver(template):
    class Base: pass

    func_list = []
    for name, val in globals().items():
        if isinstance(val, types.FunctionType):
            func_list.append(val)
    
    resolver = ClassResolver(func_list, base=Base)
    template = resolver.make(template)
    return template


def prompt_template(template, output_parser):
    prompt = PromptTemplate(template=template, input_variables=["context", "question"], output_parser=output_parser)
    return prompt
    
def output_parser_v1():
    output_parser = RegexParser(
        regex=r"(.*?)\nScore: (.*)",
        output_keys=["answer", "score"],
    )
    return output_parser

def output_parser_v2():
    output_parser = RegexParser(
        regex=r"(.*?)\nScore: (.*)\nReason: (.*)",
        output_keys=["answer", "score", "reason"],
    )
    return output_parser

def output_parser_v3():
    output_parser = RegexParser(
        regex=r"(.*?)\nHelpful Answer: (.*)\nScore: (.*)",
        output_keys=["reason", "answer", "score"],
    )
    return output_parser


def old_old_template():
    template="""### System:
You must answer the given question based on the given Input.

### User:
{question}

### Input:
{context}

### Response:"""
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    return prompt