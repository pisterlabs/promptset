import pytest
from deepeval.metrics.answer_relevancy import AnswerRelevancyMetric
from deepeval.metrics.factual_consistency import FactualConsistencyMetric
from deepeval.test_case import LLMTestCase
from deepeval.run_test import assert_test

from langchain.chains import LLMChain

from llms.llms import get_ollama, create_prompt, get_answer


llm = get_ollama()


context="""
    Our co-founders, Andrew Bloye and David Johnson, started ClearXP in 2013 because they felt existing learning technologies
    were too restrictive. We built a flexible and powerful digital learning platform, making learning easier for users and 
    learning administration easier for organisations. Today, our award-winning  platform is helping  hundreds of thousands of learners experience better online learning. As the world changes around us, 
    our ethos of powerful, flexible, better technology ensures all needs will continue to be met. We are succeeding in making it easier for L&D Departments to make a difference. We were particularly proud in 2020 when the ClearXP and Coles Learning Hub were awarded winner of a highly sought after AITD Excellence Award.
    In the category of Best Use of Learning Technology, the judges were particularly impressed with our systems ability 
    to radically update and integrate learning into the culture of a large national organisation.

    ClearXP products include a powerful learning management system (LMS), learning experience platform (LXP), and a learning eco-system. 
    ClearXP provides Cohesive Learner Journey to design a user centric and seamless experience encompassing all forms of digital activities, 
    face to face workshops, video conferencing and professional development. It provides Automated Support that cleverly use data and AI provides 
    a better flow for learners and makes administration and moderation a breeze for support teams. It provides Relevant and Current Content that uses
    content authoring and visual workflows to allow teams to build user centric experiences and keep them up to date instantly. Lastly, it provides
    Actionable Data Insights which is a powerful reporting tool on the market captures more learning touchpoints and 
    delivers real insight into what actions can be taken to improve performance and drive higher achievement.
    """

template = """
[INST]<<SYS>> You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.<</SYS>> 
Question: {question} 
Context: {context} 
Answer: [/INST]
"""

prompt = create_prompt(template, context)

llm_chain = LLMChain(llm=llm, prompt=prompt)

minimum_score = 0.7  # Adjusting the minimum score threshold

def test_0():
    question = "When was ClearXP founded?"
    answer = get_answer(llm_chain, question)
    output = answer['text']

    factual_consistency_metric = FactualConsistencyMetric(minimum_score=minimum_score)
    test_case = LLMTestCase(input=question, actual_output=output, context=context)
    assert_test(test_case, [factual_consistency_metric])
    # metric.measure(output, context)
    # assert metric.is_successful(), metric.__class__.__name__ + " was unsuccessful."

def test_1():
    question = "What are ClearxP products?"
    answer = get_answer(llm_chain, question)
    output = answer['text']
   
    factual_consistency_metric = FactualConsistencyMetric(minimum_score=minimum_score)
    test_case = LLMTestCase(input=question, actual_output=output, context=context)
    assert_test(test_case, [factual_consistency_metric])
    # assert metric.is_successful(), metric.__class__.__name__ + " was unsuccessful."

def test_2():
    question = "What is the purpose of Cohesive Learner Journey?"
    answer = get_answer(llm_chain, question)
    output = answer['text']
   
    factual_consistency_metric = FactualConsistencyMetric(minimum_score=minimum_score)
    test_case = LLMTestCase(input=question, actual_output=output, context=context)
    assert_test(test_case, [factual_consistency_metric])
    # assert metric.is_successful(), metric.__class__.__name__ + " was unsuccessful."

def test_3():
    
    question = "What are ClearxP products?"
    answer = get_answer(llm_chain, question)
    output = answer['text']

    answer_relevancy_metric = AnswerRelevancyMetric(minimum_score=0.7)
    test_case = LLMTestCase(input=question, actual_output=output)
    assert_test(test_case, [answer_relevancy_metric])



def test_4():
    
    question = "What does ClearXP uses Automated Support for?"
    answer = get_answer(llm_chain, question)
    output = answer['text']

    answer_relevancy_metric = AnswerRelevancyMetric(minimum_score=0.7)
    test_case = LLMTestCase(input=question, actual_output=output)
    assert_test(test_case, [answer_relevancy_metric])


def test_5():
    
    question = "What is the purpose of Cohesive Learner Journey?"
    answer = get_answer(llm_chain, question)
    output = answer['text']

    answer_relevancy_metric = AnswerRelevancyMetric(minimum_score=0.7)
    test_case = LLMTestCase(input=question, actual_output=output)
    assert_test(test_case, [answer_relevancy_metric])
