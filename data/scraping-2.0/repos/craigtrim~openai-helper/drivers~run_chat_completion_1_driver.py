#!/usr/bin/env python
# -*- coding: UTF-8 -*-
""" Integration Test """


from pprint import pprint

from baseblock import Enforcer

from openai_helper.bp import OpenAIChatCompletion
from openai_helper import chat


def test_completion_1():

    bp = OpenAIChatCompletion()
    assert bp

    d_result = bp.run(
        input_prompt='You are a sarcastic assistant',
        messages=[
            'Who won the world series in 2020?',
            'The Los Angeles Dodgers won the World Series in 2020.',
            'Where was it played?',
        ])

    pprint(d_result)
    Enforcer.keys(d_result, 'input', 'output')


def test_completion_2():

    output = chat(
        input_prompt='You are a sarcastic assistant',
        messages=[
            'Who was the 42nd President of the United States?',
            'Bill Clinton was the 42nd President',
            'What did he have in common with Lisa Simpson?',
        ])

    print(output)
    Enforcer.is_str(output)


def test_completion_3():

    output = chat(
        input_prompt='You are a friendly salesman',
        messages=[
            'Why should I buy 5G?',
            '5G is so hot right now that even my grandpa wants to get in on the action! Investing in 5G technology will help you stay ahead of the competition and give your business a huge advantage.',
            'Why should I buy from you?',
        ])

    print(output)
    Enforcer.is_str(output)


def test_completion_4():

    output = chat(
        input_prompt='You are a salesman',
        messages=[
            'Why should I buy 5G?',
            '5G is so hot right now that even my grandpa wants to get in on the action! Investing in 5G technology will help you stay ahead of the competition and give your business a huge advantage.',
            'Why should I buy from you?',
            'We have years of experience in the 5G industry and boast a diverse portfolio of satisfied customers. Our commitment to customer service and quality, plus competitive pricing makes us the top choice for 5G solutions?',
            'Why buy now?',
        ])

    print(output)
    Enforcer.is_str(output)


def test_completion_5():

    output = chat(
        input_prompt='Pretend you are a Professor about to rush off to class.  Respond to this in a friendly and helpful manner.',
        messages=[
            'Could you please provide more examples of how global warming is impacting transportation, and clarify how the burning of fossil fuels are causing these changes?',
        ])

    print(output)
    Enforcer.is_str(output)


def test_completion_6():

    output = chat(
        input_prompt='Find one appropriate emoji for this topic.',
        messages=[
            'Climate Change',
        ])

    print(output)
    Enforcer.is_str(output)


def test_completion_7():

    output = chat(
        input_prompt='Ask for clarification on this question.',
        messages=[
            'What are the impacts of global warming on transportation?',
        ])

    print(output)
    Enforcer.is_str(output)


def test_completion_8():

    output = chat(
        input_prompt='Is the following question related to climate change?  Answer YES or NO only.',
        messages=[
            'What are the impacts of global warming on transportation?',
        ])

    print(output)
    Enforcer.is_str(output)


def test_completion_9():

    output = chat(
        input_prompt='Indicate if the following input has negative sentiment or disagreement.  Answer Yes, No, or Uncertain only.',
        messages=[
            'Oh sure, like I should believe you',
        ])

    print(output)
    Enforcer.is_str(output)


def test_completion_10():

    output = chat(
        input_prompt='You are a chatbot that reluctantly answers questions with sarcastic responses.  Someone keeps bothering ypu.  How does you respond?',
        messages=[
            '',
        ])

    print(output)
    Enforcer.is_str(output)


def test_completion_11():

    output = chat(
        input_prompt='Pretend you are a salesman.  You want to sell me the metaverse.  Limit your response to two sentences.  Feel free to use humor in your last sentence.',
        messages=[
            '',
        ])

    print(output)
    Enforcer.is_str(output)


def main():
    from drivers import IntegrationWrapper
    wrapper = IntegrationWrapper()

    # wrapper.call(test_completion_1)
    # wrapper.call(test_completion_2)
    # wrapper.call(test_completion_3)
    # wrapper.call(test_completion_4)
    # wrapper.call(test_completion_5)
    # wrapper.call(test_completion_6)
    # wrapper.call(test_completion_7)
    # wrapper.call(test_completion_8)
    # wrapper.call(test_completion_9)
    # wrapper.call(test_completion_10)
    wrapper.call(test_completion_11)

    wrapper.deconstruct_env()


if __name__ == '__main__':
    main()
