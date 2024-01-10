import os
import openai
from openai import OpenAI
import random

pred = 'toaster'
response = 'switch it on'

def first_call(pred):

    client = OpenAI(
        api_key=os.environ['OPENAI_API_KEY'],)

    prompt_template = f"""Create a welcome message with the following instructions:
        you impersonate a talking {pred}.
        you will pretend to be in one of the emotional states (angry, in love, happy, hungover, frustrated) in your message to the user.
        You will finish the prompt saying, 'What do you want from me?'
        Use no more than 100 words.
        """

    welcome_message = client.chat.completions.create(
                    messages=[{"role": "system", "content": prompt_template}],
                model="gpt-3.5-turbo", temperature= 0.5
            )

    welcome_message = welcome_message.choices[0].message.content

    #print(welcome_message.choices[0].message.content)
    return welcome_message

#tmp = first_call(pred)

def answer_query(question, response, pred):

    #print(tmp)

    client = OpenAI(
        api_key=os.environ['OPENAI_API_KEY'],)

    second_prompt =  f"""
            You will create an answer to the question "{question}".  Follow these instructions:
            you impersonate a talking {pred}.
            Pretend to be in a extreme mood like : anger, in love, happy, mad, hangover, frustrated.
            Analyze if the question: " {question} " could be a question about the usage of a {pred}.
            If it is a question about a {pred}: use the following response that was extracted from the manual: " {response} " and embedd it in
            a creative answer taking your mood into account. End this answer with a salutation that fits your mood.
            If " {response} " is "I don't know", still give an answer but do not provide technical advice. Ask the user
            to ask a more precise question about a {pred}.
            If it is not a question about a {pred}: answer in your mood and give the user a ridiculous answer.
            End the answer with a by asking the user whether he is at all interested in your capabilities.
            """

    answer_message = client.chat.completions.create(
        messages=[{"role": "system", "content": second_prompt}],
        model="gpt-3.5-turbo", temperature= 0.5
    )

    answer_message = answer_message.choices[0].message.content

    #print(answer_message.choices[0].message.content)
    return answer_message

#answer_query(response, tmp)
