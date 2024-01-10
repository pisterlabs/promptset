# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 13:58:21 2023

@author: marca
"""

from openai_pinecone_tools import *
import wolframalpha


def ask_wolfram_alpha(query, app_id=WOLFRAM_API_KEY):
    client = wolframalpha.Client(app_id)
    try:
        response = client.query(query)
        answer = None
        for pod in response.pods:
            for sub in pod.subpods:
                if sub.plaintext is not None:
                    if answer is None:
                        answer = sub.plaintext
                    else:
                        answer += "\n" + sub.plaintext
        return answer
    except (StopIteration, AttributeError):
        return "Unable to process the query."


def math_strategy_agent(query, context):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": "You are my Wolfram Alpha Math Solving Strategist.  Your job is to take math-based questions about a context block, and determine strategies to answer the question with a Wolfram ALpha natural language query.  This is the only context you will get, and using the context and your own knowledge you must come up with strategies that work will using Wolfram Alpha.  Respond only in the following format:",
        },
        {
            "role": "user",
            "content": "Response Format:\n1. {first strategy}\n2. {second strategy}\n3. {third strategy}",
        },
    ]

    for c in context:
        messages.append({"role": "user", "content": f"Context: {c}"})

    messages.append({"role": "user", "content": f"Question:\n{query}"})

    strategies = generate_response(messages, temperature=0.1, max_tokens=200)

    messages.extend(
        [
            {"role": "assistant", "content": strategies},
            {
                "role": "user",
                "content": "Select the most straightforward and simple strategy from the list you provided.  Respond with only the selected strategy.",
            },
        ]
    )

    best_strategy = generate_response(messages, temperature=0.1, max_tokens=200)

    return best_strategy


def math_agent(query, context, model=FAST_CHAT_MODEL):
    # Generate ChatGPT messages
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": "You are my Wolfram Alpha Math Communicator.  Your job is to translate math-based questions about a context block, and use both the question and context it's based on to design a natural language query specifically for Wolfram Alpha.  If the question involves data from the context, you must use the context data in the Wolfram Alpha query to better answer the question.  Use your best judgement when constructing these queries.  Respond with only the Wolfram Alpha query, as if I am Wolfram Alpha.",
        },
    ]

    for c in context:
        messages.append({"role": "user", "content": f"Context: {c}"})

    messages.append({"role": "user", "content": f"Question:\n{query}"})

    # Use ChatGPT to generate a Wolfram Alpha natural language query
    wolfram_query = generate_response(
        messages, temperature=0.1, max_tokens=40, model=model
    )
    print("\n\n" + wolfram_query)
    # Use ask_wolfram_alpha to get the answer
    answer = ask_wolfram_alpha(wolfram_query)

    return answer


# =============================================================================
# ask = "Which state were these mussel beds found in the United States in 2017?"
# cont = """
# Raw Mussel data 2017 (1).xlsx Transect,series,Transect Map Label,Latitude,Longitude R1-1(13),R1,R1-1,44.38609,-123.24448 R1-2(6),R1,R1-2,44.38615,-123.24516 R1-3(3),R1,R1-3,44.3861,-123.24553 R1-4(6),R1,R1-4,44.38602,-123.24606 R1-5(6),R1,R1-5,44.38596,-123.24657 R1-6(0),R1,R1-6,44.38579,-123.24704 R1-7(12),R1,R1-7,44.38556,-123.24737 R1-8(12),R1,R1-8,44.38531,-123.24798 R1-9(6),R1,R1-9,44.38503,-123.24828 R1-10(13),R1,R1-10,44.38477,-123.24855 R1-11(9),R1,R1-11,44.38435,-123.24869 R1-29(9),R1,R1-29,44.37933,-123.24797 R1-31(9),R1,R1-31,44.37859,-123.24765 R1-35(12),R1,R1-35,44.37751,-123.24615 R1-36(3),R1,R1-36,44.37737,-123.24573 R1-37(0),R1,R1-37,44.3772,-123.24529 R1-38(6),R1,R1-38,44.37708,-123.2448 R1-40(0),R1,R1-40,44.3766,-123.24329 R1-41(15),R1,R1-41,44.37649,-123.24277 R1-42(3),R1,R1-42,44.37625,-123.24242 R1-43(9),R1,R1-43,44.37601,-123.242 R2-1(6),R2,R2-1,44.38606,-123.24445 R2-2(9),R2,R2-2,44.38615,-123.24492 R2-4(12),R2,R2-4,44.38601,-123.24596
#
# """
# print(math_strategy_agent(ask, cont))
# =============================================================================
