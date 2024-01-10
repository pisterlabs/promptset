from datetime import datetime

from langchain import PromptTemplate, LLMChain
from tqdm import tqdm

from ubix.common.llm import get_llm

prompt = PromptTemplate(
    input_variables=["input"],
    template="""
You are currently doing a classification task. Given a raw text input to a language model select the class best suited for \
the input. You will be given the names of the available class and a description of \
what the prompt is best suited for.

<< FORMATTING >>
Return the class name directly, the output should only include one word.

REMEMBER: "class" MUST be one of the candidate names specified below OR \
it can be "other" if the input is not well suited for any of the candidate prompts.


<< CLASS DEFINITION >>
query: the request to query the table in database, here are some key words: maximum, min, max, avg, table and so on.
other: general questions.
api: The request to create something, or query information about workspace, function, modelspace or action.


<< EXAMPLE >>
class: query, definition: How many records in the table?
class: query, definition: What's the maximum number in table
class: query, definition: Could you help to summary how many sells in this quarter.
class: other, definition: who are you? 
class: other, definition: what is your name?
class: other, definition: What is black body radiation?
class: api, definition: help to create a modelspace
class: api, definition: How many workspaces are created

<< INPUT >>
{input}

<< OUTPUT (the output should only include one word.) >>


"""
)

search_chain = LLMChain(llm=get_llm(), prompt=prompt)

for _ in tqdm(range(1)):
    question_list = [
        "Hello, I'm Felix",
        # "Who are you?",
        # "How many records in this table",
        "What is the maximum total in this table?",
        "What is black body radiation?",
        "Sum the total in table customer_invoice_item whose associated table customer_invoice_header's billing_organization is '00163E6A-610A-1EE9-91CE-E786A927A44A' and common field is postal_code . table customer_invoice_item don't contain field billing_organization"
    ]
    for question in question_list:
        start = datetime.now()
        answer = search_chain.run(input=question, stop="\n")
        end = datetime.now()
        duration_route = (end-start).total_seconds()
        duration = (end-start).total_seconds()
        print(f">>>>>"*10 + f"\nEnd ask about question {question}, cost:{duration} sec, answer:{answer}\n" + "<<<<"*10)

"""
CUDA_VISIBLE_DEVICES=3 python test/routeSample_v3.py
"""
