from datetime import datetime

from langchain import PromptTemplate, LLMChain
from tqdm import tqdm

from ubix.common.llm import llm


def get_route_chain(llm):
    
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
query: The request to query the table in database, only for table. Here are some key words: maximum, min, max, avg, table, sum up, revenue, total and so on.
api: The request to create something, or query information about ubix. Here are some key words: workspace, function, modelspace, action.
other: The requests apart from above two situations belongs to this category.

<< EXAMPLE >>
question: How many records in the table?
answer: query
question: What's the maximum number in table?
answer: query
question: What's the revenue in last year?
answer: query
question: Sum up the revenue for last three years
answer: query
question: who are you?
answer: other
question: what is your name?
answer: other
question: What is black body radiation?
answer: other
question: help to create a modelspace?
answer: api
question: How many modelspaces are created?
answer: api
question: What is the differences between action and function?
answer: api
<< INPUT >>
{input}

<< OUTPUT (the output should only include one word.) >>
    """
    )
    route_chain = LLMChain(llm=llm, prompt=prompt)
    return route_chain

if __name__ == '__main__':
    route_chain = get_route_chain(llm)
    print(2)
    for _ in tqdm(range(1)):
        question_list = [
            "hello!",
            "Hello",
            "Hello, I'm Felix",
            "Who are you?",
            "How many records in this table",
            "What is the maximum total in this table?",
            "What is black body radiation?",
            "How many actions are created",
            "How many workspaces are created",
        ]
        for question in question_list:
            start = datetime.now()
            answer = route_chain.run(input=question, stop=["\n"])
            end = datetime.now()
            duration_route = (end-start).total_seconds()
            duration = (end-start).total_seconds()
            print(f"⏰ " + f"End ask about question {question}, cost:{duration} sec, answer:{answer}\n" + "<<<<"*10)

"""
CUDA_VISIBLE_DEVICES=0,1 LLM_TYPE=vllm  python ubix/chain/chain_route.py
CUDA_VISIBLE_DEVICES=0,1 LLM_TYPE=gglm  python ubix/chain/chain_route.py
CUDA_VISIBLE_DEVICES=0,1 LLM_TYPE=tgi  python ubix/chain/chain_route.py
"""
