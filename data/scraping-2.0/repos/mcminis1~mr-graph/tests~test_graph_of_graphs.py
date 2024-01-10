import pytest
from mr_graph.graph import Graph


async def get_answer(user_question: str, temp=0):
    """get an answer from openai

    Args:
        user_question (str): question from the user

    Returns
    -------
    completion : str
        LLM completion
    """
    return "a random completion"

def format_answer(user_question:str, completion:str):
    """parse the answer

    Args:
        user_question (str): user question sent to the LLM. might be needed to determine formatting.
        completion (str): LLM completion.

    Returns
    -------
    answer : dict[str, str]
        LLM completion
    """
    answer = completion.strip(' \n')
    return answer


async def get_structured_answer(user_question: str):
    """get answer + structure it

    Args:
        user_question (str): user question sent to the LLM

    Returns
    -------
    answer : dict[str, str]
        LLM completion
    """
    llm = Graph(nodes=[get_answer, format_answer])
    q = llm.input(name='user_question')
    o1 = llm.get_answer(q)
    llm.outputs = llm.format_answer(q, o1)
    a = await llm(user_question=user_question)
    return a.answer


async def summarize_answers(answers: list[str], temp=0):
    """summarize answers

    Args:
        answers (str): answers sent to the LLM for summary

    Returns
    -------
    summary : dict[str, str]
        LLM completion
    """

    nl = "\n"
    prompt = f"""
    summarize the following text.

    {nl.join(answers)}
    """
    return prompt


async def get_summarized_q_and_a(questions: list[str]):
    """ask a bunch of questions, get answers, summarize them.
    
    Args:
        questions (list[str]): user questions sent to the LLM

    Returns
    -------
    summary : dict[str, str]
        LLM completion
    """
    llm = Graph(nodes=[get_structured_answer, summarize_answers])

    answers = llm.aggregator(name="answers")
    for question in questions:
        sa = llm.get_structured_answer(user_question=question)
        answers += sa.answer
    llm.outputs = llm.summarize_answers(answers=answers)

    v = await llm(answers)
    return v.summary


summary_text = """
    summarize the following text.

    a random completion
a random completion
a random completion
a random completion
a random completion
    """


@pytest.mark.asyncio
async def test_graph_of_graphs():
    questions = [
        'who is abraham lincoln?',
        'what did abraham lincoln do?',
        'when was abraham lincoln alive?',
        'where did abraham lincoln live?',
        'why is abraham lincoln famous?'
    ]
    r = await get_summarized_q_and_a(questions)
    assert r == summary_text



