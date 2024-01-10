from langchain.tools import tool
from langchain.tools import StructuredTool

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.callbacks import get_openai_callback
from langchain.utilities import DuckDuckGoSearchAPIWrapper
from langchain import hub

from repolya._log import logger_toolset


def stepback_question(_question):
    _examples = [
        {
            "input": "Could the members of The Police perform lawful arrests?",
            "output": "what can the members of The Police do?"
        },
        {
            "input": "Jan Sindel’s was born in what country?", 
            "output": "what is Jan Sindel’s personal history?"
        },
    ]
    # We now transform these to example messages
    _example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{input}"),
            ("ai", "{output}"),
        ]
    )
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=_example_prompt,
        examples=_examples,
    )
    _prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert at world knowledge. Your task is to step back and paraphrase a question to a more generic step-back question, which is easier to answer. Here are a few examples:"""),
        # Few shot examples
        few_shot_prompt,
        # New question
        ("user", "{question}"),
    ])
    _chain = _prompt | ChatOpenAI(temperature=0, model='gpt-4') | StrOutputParser()
    with get_openai_callback() as cb:
        _stepback = _chain.invoke({"question": _question})
        _token_cost = f"Tokens: {cb.total_tokens} = (Prompt {cb.prompt_tokens} + Completion {cb.completion_tokens}) Cost: ${format(cb.total_cost, '.5f')}"
    return _stepback, _token_cost

def tool_stepback_question():
    tool = StructuredTool.from_function(
        stepback_question,
        name="Stepback Question",
        description="Stepback a question to a more generic question.",
        verbose=True,
    )
    return tool


def stepback_ddg(_question):
    _stepback, _tc = stepback_question(_question)
    logger_toolset.info(f"'{_question}' --stepback--> '{_stepback}'")
    search = DuckDuckGoSearchAPIWrapper(max_results=3)
    def retriever(query):
        return search.run(query)
    response_prompt = hub.pull("langchain-ai/stepback-answer")
    chain = (
        {
            "normal_context": RunnableLambda(lambda x: x['question']) | retriever,
            "step_back_context": RunnableLambda(lambda x: x['stepback_question']) | retriever,
            "question": lambda x: x["question"]
        }
        | response_prompt
        | ChatOpenAI(temperature=0, model='gpt-3.5-turbo')
        | StrOutputParser()
    )
    with get_openai_callback() as cb:
        _re = chain.invoke({"question": _question, "stepback_question": _stepback})
        _token_cost = f"Tokens: {cb.total_tokens} = (Prompt {cb.prompt_tokens} + Completion {cb.completion_tokens}) Cost: ${format(cb.total_cost, '.5f')}"
    return _re, _token_cost

def tool_stepback_ddg():
    tool = StructuredTool.from_function(
        stepback_ddg,
        name="Stepback Question and Search DuckDuckGo",
        description="Stepback a question to a more generic question and search for answers with DuckDuckGo.",
        verbose=True,
    )
    return tool

