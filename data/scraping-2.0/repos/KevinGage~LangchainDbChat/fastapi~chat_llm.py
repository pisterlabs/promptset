from langchain.llms import KoboldApiLLM
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate


class ChatLLM:
    def __init__(self):
        self.llm = KoboldApiLLM(
            endpoint="http://localhost:5002/api",
            temperature=0.1,
            max_context_length=2048,
            max_length=1024,
        )

        self.template = """[INST]
This QUERY was run
[QUERY]
{sql_query}
[/QUERY]

This was the DATA returned from the query
[DATA]
{data}
[/DATA]

Answer this QUESTION
[QUESTION]
{question}
[/QUESTION]
[/INST]
"""
        self.prompt = PromptTemplate(
            template=self.template, input_variables=["question", "sql_query", "data"]
        )

        self.llm_chain = LLMChain(prompt=self.prompt, llm=self.llm)

    def ask(self, question: str, sql_query: str, data: str):
        return self.llm_chain.run(question=question, sql_query=sql_query, data=data)
