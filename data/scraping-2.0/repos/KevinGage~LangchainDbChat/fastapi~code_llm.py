from langchain.llms import KoboldApiLLM
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate


class CodeLLM:
    def __init__(self, table_definitions: str):
        self.llm = KoboldApiLLM(
            endpoint="http://localhost:5001/api",
            temperature=0.1,
            max_context_length=2048,
            max_length=1024,
        )
        self.table_definitions = table_definitions
        self.template = """You are an AI programming assistant, utilizing the Deepseek Coder model, developed by Deepseek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer.
### Instruction:
Create a SQL query that can be used to answer this question: {question}

The query should respond with information other than ids.  For instance instead of returning id return email.

If using GROUP BY be sure to include all required columns

Do not use the LIMIT keyword because it is not supported in Microsoft SQL

Use these TABLE_DEFINITIONS to satisfy the database query.

TABLE_DEFINITIONS

{table_definitions}
### Response:
"""
        self.prompt = PromptTemplate(
            template=self.template, input_variables=["question", "table_definitions"]
        )

        self.llm_chain = LLMChain(prompt=self.prompt, llm=self.llm)

    def ask(self, question: str):
        return self.llm_chain.run(
            question=question, table_definitions=self.table_definitions
        )
