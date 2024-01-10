from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from context_store.fql import FQLEngine
from context_store.llm_logger import llm_logger
from llm.llm_factory import LLMFactory


class DataGatherer:
    fql_gen_prompt = '''Given a user query, you have to identify what information is needed to answer the query.
Include reason on why you need the information to answer the question. If the question does not require any information, return "None" as output.

Example:
Input: What is the savings rate for August?
Reason: To answer the question, I need to know the savings rate for August.
Output: savings rate for august

Input: What is my total savings across all bank accounts?
Reason: To answer the question, I need to know the total savings across all bank accounts.
Output: total savings across all bank accounts

Input: How much interest have I earned on my savings over the past year for accountA?
Reason: To answer the question, I need to know the interest earned on savings over the past year for accountA.
Output: interest earned on savings over the past year for accountA

Input: What is my highest and lowest savings month so far this year?
Reason: To answer the question, I need to know the highest and lowest savings month so far this year.
Output: highest and lowest savings month so far this year

Now you have to give reason and output for the following input:
Input: {user_query}'''

    def __init__(self):
        self.fql_engine = FQLEngine()
        self._llm = LLMFactory().get_chat_llm()
        fql_gen_template = PromptTemplate(
            input_variables=["user_query"],
            template=self.fql_gen_prompt,
        )
        self.fql_gen_chain = LLMChain(prompt=fql_gen_template, llm=self._llm, verbose=False)
        pass

    # in order to answer a user_query we need to gather different types
    # of data from different sources.
    # this method generates fql for user financial data and calls FQLEngine to get the result
    # other types of sources(?) are not implemented yet.
    def gather(self, user_query: str, user_id: str):
        fql_gen = self.fql_gen_chain.run(user_query=user_query)
        print(fql_gen)
        if 'Output:' in fql_gen:
            fql_gen = fql_gen.split('Output:')[1].strip()
        if fql_gen == 'None':
            return None
        fql_result = self.fql_engine.run(fql_gen, user_id)
        llm_logger.log(use_case='question_to_fql', prompt=user_query, response=fql_gen,
                       llm_params=self._llm._default_params)
        return fql_result
