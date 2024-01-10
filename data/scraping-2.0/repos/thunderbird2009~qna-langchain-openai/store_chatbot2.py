from langchain.callbacks.manager import Callbacks
from typing import Tuple, Any
import re
from langchain.schema import AgentAction, AgentFinish
from typing import List, Union
from langchain import OpenAI, SerpAPIWrapper, LLMChain
from langchain.prompts import StringPromptTemplate
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.tools.base import BaseTool
from typing import Optional, Type, Any
import json
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAIChat

from store_tools import ProdSearchTool, CustServiceTool, DefaultTool

DEFAULT_OPENAI_API_KEY = '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'

# Set up a prompt template

class MyBasePromptTemplate(StringPromptTemplate):
    # The template to use
    template: str

    def format(self, **kwargs) -> str:
        return self.template.format(**kwargs)


class CustomOutputParser(AgentOutputParser):
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        data_dict = json.loads(llm_output)
        if not data_dict:
            return AgentFinish(
                return_values={"output": "My LLM has an error and please retry a different question."},
                log=llm_output,
            )
        # Check if agent should finish
        if "answer:" in data_dict:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": data_dict["answer"]},
                log=llm_output,
            )
        # Return the action and action input
        return AgentAction(tool=data_dict["intent_name"], tool_input=data_dict["query"], log=llm_output)


class MyTwoStepActionAgent(LLMSingleActionAgent):
    step2_llm: LLMChain = None

    def __init__(self, step2_llm, **data: Any) -> None:
        super().__init__(**data)
        self.step2_llm = step2_llm

    def plan(
        self,
        intermediate_steps: List[Tuple[AgentAction, str]],
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> Union[AgentAction, AgentFinish]:
        """
        Given input, decided what to do.

        Args:
            intermediate_steps: Steps the LLM has taken to date,
                along with observations
            callbacks: Callbacks to run.
            **kwargs: User inputs.

        Returns:
            Action specifying what tool to use.
        """
        if len(intermediate_steps) > 0:
            obs = json.loads(intermediate_steps[-1][1])
            if obs['type'] == 'prod_list':
                return AgentFinish(
                    return_values={"output": f'I found the following products:\n{json.dumps(obs["products"])}\n'}, log=""
                )
            elif obs['type'] == 'final_msg':
                return AgentFinish(return_values={"output": obs['msg']}, log="")
            elif obs['type'] == 'kb_src':
                kwargs['contenxt'] = obs['context']
                # should run the step2_llm here
                output = self.step2_llm.run(
                    callbacks=callbacks,
                    **kwargs,
                )
                if output == 'NO_ANSWER':
                    return AgentFinish(return_values=
                        {"output": f'I do not have a good answer. But this may be good reference: {obs["src"]}'},
                        log="")
                else:
                    output = output + f'\nsources:\n{obs["src"]}'
                    return self.output_parser.parse(output)

        llm_output = self.llm_chain.run(
            #intermediate_steps=intermediate_steps,
            #stop=self.stop,
            callbacks=callbacks,
            **kwargs,
        )
        return self.output_parser.parse(llm_output)


class StoreChatBot2:
    # Set up the base template
    template = """You are a chatbot of a Web store to generate answers to user messages (referred to as Message below).
You use the following processing logic as described in the following python code that includes psedu code in between comments "#psedu_start" and "#psedo_end".

class IntentName(Enum):
    INTENT_greeting = "Intent_greeting" 
    INTENT_cs = "Intent_cs"
    INTENT_prod = "Intent_prod"
    INTENT_default = "Intent_default"

class Intent:
    def __init__(self, intent_name: Intent, query: str, answer:str):
        self.intent_name = intent_name
        self.query = query
        self.answer = answer

# This function contains psedo code.
def process(msg:str) -> str:
    intents = []
    #psedu_start
    if msg contains a greeting:
        prepare an answer to greet the user and also to ask how you can help.  #psedo_end
        intents.append(Intent(INTENT_greeting, answer=answer))
    #psedu_start
    if msg contains a product inqury:
        prepare an query to query a product catalog.  #psedo_end
        intents.append(Intent(INTENT_prod, query=query))
    #psedu_start
    if msg contains customer service question about the web store, such as account, user profile, order, payment, shipment, return, shopping cart, etc
        prepare an query to query a store's knowledge base. #psedo_end
        intents.append(Intent(INTENT_cs, query=query))
    #psedu_start
    if msg containts questions unrelated to the web store:
        prepare an answer to ask you to either rephrase the question or redirect to a human agent. #psedo_end
        intents.append(Intent(INTENT_default, answer=answer))

    return json.dumps(intents)    

# msg is user input and output is your output.
output = process(msg)

Begin!
msg={input}"""

    template2 = """You are a chatbot of a Web store to generate answer to a query given a context. Output string NO_ANSWER if you can not
generate an answer for the query.

Begin!
context={context}
query={query}"""


    def __init__(self, prod_embedding_store, faq_embedding_store, openai_api_key, verbose=False):
        # Get your embeddings engine ready
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        cst = CustServiceTool(faq_embedding_store, embeddings)
        #cst.init(faq_embedding_store, embeddings)
        pst = ProdSearchTool(prod_embedding_store, embeddings)
        tools = [pst, cst, DefaultTool()]
        pst.name = "Intent_prod"
        cst.name = "Intent_cs"
        tool_names = [tool.name for tool in tools]
        prompt = MyBasePromptTemplate(
            template=self.template,            
            input_variables=["input"]
        )
        print(prompt)

        prompt2 = MyBasePromptTemplate(
            template=self.template2,
            input_variables=["context", "query"]
        )
        print(prompt2)

        # llm = OpenAIChat(model_name='gpt-3.5-turbo', temperature=0)
        # step2_llm = OpenAI(model_name='text-davinci-003', temperature=0)
        llm = OpenAI(model_name='text-davinci-003', temperature=0)
        # LLM chain consisting of the LLM and a prompt
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        step2_llm = LLMChain(llm=llm, prompt=prompt2)

        output_parser = CustomOutputParser()
        agent = MyTwoStepActionAgent(
            step2_llm=step2_llm,
            llm_chain=llm_chain,
            output_parser=output_parser,
            stop=[],
            allowed_tools=tool_names
        )
        self.agent_executor = AgentExecutor.from_agent_and_tools(
            agent=agent, tools=tools, verbose=verbose)

    def answer(self, user_msg) -> str:
        return self.agent_executor(user_msg)

