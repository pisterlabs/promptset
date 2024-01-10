import os
import openai


from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

from langchain.chat_models import ChatOpenAI # model
from langchain.prompts import ChatPromptTemplate # prompt
from langchain.chains import LLMChain # chain
from langchain.chains.router import MultiPromptChain
from langchain.chains.router.llm_router import LLMRouterChain,RouterOutputParser
from langchain.prompts import PromptTemplate

from langchain.agents.agent_toolkits import create_python_agent
from langchain.agents import load_tools, initialize_agent
from langchain.agents import AgentType
from langchain.tools.python.tool import PythonREPLTool
from langchain.python import PythonREPL


from langchain.agents import tool
from datetime import date

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']

class Memory:
    def __init__(self):
        self.conversation = None

    def function(self):
        llm = ChatOpenAI(temperature=0.0)
        memory = ConversationBufferMemory()
        conversation = ConversationChain(
            llm=llm, 
            verbose=True,
            memory = memory)
        
        self.conversation = conversation
    
    def conv_func(self,string):

        x = self.conversation.predict(input = string)

        return x
     

class llmchain:
    def __init__(self):
        self.MULTI_PROMPT_ROUTER_TEMPLATE = None
        self.prompt_infos = None

        self.prompt_infos_()
        self.prompt_router()
        self.function_llm()

    def prompt_router(self):
        MULTI_PROMPT_ROUTER_TEMPLATE = """Given a raw text input to a \
        language model select the model prompt best suited for the input. \
        You will be given the names of the available prompts and a \
        description of what the prompt is best suited for. \
        You may also revise the original input if you think that revising\
        it will ultimately lead to a better response from the language model.

        << FORMATTING >>
        Return a markdown code snippet with a JSON object formatted to look like:
        ```json
        {{{{
            "destination": string \ name of the prompt to use or "DEFAULT"
            "next_inputs": string \ a potentially modified version of the original input
        }}}}
        ```

        REMEMBER: "destination" MUST be one of the candidate prompt \
        names specified below OR it can be "DEFAULT" if the input is not\
        well suited for any of the candidate prompts.
        REMEMBER: "next_inputs" can just be the original input \
        if you don't think any modifications are needed.

        << CANDIDATE PROMPTS >>
        {destinations}

        << INPUT >>
        {{input}}

        << OUTPUT (remember to include the ```json)>>"""

        self.MULTI_PROMPT_ROUTER_TEMPLATE = MULTI_PROMPT_ROUTER_TEMPLATE

    def prompt_infos_(self):
        electronic_engineer_template = """You are a skilled electronic engineer\
        specializing in various aspects of electronics and electrical engineering. \
        You have a deep understanding of circuit design, microelectronics, and electrical systems. \
        Your expertise allows you to provide clear and precise answers to questions related to electronic engineering. \
        However, when faced with a question beyond your knowledge, you readily admit it. \

        Here is a question:
        {input}"""

        physics_template = """You are a very smart physics professor. \
        You are great at answering questions about physics in a concise\
        and easy to understand manner. \
        When you don't know the answer to a question you admit\
        that you don't know.

        Here is a question:
        {input}"""

        biology_template = """You are a knowledgeable biologist with expertise in various aspects of biology. \
        You excel at providing clear and insightful answers to questions related to biology. \
        However, if you ever come across a question you can't answer, you openly acknowledge it.

        Here is a question:
        {input}"""

        math_template = """You are a very good mathematician. \
        You are great at answering math questions. \
        You are so good because you are able to break down \
        hard problems into their component parts, 
        answer the component parts, and then put them together\
        to answer the broader question.

        Here is a question:
        {input}"""

        history_template = """You are a very good historian. \
        You have an excellent knowledge of and understanding of people,\
        events and contexts from a range of historical periods. \
        You have the ability to think, reflect, debate, discuss and \
        evaluate the past. You have a respect for historical evidence\
        and the ability to make use of it to support your explanations \
        and judgements.

        Here is a question:
        {input}"""


        computerscience_template = """ You are a successful computer scientist.\
        You have a passion for creativity, collaboration,\
        forward-thinking, confidence, strong problem-solving capabilities,\
        understanding of theories and algorithms, and excellent communication \
        skills. You are great at answering coding questions. \
        You are so good because you know how to solve a problem by \
        describing the solution in imperative steps \
        that a machine can easily interpret and you know how to \
        choose a solution that has a good balance between \
        time complexity and space complexity. 

        Here is a question:
        {input}"""

        prompt_infos = [
        {
        "name": "electronics",
        "description": "Good for answering questions about electronic circuits and devices",
        "prompt_template": electronic_engineer_template
            },
            {
                "name": "physics", 
                "description": "Good for answering questions about physics", 
                "prompt_template": physics_template
            },
            {
                "name": "math", 
                "description": "Good for answering math questions", 
                "prompt_template": math_template
            },
            {
                "name": "History", 
                "description": "Good for answering history questions", 
                "prompt_template": history_template
            },
            {
                "name": "computer science", 
                "description": "Good for answering computer science questions", 
                "prompt_template": computerscience_template
            }
        ]


        self.prompt_infos = prompt_infos
    
    def function_llm(self):
        llm = ChatOpenAI(temperature=0)
        destination_chains = {}
        for p_info in self.prompt_infos:
            name = p_info["name"]
            prompt_template = p_info["prompt_template"]
            prompt = ChatPromptTemplate.from_template(template=prompt_template)
            chain = LLMChain(llm=llm, prompt=prompt)
            destination_chains[name] = chain  
    
        destinations = [f"{p['name']}: {p['description']}" for p in self.prompt_infos]
        destinations_str = "\n".join(destinations)

        default_prompt = ChatPromptTemplate.from_template("{input}")
        default_chain = LLMChain(llm=llm, prompt=default_prompt)

        self.destinations_str = destinations_str
        self.default_chain = default_chain
        self.llm = llm
        self.destination_chains = destination_chains

    def router_chain(self, string):

        router_template = self.MULTI_PROMPT_ROUTER_TEMPLATE.format(
            destinations=self.destinations_str
        )
        router_prompt = PromptTemplate(
            template=router_template,
            input_variables=["input"],
            output_parser=RouterOutputParser(),
        )

        router_chain = LLMRouterChain.from_llm(self.llm, router_prompt)

        chain = MultiPromptChain(router_chain=router_chain, 
                         destination_chains=self.destination_chains, 
                         default_chain=self.default_chain, verbose=True
                        )
        
        c = chain.run(string)
        return c


class LLMtool:
    def __init__(self):
        self.test()
    def test(self):
        llm = ChatOpenAI(temperature=0)
        tools = load_tools(["llm-math","wikipedia"], llm=llm)
        agent= initialize_agent(
            tools + [self.time], 
            llm, 
            agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            handle_parsing_errors=True,
        verbose = True)

        try:
            result = agent("whats the date today?") 
        except: 
            print("exception on external access")

    @tool
    def time(text: str) -> str:
        """Returns todays date, use this for any \
        questions related to knowing todays date. \
        The input should always be an empty string, \
        and this function will always return todays \
        date - any date mathmatics should occur \
        outside this function."""
        return str(date.today())
    



if __name__ == "__main__":

    instance = LLMtool()

if __name__ == "__main__":

    instance = llmchain()


    str_list = ["What is black body radiation?", "What is the importance of Fourier transform in electronics engineering?"]
    for string in str_list:

        x = instance.router_chain(string)
        print("\n")
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        print(x)

if __name__ == "__main__":

    instance = Memory()
    instance.function()

    str_list = ["Hi, my name is Şaban", "What is 1+1?", "What is my name?"]
    for string in str_list:

        x = instance.conv_func(string)

        print(x)

    c=5
