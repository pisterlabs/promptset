#Custom Agent with PlugIn Retrieval
'''
This notebook combines two concepts in order to build a custom agent that can interact with AI Plugins:
  1.Custom Agent with Retrieval: This introduces the concept of retrieving many tools, which is useful when trying to work with 
    arbitrarily many plugins.
  2.Natural Language API Chains: This creates Natural Language wrappers around OpenAPI endpoints. This is useful because (1)
   plugins use OpenAPI endpoints under the hood, (2) wrapping them in an NLAChain allows the router agent to call it more easily.
'''
'''
The novel idea introduced in this notebook is the idea of using retrieval to select not the tools explicitly, 
but the set of OpenAPI specs to use. We can then generate tools from those OpenAPI specs. The use case for this 
is when trying to get agents to use plugins. It may be more efficient to choose plugins first, then the endpoints, 
rather than the endpoints directly. This is because the plugins may contain more useful information for selection.
'''
#Custom Agent with PlugIn Retrieval
'''
This notebook combines two concepts in order to build a custom agent that can interact with AI Plugins:
  1.Custom Agent with Retrieval: This introduces the concept of retrieving many tools, which is useful when trying to work with 
    arbitrarily many plugins.
  2.Natural Language API Chains: This creates Natural Language wrappers around OpenAPI endpoints. This is useful because (1)
   plugins use OpenAPI endpoints under the hood, (2) wrapping them in an NLAChain allows the router agent to call it more easily.
'''
'''
The novel idea introduced in this notebook is the idea of using retrieval to select not the tools explicitly, 
but the set of OpenAPI specs to use. We can then generate tools from those OpenAPI specs. The use case for this 
is when trying to get agents to use plugins. It may be more efficient to choose plugins first, then the endpoints, 
rather than the endpoints directly. This is because the plugins may contain more useful information for selection.
'''

import os
os.environ["OPENAI_API_KEY"] ="your_api_key"                       


#Set up environment
#Do necessary imports, etc.
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import StringPromptTemplate
from langchain import OpenAI, SerpAPIWrapper, LLMChain
from typing import List, Union
from langchain.schema import AgentAction, AgentFinish
from langchain.agents.agent_toolkits import NLAToolkit
from langchain.tools.plugin import AIPlugin
import re

#Setup LLM
def ai_plugins():
    llm = OpenAI(temperature=0)

    #wSet up plugins
    #Load and index plugins

    urls = [
        "https://datasette.io/.well-known/ai-plugin.json",
        "https://api.speak.com/.well-known/ai-plugin.json",
        "https://www.wolframalpha.com/.well-known/ai-plugin.json",
        "https://www.zapier.com/.well-known/ai-plugin.json",
        "https://www.klarna.com/.well-known/ai-plugin.json",
        "https://www.joinmilo.com/.well-known/ai-plugin.json",
        "https://slack.com/.well-known/ai-plugin.json",
        "https://schooldigger.com/.well-known/ai-plugin.json",
    ]

    AI_PLUGINS = [AIPlugin.from_url(url) for url in urls]

    #Tool Retriever
    #We will use a vectorstore to create embeddings for each tool description. Then, for an incoming query we can 
    #create embeddings for that query and do a similarity search for relevant tools.

    from langchain.vectorstores import FAISS
    from langchain.embeddings import OpenAIEmbeddings
    from langchain.schema import Document


    embeddings = OpenAIEmbeddings()
    docs = [
        Document(page_content=plugin.description_for_model, 
                metadata={"plugin_name": plugin.name_for_model}
                )
        for plugin in AI_PLUGINS
    ]
    vector_store = FAISS.from_documents(docs, embeddings)
    toolkits_dict = {plugin.name_for_model: 
                    NLAToolkit.from_llm_and_ai_plugin(llm, plugin) 
                    for plugin in AI_PLUGINS}

    #print(toolkits_dict)

    retriever = vector_store.as_retriever()

    def get_tools(query):
        # Get documents, which contain the Plugins to use
        docs = retriever.get_relevant_documents(query)
        # Get the toolkits, one for each plugin
        tool_kits = [toolkits_dict[d.metadata["plugin_name"]] for d in docs]
        # Get the tools: a separate NLAChain for each endpoint
        tools = []
        for tk in tool_kits:
            tools.extend(tk.nla_tools)
        return tools

    #We can now test this retriever to see if it seems to work.
    tools = get_tools("What could I do today with my kiddo")
    a=[t.name for t in tools]
    #print(a)

    tools = get_tools("what shirts can i buy?")
    a=[t.name for t in tools]
    #print(a)


    #Prompt Template
    '''
    The prompt template is pretty standard, because we’re not actually changing that much logic in the actual prompt template,
    but rather we are just changing how retrieval is done.
    '''
    # Set up the base template
    template = """Answer the following questions as best you can, but speaking as a pirate might speak. You have access to the following tools:

    {tools}

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Begin! Remember to speak as a pirate when giving your final answer. Use lots of "Arg"s

    Question: {input}
    {agent_scratchpad}"""

    #The custom prompt template now has the concept of a tools_getter, which we call on the input to select the tools to use

    from typing import Callable
    # Set up a prompt template
    class CustomPromptTemplate(StringPromptTemplate):
        # The template to use
        template: str
        ############## NEW ######################
        # The list of tools available
        tools_getter: Callable
        
        def format(self, **kwargs) -> str:
            # Get the intermediate steps (AgentAction, Observation tuples)
            # Format them in a particular way
            intermediate_steps = kwargs.pop("intermediate_steps")
            thoughts = ""
            for action, observation in intermediate_steps:
                thoughts += action.log
                thoughts += f"\nObservation: {observation}\nThought: "
            # Set the agent_scratchpad variable to that value
            kwargs["agent_scratchpad"] = thoughts
            ############## NEW ######################
            tools = self.tools_getter(kwargs["input"])
            # Create a tools variable from the list of tools provided
            kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in tools])
            # Create a list of tool names for the tools provided
            kwargs["tool_names"] = ", ".join([tool.name for tool in tools])
            return self.template.format(**kwargs)
        
    prompt = CustomPromptTemplate(
        template=template,
        tools_getter=get_tools,
        # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
        # This includes the `intermediate_steps` variable because that is needed
        input_variables=["input", "intermediate_steps"]
    )

    #Output Parser
    #The output parser is unchanged from the previous notebook, since we are not changing anything about the output format.

    class CustomOutputParser(AgentOutputParser):
        
        def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
            # Check if agent should finish
            if "Final Answer:" in llm_output:
                return AgentFinish(
                    # Return values is generally always a dictionary with a single `output` key
                    # It is not recommended to try anything else at the moment :)
                    return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                    log=llm_output,
                )
            # Parse out the action and action input
            regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
            match = re.search(regex, llm_output, re.DOTALL)
            if not match:
                raise ValueError(f"Could not parse LLM output: `{llm_output}`")
            action = match.group(1).strip()
            action_input = match.group(2)
            # Return the action and action input
            return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)
        
    output_parser = CustomOutputParser()

    #Set up LLM, stop sequence, and the agent
    #Also the same as the previous notebook

    llm = OpenAI(temperature=0)
    # LLM chain consisting of the LLM and a prompt
    llm_chain = LLMChain(llm=llm, prompt=prompt)

    tool_names = [tool.name for tool in tools]
    agent = LLMSingleActionAgent(
        llm_chain=llm_chain, 
        output_parser=output_parser,
        stop=["\nObservation:"], 
        allowed_tools=tool_names
    )

    #Use the Agent
    #Now we can use it!

    agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)
    agent_executor.run("what shirts can i buy?")

ai_plugins()