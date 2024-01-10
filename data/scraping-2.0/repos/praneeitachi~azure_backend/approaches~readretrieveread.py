from typing import Any

import openai
from azure.search.documents.aio import SearchClient
from azure.search.documents.models import QueryType
from langchain.agents import AgentExecutor, Tool, ZeroShotAgent
from langchain.callbacks.manager import CallbackManager, Callbacks
from langchain.callbacks import get_openai_callback # import for token count
from langchain.chains import LLMChain
from langchain.llms.openai import AzureOpenAI
from langchain.chat_models import AzureChatOpenAI
from langchain.agents import AgentType
from core.messagebuilder import MessageBuilder
from core.evaluation import calculate_relevance_score
from approaches.approach import AskApproach
from langchainadapters import HtmlCallbackHandler
from lookuptool import CsvLookupTool
from text import nonewlines

class ReadRetrieveReadApproach(AskApproach):
    """
    Attempt to answer questions by iteratively evaluating the question to see what information is missing, and once all information
    is present then formulate an answer. Each iteration consists of two parts:
     1. use GPT to see if we need more information
     2. if more data is needed, use the requested "tool" to retrieve it.
    The last call to GPT answers the actual question.
    This is inspired by the MKRL paper[1] and applied here using the implementation in Langchain.

    [1] E. Karpas, et al. arXiv:2205.00445
    """
    system_chat_template = \
"You are an intelligent assistant helping employees with their healthcare plan questions and employee handbook questions. " + \
"Use 'you' to refer to the individual asking the questions even if they ask with 'I'. " + \
"Answer the following question using only the data provided in the sources below. " + \
"For tabular information return it as an html table. Do not return markdown format. "  + \
"Each source has a name followed by colon and the actual information, always include the source name for each fact you use in the response. " + \
"If you cannot answer using the sources below, say you don't know"
    
    template_prefix = \
"You are an intelligent assistant helping employees with their questions regarding the benifits and schemes they are eligible for ." \
"Dont give responses irrelevant to the question or the context provided to you by the tools." \
"First Think the steps to be taken to get the answer for the question, then take an action based on the thought ."\
"Use the result of the action as an observation to generate the next thought "\
"Answer the question using only the data provided in the information sources below. " \
"For tabular information return it as an html table. Do not return markdown format. " \
"Each source has a name followed by colon and the actual data, quote the source name for each piece of data you use in the response. " \
"For example, if the question is \"What color is the sky?\" and one of the information sources says \"info123: the sky is blue whenever it's not cloudy\", then answer with \"The sky is blue [info123]\" " \
"It's important to strictly follow the format where the name of the source is in square brackets at the end of the sentence, and only up to the prefix before the colon (\":\"). " \
"If there are multiple sources, cite each one in their own square brackets. For example, use \"[info343][ref-76]\" and not \"[info343,ref-76]\". " \
"Never quote tool names as sources." \
"Both tools take a string as input." \
"No need to provide type of input with action input"\
"If you cannot answer using the sources below, say that you don't know. " \
"Mention only the input provided "\
"Dont generate any other Question/Action to take once you have the final answer"\
"\n\nYou can access to the following tools:"

    FORMAT_INSTRUCTIONS = """Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do.
Action: the action to take, should be one of [Employee , CognitiveSearch]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times except in the final action which should not contain any Queston/Thougt/Action)
Thought: I now know the final answer (Dont generate any Question/Thought/Action in the final answer)
Final Answer: the final answer to the original input question """
    template_suffix = """Begin!

Question: {input}
Thought:{agent_scratchpad}"""

    CognitiveSearchToolDescription = "provide it with the specific search term to look for it in the documents for refernce to answer the users query"

    def __init__(self, search_client: SearchClient, openai_deployment: str, embedding_deployment: str, sourcepage_field: str, content_field: str):
        self.search_client = search_client
        self.openai_deployment = openai_deployment
        self.embedding_deployment = embedding_deployment
        self.sourcepage_field = sourcepage_field
        self.content_field = content_field

    async def retrieve(self, query_text: str, overrides: dict[str, Any]) -> Any:
        has_text = overrides.get("retrieval_mode") in ["text", "hybrid", None]
        has_vector = overrides.get("retrieval_mode") in ["vectors", "hybrid", None]
        use_semantic_captions = True if overrides.get("semantic_captions") and has_text else False
        top = overrides.get("top") or 3
        exclude_category = overrides.get("exclude_category") or None
        filter = None
        if exclude_category:
            filter = " and ".join(f"category ne '{category}'" for category in exclude_category)
            filter = f"({filter})"
        # If retrieval mode includes vectors, compute an embedding for the query
        if has_vector:
            query_vector = (await openai.Embedding.acreate(engine=self.embedding_deployment, input=query_text))["data"][0]["embedding"]
        else:
            query_vector = None

        # Only keep the text query if the retrieval mode uses text, otherwise drop it
        if not has_text:
            query_text = ""

        # Use semantic ranker if requested and if retrieval mode is text or hybrid (vectors + text)
        if overrides.get("semantic_ranker") and has_text:
            r = await self.search_client.search(query_text,
                                          filter=filter,
                                          query_type=QueryType.SEMANTIC,
                                          query_language="en-us",
                                          query_speller="lexicon",
                                          semantic_configuration_name="my-semantic-config",
                                          top = top,
                                          query_caption="extractive|highlight-false" if use_semantic_captions else None,
                                          vector=query_vector,
                                          top_k=50 if query_vector else None,
                                          vector_fields="contentVector" if query_vector else None)
        else:
            r = await self.search_client.search(query_text,
                                          filter=filter,
                                          top=top,
                                          vector=query_vector,
                                          top_k=50 if query_vector else None,
                                          vector_fields="contentVector" if query_vector else None)
        if use_semantic_captions:
            results = [doc[self.sourcepage_field] + ":" + nonewlines(" -.- ".join([c.text for c in doc['@search.captions']])) async for doc in r]
        else:
            results = [doc[self.sourcepage_field] + ":" + nonewlines(doc[self.content_field]) async for doc in r]
        content = "\n".join(results)
        return results, content

    async def run(self, q: str, overrides: dict[str, Any]) -> Any:
        print("-----Chat Step 3---------------")
        retrieve_results = ['None']
        
        async def retrieve_and_store(q: str) -> Any:
            nonlocal retrieve_results
            retrieve_results, content = await self.retrieve(q, overrides)
            return content

        # Use to capture thought process during iterations
        cb_handler = HtmlCallbackHandler()
        cb_manager = CallbackManager(handlers=[cb_handler])

        acs_tool = Tool(name="CognitiveSearch",
                        func=lambda _: 'Not implemented',
                        coroutine=retrieve_and_store,
                        description=self.CognitiveSearchToolDescription,
                        callbacks=cb_manager)
        employee_tool = EmployeeInfoTool("Employee1", callbacks=cb_manager)
        tools = [acs_tool, employee_tool]

        prompt = ZeroShotAgent.create_prompt(
            tools=tools,
            prefix=overrides.get("prompt_template_prefix") or self.template_prefix,
            suffix=overrides.get("prompt_template_suffix") or self.template_suffix,
            format_instructions= self.FORMAT_INSTRUCTIONS ,
            input_variables = ["input", "agent_scratchpad"])
        #print("------Prompt--------- : ",prompt)
        llm = AzureChatOpenAI(deployment_name=self.openai_deployment, temperature= 0.1 or overrides.get("temperature"),openai_api_base=openai.api_base, openai_api_key=openai.api_key, openai_api_version=openai.api_version)
        #llm = AzureOpenAI(deployment_name=self.openai_deployment, temperature= 0, openai_api_key=openai.api_key, openai_api_version=openai.api_version)
        chain = LLMChain(llm = llm, prompt = prompt )
        """agent = initialize_agent(llm = llm,
            agent = AgentType.ZERO_SHOT_REACT_DESCRIPTION , 
            agent_kwargs={'prefix':self.template_prefix,'format_instructions':self.FORMAT_INSTRUCTIONS,'suffix':self.template_suffix},
            #agent = ZeroShotAgent(llm_chain = chain,early_stopping_method="generate",max_iterations=15),
            tools = tools,
            verbose = True,
            callback_manager = cb_manager,handle_parsing_errors=True
                    )"""
        
        try :
            agent_exec = AgentExecutor.from_agent_and_tools(
                agent = ZeroShotAgent(llm_chain = chain,early_stopping_method="generate",max_iterations=7),# max_execution_time = 60),
                tools = tools,
                verbose = True,
                callback_manager = cb_manager,handle_parsing_errors=True)
            # measuring the token count throught out the agent process 
            #result = await agent_exec.arun(q)
            with get_openai_callback() as cb:
                result = await agent_exec.arun(q)
                token_count = cb.total_tokens
                usage_cost = cb.total_cost

            
            print("-----Chat Step 4---------------")
            #emp = employee_detail()
            from lookuptool import emp
            print("------retrieve_results--------- : ", emp)
            context = emp+"\n".join(retrieve_results)
            #print("*"*10,'context',"*"*10)

            #print(emp)
            result = result.replace("[CognitiveSearch]", "").replace("[Employee]", "")
            print("-----Chat Step 5---------------")
            score = calculate_relevance_score(context,q,result)
            #if score > 2:
            print("-----Chat Step 6---------------")
            if score <3 :
                message_builder = MessageBuilder(overrides.get("prompt_template_prefix") or self.template_prefix, "gpt-4")
                #user_content = q + "\n" + f"Sources:\n {context}"
                message_builder.append_message('user', q)
                message_builder.append_message('user', context)
                messages = message_builder.messages
                chat_completion = await openai.ChatCompletion.acreate(
                    deployment_id=self.openai_deployment,
                    model="gpt-4",
                    messages=messages,
                    temperature=overrides.get("temperature") or 0.1,
                    # max_tokens=1024,
                    n=1)
                result = chat_completion.choices[0].message.content
                score = calculate_relevance_score(context,q,result)
        # Remove references to tool names that might be confused with a citation
            

        except openai.error.InvalidRequestError as e:
            if e.error.code == "content_filter": # and e.error.innererror:
                #content_filter_result = e.error.innererror.content_filter_result
                # print the formatted JSON
                #print("*"*10,"content_filter_result")
                result = "Triggering content please modify and retry "
                score =0



        return {"data_points": retrieve_results or [], "answer": result, "thoughts": cb_handler.get_and_reset_log(), "token_usage":token_count,"relevance_score":score}

class EmployeeInfoTool(CsvLookupTool):
    employee_name: str = ""

    def __init__(self, employee_name: str, callbacks: Callbacks = None):
        super().__init__(filename="data/employeeinfo.csv",
                         key_field="name",
                         name="Employee",
                         description="provide the employee name as input to asnwer question regarding the benefits they are eligible to avail based on their employment details and other personal information",
                         callbacks=callbacks)
        self.func = lambda _: 'Not implemented'
        self.coroutine = self.employee_info
        self.employee_name = employee_name
        

    async def employee_info(self, name: str) -> str:
        return self.lookup(name)
