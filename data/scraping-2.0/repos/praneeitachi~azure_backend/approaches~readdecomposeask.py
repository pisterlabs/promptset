import re
from typing import Any, Optional, Sequence

import openai
from azure.search.documents.aio import SearchClient
from azure.search.documents.models import QueryType
from langchain.agents import Tool,AgentExecutor 
from langchain.agents.react.base import ReActDocstoreAgent
from langchain.agents.self_ask_with_search.base import SelfAskWithSearchAgent
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks import get_openai_callback  # import for token count
from langchain.llms.openai import AzureOpenAI
from langchain.prompts import BasePromptTemplate, PromptTemplate
from langchain.tools.base import BaseTool
from langchain.chat_models import AzureChatOpenAI
from core.evaluation import calculate_relevance_score
from core.messagebuilder import MessageBuilder
from approaches.approach import AskApproach
from langchainadapters import HtmlCallbackHandler #, CustomAgentExecutor as AgentExecutor
from text import nonewlines


class ReadDecomposeAsk(AskApproach):

    system_chat_template = \
"You are an intelligent assistant helping employees with their healthcare plan questions and employee handbook questions. " + \
"Use 'you' to refer to the individual asking the questions even if they ask with 'I'. " + \
"Answer the following question using only the data provided in the sources below. " + \
"For tabular information return it as an html table. Do not return markdown format. "  + \
"Each source has a name followed by colon and the actual information, always include the source name for each fact you use in the response. " + \
"If you cannot answer using the sources below, say you don't know"


    def __init__(self, search_client: SearchClient, openai_deployment: str, embedding_deployment: str, sourcepage_field: str, content_field: str):
        self.search_client = search_client
        self.openai_deployment = openai_deployment
        self.embedding_deployment = embedding_deployment
        self.sourcepage_field = sourcepage_field
        self.content_field = content_field

    async def search(self, query_text: str, overrides: dict[str, Any]) -> tuple[list[str], str]:
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

        if overrides.get("semantic_ranker") and has_text:
            r = await self.search_client.search(query_text,
                                          filter=filter,
                                          query_type=QueryType.SEMANTIC,
                                          query_language="en-us",
                                          query_speller="lexicon",
                                          semantic_configuration_name="my-semantic-config",
                                          top=top,
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

        #async for ele in r:
        #    print(ele['@search.captions'])
        if use_semantic_captions:
            print("----------------1")
            results = [doc[self.sourcepage_field] + ":" + nonewlines(" . ".join([c.text for c in doc['@search.captions'] ])) async for doc in r]
        else:
            print("----------------2")
            results = [doc[self.sourcepage_field] + ":" + nonewlines(doc[self.content_field]) async for doc in r]
        #print("XXXXXXXXXXXXXXXXXx")
        #print(results)
        return results, "\n".join(results)

    async def lookup(self, q: str) -> Optional[str]:
        r = await self.search_client.search(q,
                                      top = 1,
                                      include_total_count=True,
                                      query_type=QueryType.SEMANTIC,
                                      query_language="en-us",
                                      query_speller="lexicon",
                                      semantic_configuration_name="my-semantic-config",
                                      query_answer="extractive|count-1",
                                      query_caption="extractive|highlight-false")

        answers = await r.get_answers()
        if answers and len(answers) > 0:
            return answers[0].text
        if await r.get_count() > 0:
            return "\n".join([d['content'] async for d in r])
        return None
    
    async def run(self, q: str, overrides: dict[str, Any]) -> Any:
        print("-----RDA Ask Step 1---------------")
        search_results = ['None']
        #context = []
        async def search_and_store(q: str) -> Any:
            nonlocal search_results
            search_results, content = await self.search(q, overrides)
            #context.extend(search_results)
            return content

        # Use to capture thought process during iterations
        cb_handler = HtmlCallbackHandler()
        cb_manager = CallbackManager(handlers=[cb_handler])
        #llm = AzureOpenAI(deployment_name=self.openai_deployment, temperature=overrides.get("temperature") or 0.3, openai_api_key=openai.api_key, openai_api_version=openai.api_version)
        llm = AzureChatOpenAI(deployment_name=self.openai_deployment, temperature=0.1,top_p=0.1, openai_api_key=openai.api_key, openai_api_version=openai.api_version, openai_api_base=openai.api_base )
        print("-----RDA Ask Step 2---------------")
        tools = [
            Tool(name="Intermediate Answer", func=lambda _: 'Not implemented', coroutine=search_and_store, description="useful for when you need to search for a particulat piece of information", callbacks=cb_manager)]
            #,Tool(name="Lookup", func=lambda _: 'Not implemented', coroutine=self.lookup, description="useful for when you need to lookup a particular term in the documents", callbacks=cb_manager)]
        print("-----RDA Ask Step 3---------------")
        prompt_prefix = overrides.get("prompt_template")
        prompt = PromptTemplate.from_examples(
            EXAMPLES, SUFFIX, ["input", "agent_scratchpad"], prompt_prefix + "\n\n" + PREFIX if prompt_prefix else PREFIX,partial_variables={"format_instructions": format_instructions})
        print("------RDA Ask Step 4---------------")
        class ReAct(SelfAskWithSearchAgent):
            @classmethod
            def create_prompt(cls, tools: Sequence[BaseTool]) -> BasePromptTemplate:
                return prompt

        try:
            agent = SelfAskWithSearchAgent.from_llm_and_tools(llm, tools,prompt = prompt,early_stopping_method="generate")
            chain = AgentExecutor.from_agent_and_tools(agent, tools, verbose=True, callback_manager=cb_manager,handle_parsing_errors=True) #,return_intermediate_steps=True)
            # measuring the token count throught out the agent process 
            #response = await chain.arun(q)
            with get_openai_callback() as cb:
                response = await chain.arun(q)
                token_count = cb.total_tokens
                usage_cost = cb.total_cost

            print("------RDA Ask Step 5---------------")
            context = " ".join(search_results)
            print("------RDA Ask Step 6---------------")
            score = calculate_relevance_score(context,q,response)
            print("------RDA Ask Step 7---------------")

            # Prompt Flow
            if score > 3:
                result = response #['output']
            else :
                print ("-"*10, "Context : ",context)
                message_builder = MessageBuilder(overrides.get("prompt_template") or self.system_chat_template, "gpt-4")
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
                print("------RDA Ask Step 8---------------")
                #print ("-"*10, "Second Iteration : ",result)
                print("------RDA Ask Step 9---------------")

        except openai.error.InvalidRequestError as e:
            if e.error.code == "content_filter": # and e.error.innererror:
                #content_filter_result = e.error.innererror.content_filter_result
                # print the formatted JSON
                print("*"*10,"content_filter_result")
                result = "Triggering content please modify and retry "
                score=0
                print("-----Ask Step 12---------------")

        #print ("-"*10, "Context : ","\n".join(search_results))
        #print ("-"*10, "Question : ",q)
        #print ("-"*10, "Answer : ",response)

        #intermediate = response["intermediate_steps"]
        # Replace substrings of the form <file.ext> with [file.ext] so that the frontend can render them as links, match them with a regex to avoid
        # generalizing too much and disrupt HTML snippets if present
        print ("-"*10, "end")
        #print(intermediate)
        result = re.sub(r"<([a-zA-Z0-9_ \-\.]+)>", r"[\1]", result)
        #print("----Agent Scratchpad------:",agent.observation_prefix)
        return {"data_points": search_results or [], "answer": result, "thoughts": cb_handler.get_and_reset_log(),"token_usage":token_count,"relevance_score":score}



# Modified version of langchain's ReAct prompt that includes instructions and examples for how to cite information sources


EXAMPLES = [
    """Question: Were Pavel Urysohn and Leonid Levin known for the same type of work?
Thought: I need to search Pavel Urysohn and Leonid Levin, find their types of work,
then find if they are the same.
Action: Intermediate Answer[Pavel Urysohn]
Observation: <info4444.pdf> Pavel Samuilovich Urysohn (February 3, 1898 - August 17, 1924) was a Soviet
mathematician who is best known for his contributions in dimension theory.
Thought: Pavel Urysohn is a mathematician. I need to search Leonid Levin next and
find its type of work.
Action: Intermediate Answer[Leonid Levin]
Observation: <datapoints_aaa.txt> Leonid Anatolievich Levin is a Soviet-American mathematician and computer
scientist.
Thought: Leonid Levin is a mathematician and computer scientist. So Pavel Urysohn
and Leonid Levin have the same type of work.
Action: Finish[yes <info4444.pdf><datapoints_aaa.txt>]""",
]

format_instructions = """Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do.
Action: the action to take, should be one of [Intermediate Answer]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times except in the final action which should not contain any Queston/Thougt/Action)
Thought: I now know the final answer (Dont generate any Question/Thought/Action in the final answer)
Final Answer: the final answer to the original input question """

SUFFIX = """
Format Instructions:{format_instructions}

Begin!

Question:{input}
Thought:{agent_scratchpad}"""
PREFIX = "You are a helpful assistant that helps employee with their question regarding the ploicies and benefits mentioned in the employee handbook that you can access through the tools provided to you ."\
    "Answer questions as shown in the following examples, by splitting the question into individual Intermediate Answer actions to find facts until you can answer the question. " \
"Observations are prefixed by their source name in angled brackets, source names MUST be included with the actions in the answers." \
"All questions must be answered from the results from Intermediate Answer actions, only facts resulting from those can be used in an answer. "\
"Dont ask follow up question . Keep the response precise and limited. "\
"Answer questions as truthfully as possible, and ONLY answer the questions using the information from Intermediate Answers.If you dont have an answer just say I dont know."
