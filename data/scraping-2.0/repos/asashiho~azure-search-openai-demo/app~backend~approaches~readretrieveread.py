from typing import Any

import openai
from azure.search.documents.aio import SearchClient
from azure.search.documents.models import QueryType
from langchain.agents import AgentExecutor, Tool, ZeroShotAgent
from langchain.callbacks.manager import CallbackManager, Callbacks
from langchain.chains import LLMChain
from langchain.llms.openai import AzureOpenAI

from approaches.approach import AskApproach
from langchainadapters import HtmlCallbackHandler
from lookuptool import CsvLookupTool
from text import nonewlines


class ReadRetrieveReadApproach(AskApproach):
    """
    どのような情報が欠けているかを確認するために質問を繰り返し評価することによって、質問に答えます。質問は2つの部分で構成されます:
     1. GPTを使用して、さらに情報が必要かどうかを確認する
     2. より多くのデータが必要な場合は、要求された "ツール "を使用してデータを取得する。
    GPTへの最後の呼び出しが、実際の質問に答えます。
    """

    template_prefix = \
"あなたは、鉄道技術に関する質問をサポートするインテリジェントアシスタントです。 " \
"以下の情報源に記載されているデータのみを使用して質問に答えてください。 " \
"表形式の情報については、HTMLとして返してください。マークダウン形式は返さないでください。" \
"各ソースには、名前の後にコロンと実際のデータが続き、レスポンスで使用するデータごとにソース名を引用します。" \
"たとえば, もし質問が \"空の色は?\" であり、情報源が \"info123: 曇っていなければ空は青い\", という場合は \"空は青い[info123]\"と答えます " \
"出典の名前は文末の角括弧の中に、コロンの前の接頭辞までしか書かないという形式に従う必要があります。 (\":\"). " \
"複数の出典がある場合は、それぞれの出典を角括弧で囲んで引用します。例えば, \"[info343][ref-76]\" と \"[info343,ref-76]\". " \
"ツール名を情報源として引用してはいけません" \
"以下の資料で答えられない場合は、「わからない」と答えてください。" \
"\n\n以下のツールにアクセスできます:"

    template_suffix = """
Begin!

Question: {input}

Thought: {agent_scratchpad}"""

    CognitiveSearchToolDescription = "鉄道技術を知りたいときに便利です"

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
        filter = "category ne '{}'".format(exclude_category.replace("'", "''")) if exclude_category else None

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
                                          semantic_configuration_name="default",
                                          top = top,
                                          query_caption="extractive|highlight-false" if use_semantic_captions else None,
                                          vector=query_vector,
                                          top_k=50 if query_vector else None,
                                          vector_fields="embedding" if query_vector else None)
        else:
            r = await self.search_client.search(query_text,
                                          filter=filter,
                                          top=top,
                                          vector=query_vector,
                                          top_k=50 if query_vector else None,
                                          vector_fields="embedding" if query_vector else None)
        if use_semantic_captions:
            results = [doc[self.sourcepage_field] + ":" + nonewlines(" -.- ".join([c.text for c in doc['@search.captions']])) async for doc in r]
        else:
            results = [doc[self.sourcepage_field] + ":" + nonewlines(doc[self.content_field][:250]) async for doc in r]
        content = "\n".join(results)
        return results, content

    async def run(self, q: str, overrides: dict[str, Any]) -> Any:

        retrieve_results = None
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
            input_variables = ["input", "agent_scratchpad"])
        llm = AzureOpenAI(deployment_name=self.openai_deployment, temperature=overrides.get("temperature") or 0.3, openai_api_key=openai.api_key)
        chain = LLMChain(llm = llm, prompt = prompt)
        agent_exec = AgentExecutor.from_agent_and_tools(
            agent = ZeroShotAgent(llm_chain = chain),
            tools = tools,
            verbose = True,
            callback_manager = cb_manager)
        result = await agent_exec.arun(q)

        # Remove references to tool names that might be confused with a citation
        result = result.replace("[CognitiveSearch]", "").replace("[Employee]", "")

        return {"data_points": retrieve_results or [], "answer": result, "thoughts": cb_handler.get_and_reset_log()}

class EmployeeInfoTool(CsvLookupTool):
    employee_name: str = ""

    def __init__(self, employee_name: str, callbacks: Callbacks = None):
        super().__init__(filename="data/employeeinfo.csv",
                         key_field="name",
                         name="Employee",
                         description="useful for answering questions about the employee, their benefits and other personal information",
                         callbacks=callbacks)
        self.func = lambda _: 'Not implemented'
        self.coroutine = self.employee_info
        self.employee_name = employee_name

    async def employee_info(self, name: str) -> str:
        return self.lookup(name)
