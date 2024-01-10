import re
from typing import Any, Optional, Sequence

import openai
from azure.search.documents.aio import SearchClient
from azure.search.documents.models import QueryType
from langchain.agents import AgentExecutor, Tool
from langchain.agents.react.base import ReActDocstoreAgent
from langchain.callbacks.manager import CallbackManager
from langchain.llms.openai import AzureOpenAI
from langchain.prompts import BasePromptTemplate, PromptTemplate
from langchain.tools.base import BaseTool

from approaches.approach import AskApproach
from langchainadapters import HtmlCallbackHandler
from text import nonewlines


class ReadDecomposeAsk(AskApproach):
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
        filter = "category ne '{}'".format(exclude_category.replace("'", "''")) if exclude_category else None

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
                                          semantic_configuration_name="default",
                                          top=top,
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
            results = [doc[self.sourcepage_field] + ":" + nonewlines(" . ".join([c.text for c in doc['@search.captions'] ])) async for doc in r]
        else:
            results = [doc[self.sourcepage_field] + ":" + nonewlines(doc[self.content_field][:500]) async for doc in r]
        return results, "\n".join(results)

    async def lookup(self, q: str) -> Optional[str]:
        r = await self.search_client.search(q,
                                      top = 1,
                                      include_total_count=True,
                                      query_type=QueryType.SEMANTIC,
                                      query_language="en-us",
                                      query_speller="lexicon",
                                      semantic_configuration_name="default",
                                      query_answer="extractive|count-1",
                                      query_caption="extractive|highlight-false")

        answers = await r.get_answers()
        if answers and len(answers) > 0:
            return answers[0].text
        if await r.get_count() > 0:
            return "\n".join([d['content'] async for d in r])
        return None

    async def run(self, q: str, overrides: dict[str, Any]) -> Any:

        search_results = None
        async def search_and_store(q: str) -> Any:
            nonlocal search_results
            search_results, content = await self.search(q, overrides)
            return content

        # Use to capture thought process during iterations
        cb_handler = HtmlCallbackHandler()
        cb_manager = CallbackManager(handlers=[cb_handler])

        llm = AzureOpenAI(deployment_name=self.openai_deployment, temperature=overrides.get("temperature") or 0.3, openai_api_key=openai.api_key)
        tools = [
            Tool(name="Search", func=lambda _: 'Not implemented', coroutine=search_and_store, description="useful for when you need to ask with search", callbacks=cb_manager),
            Tool(name="Lookup", func=lambda _: 'Not implemented', coroutine=self.lookup, description="useful for when you need to ask with lookup", callbacks=cb_manager)
        ]

        prompt_prefix = overrides.get("prompt_template")
        prompt = PromptTemplate.from_examples(
            EXAMPLES, SUFFIX, ["input", "agent_scratchpad"], prompt_prefix + "\n\n" + PREFIX if prompt_prefix else PREFIX)

        class ReAct(ReActDocstoreAgent):
            @classmethod
            def create_prompt(cls, tools: Sequence[BaseTool]) -> BasePromptTemplate:
                return prompt

        agent = ReAct.from_llm_and_tools(llm, tools)
        chain = AgentExecutor.from_agent_and_tools(agent, tools, verbose=True, callback_manager=cb_manager)
        result = await chain.arun(q)

        # Replace substrings of the form <file.ext> with [file.ext] so that the frontend can render them as links, match them with a regex to avoid
        # generalizing too much and disrupt HTML snippets if present
        result = re.sub(r"<([a-zA-Z0-9_ \-\.]+)>", r"[\1]", result)

        return {"data_points": search_results or [], "answer": result, "thoughts": cb_handler.get_and_reset_log()}



# Modified version of langchain's ReAct prompt that includes instructions and examples for how to cite information sources
EXAMPLES = [
    """Question: コロラド造山帯の東部が広がる地域の標高範囲は？
Thought: コロラド造山帯を検索し、その東部地域が広がっている地域を見つける必要がある。
コロラド造山帯の東部が伸びている地域を見つけ、その地域の標高範囲を調べたい。
そして、その地域の標高範囲を求める。
Action: Search[Colorado orogeny]
Observation: <info1.pdf> コロラド造山は、コロラド州とその周辺地域で起こった造山活動である。
Thought: 東部地区については書かれていない。だから、東部セクターについて調べる必要がある。
""",
    """Question: ミュージシャンで風刺作家のアリー・ゴーツが、『ザ・シンプソンズ』のキャラクター、ミルハウスについて曲を書いた。
マット・グルーニングが誰にちなんで名付けたか？
Thought: 質問は「ザ・シンプソンズ」のキャラクター、ミルハウスは誰にちなんで名づけられたか？誰の名前か？ミルハウスを検索して、誰の名前にちなんでいるのかを見つけるだけでいい。
Action: Search[Milhouse]
Observation: <info7.pdf> ミルハウス・ムッソリーニ・ヴァン・ホーテン（Milhouse Mussolini Van Houten）は、FOXのテレビアニメ『ザ・シンプソンズ』シリーズの登場人物である。「ザ・シンプソンズ」のレギュラーキャラクターで、声はパメラ・ヘイデン、作画はマット・グルーニング。よってグルーニングによって創作された。
""",
    """Question: ニコラス・レイとエリア・カザンの共通点は？
Thought: ニコラス・レイとエリア・カザンを検索し、彼らの職業を見つけ、そして共通する職業を見つける必要がある。共通する職業を見つける必要がある。
Action: Search[Nicholas Ray]
Observation: <files-987.png> ニコラス・レイ（Raymond Nicholas Kienzle Jr、1911年8月7日 - 1979年6月16日、1979年6月16日）は、1955年の映画『理由なき反抗』で知られるアメリカの映画監督、脚本家、俳優。1955年の映画『理由なき反抗』で知られる。
""",
    """Question: パヴェル・ウリソーンとレオニード・レヴィンは同じタイプの仕事で知られていたのですか？
Thought: パヴェル・ウリゾーンとレオニード・レヴィンを検索して、彼らの仕事の種類を見つける必要がある、そして、両者が同じかどうかを調べる必要がある。
Action: Search[Pavel Urysohn]
""",
]
SUFFIX = """\nQuestion: {input}
{agent_scratchpad}"""
PREFIX = "次の例に示すように、質問を個々の検索に分割して、質問に答えられるようになるまで事実を見つけることで、質問に答えます。" \
"Observationsは角括弧で囲まれたソース名を前につける。ソース名は、回答のアクションに含めなければならない。" \
"すべての質問は、検索の結果から答えなければなりません。 "
"質問にはできるだけ正直に答え、推測や自分の知識ではなく、観察から得た情報のみを使って答えること。"
