import openai
from approaches.approach import Approach
from azure.search.documents import SearchClient
from azure.search.documents.models import QueryType
from langchain.llms.openai import AzureOpenAI
from langchain.prompts import PromptTemplate, BasePromptTemplate
from langchain.callbacks.base import CallbackManager
from langchain.agents import Tool, AgentExecutor
from langchain.agents.react.base import ReActDocstoreAgent
from langchainadapters import HtmlCallbackHandler
from text import nonewlines
from typing import List

# TODO: 要検証
# 日本語でテストすると、トークン制限に引っかかるエラーが出まくるので、いろいろ制限を掛けてます
# InvalidRequestError: This model's maximum context length is xxx tokens, however you requested yyyy tokens (yyyy in your prompt; 256 for the completion). Please reduce your prompt; or completion length. 
# - Azure Cognitive Search の検索結果を制限: top = 1
# - Agent の反復回数を制限: max_iterations=3, early_stopping_method="generate"
# - プロンプトテンプレートの例を短くした（1つだけ）
class ReadDecomposeAsk(Approach):
    def __init__(self, search_client: SearchClient, openai_deployment: str, sourcepage_field: str, content_field: str):
        self.search_client = search_client
        self.openai_deployment = openai_deployment
        self.sourcepage_field = sourcepage_field
        self.content_field = content_field

    def search(self, q: str, overrides: dict) -> str:
        use_semantic_captions = True if overrides.get("semantic_captions") else False
        #top = overrides.get("top") or 3
        #hana debug 
        top = 1
        exclude_category = overrides.get("exclude_category") or None
        filter = "category ne '{}'".format(exclude_category.replace("'", "''")) if exclude_category else None

        if overrides.get("semantic_ranker"):
            print("q: ", q)
            r = self.search_client.search(q,
                                          filter=filter,
                                          query_type=QueryType.SEMANTIC, 
                                          query_language="ja-jp", 
                                          query_speller="none", 
                                          semantic_configuration_name="default", 
                                          top = top,
                                          query_caption="extractive|highlight-false" if use_semantic_captions else None)
        else:
            r = self.search_client.search(q, filter=filter, top=top)
        if use_semantic_captions:
            self.results = [doc[self.sourcepage_field] + ":" + nonewlines(" . ".join([c.text for c in doc['@search.captions'] ])) for doc in r]
            print("sc_result: ", self.results)
        else:
            self.results = [doc[self.sourcepage_field] + ":" + nonewlines(doc[self.content_field][:500]) for doc in r]
            print("result: ", self.results)
        
        #hana debug リチウム硫黄電池の技術課題は何ですか？
        #return ['JP201203.pdf:リチウムポリスルフィドの溶出などです。']
        return "\n".join(self.results)

    #セマンティックアンサー使えているのか？
    def lookup(self, q: str) -> str:
        print("q2: ", q)
        r = self.search_client.search(q,
                                      top = 1,
                                      include_total_count=True,
                                      query_type=QueryType.SEMANTIC, 
                                      query_language="ja-jp", 
                                      query_speller="none", 
                                      semantic_configuration_name="default",
                                      query_answer="extractive|count-1",
                                      query_caption="extractive|highlight-false")
        
        answers = r.get_answers()
        print("answers: ", answers)
        if answers and len(answers) > 0:
            return answers[0].text
        if r.get_count() > 0:
            print("answers2: ", "\n".join(d['content'] for d in r))
            return "\n".join(d['content'] for d in r)

        return None        

    def run(self, q: str, overrides: dict) -> any:
        # Not great to keep this as instance state, won't work with interleaving (e.g. if using async), but keeps the example simple
        self.results = None

        # Use to capture thought process during iterations
        cb_handler = HtmlCallbackHandler()
        cb_manager = CallbackManager(handlers=[cb_handler])

        llm = AzureOpenAI(deployment_name=self.openai_deployment, temperature=overrides.get("temperature") or 0.3, openai_api_key=openai.api_key)
        tools = [
            Tool(name="Search", func=lambda q: self.search(q, overrides)),
            Tool(name="Lookup", func=self.lookup),
        ]

        # Like results above, not great to keep this as a global, will interfere with interleaving
        global prompt
        prompt_prefix = overrides.get("prompt_template")
        prompt = PromptTemplate.from_examples(
            EXAMPLES, SUFFIX, ["input", "agent_scratchpad"], prompt_prefix + "\n\n" + PREFIX if prompt_prefix else PREFIX)
        
        agent = ReAct.from_llm_and_tools(llm, tools)
        #hana debug Agent の反復回数を制限
        #https://langchain.readthedocs.io/en/latest/modules/agents/examples/max_iterations.html#max-iterations
        chain = AgentExecutor.from_agent_and_tools(agent, tools, verbose=True, callback_manager=cb_manager, max_iterations=3, early_stopping_method="generate")
        result = chain.run(q)
        
        
        # Fix up references to they look like what the frontend expects ([] instead of ()), need a better citation format since parentheses are so common
        result = result.replace("(", "[").replace(")", "]")

        return {"data_points": self.results or [], "answer": result, "thoughts": cb_handler.get_and_reset_log()}
    
class ReAct(ReActDocstoreAgent):
    @classmethod
    def create_prompt(cls, tools: List[Tool]) -> BasePromptTemplate:
        return prompt
    
# 情報源の引用方法に関する指示と例を含む、langchainのReActプロンプトの修正版
# Modified version of langchain's ReAct prompt that includes instructions and examples for how to cite information sources
# hana debug 日本語で使う場合、デフォルトプロンプトの例が長すぎるので短くする！
EXAMPLES = [
"""Question: What profession does Nicholas Ray and Elia Kazan have in common?
Thought 1: I need to search Nicholas Ray and Elia Kazan, find their professions, then
find the profession they have in common.
Action 1: Search[Nicholas Ray]
Observation 1: [files-987.png] Nicholas Ray (born Raymond Nicholas Kienzle Jr., August 7, 1911 - June 16,
1979) was an American film director, screenwriter, and actor best known for
the 1955 film Rebel Without a Cause.
Thought 2: Professions of Nicholas Ray are director, screenwriter, and actor. I need
to search Elia Kazan next and find his professions.
Action 2: Search[Elia Kazan]
Observation 2: [files-654.txt] Elia Kazan was an American film and theatre director, producer, screenwriter
and actor.
Thought 3: Professions of Elia Kazan are director, producer, screenwriter, and actor.
So profession Nicholas Ray and Elia Kazan have in common is director,
screenwriter, and actor.
Action 3: Finish[director, screenwriter, actor (files-987.png)(files-654.txt) ]""",
]

SUFFIX = """\nQuestion: {input}
{agent_scratchpad}"""
#PREFIX = "Answer questions as shown in the following examples, by splitting the question into individual search or lookup actions to find facts until you can answer the question. " \
#"Observations are prefixed by their source name in square brackets, source names MUST be included with the actions in the answers." \
#"Only answer the questions using the information from observations, do not speculate."
PREFIX = "以下の例のように、質問を分割して検索やルックアップを行うことで、質問に答えられるまで事実を探すことができます。" \
"観察は、そのソース名を角括弧で前置きする。ソース名は、回答中のアクションと一緒に記載しなければならない。" \
"各出典元には、名前の後にコロンと実際の情報があり、回答で使用する各事実には必ず出典名を記載してください。ソースを参照するには、四角いブラケットを使用します。例えば、[info1.txt]です。出典を組み合わせず、各出典を別々に記載すること。例えば、[info1.txt][info2.pdf] など。" \
"観察から得た情報を使って質問に答えるだけで、推測はしないでください。絶対に観察以外の情報を使ってはいけません。" \
"回答に例文を使ってはいけません。"
