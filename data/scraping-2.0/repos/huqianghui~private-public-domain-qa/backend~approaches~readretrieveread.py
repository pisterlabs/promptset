import openai
from approaches.approach import Approach
from azure.search.documents import SearchClient
from azure.search.documents.models import QueryType
from langchain.llms.openai import AzureOpenAI
from langchain.callbacks.base import CallbackManager
from langchain.chains import LLMChain
from langchain.agents import Tool, ZeroShotAgent, AgentExecutor
from langchain.llms.openai import AzureOpenAI
from langchainadapters import HtmlCallbackHandler
from text import nonewlines
from lookuptool import CsvLookupTool

# Attempt to answer questions by iteratively evaluating the question to see what information is missing, and once all information
# is present then formulate an answer. Each iteration consists of two parts: first use GPT to see if we need more information, 
# second if more data is needed use the requested "tool" to retrieve it. The last call to GPT answers the actual question.
# This is inspired by the MKRL paper[1] and applied here using the implementation in Langchain.
# [1] E. Karpas, et al. arXiv:2205.00445
class ReadRetrieveReadApproach(Approach):

    template_prefix = \
"你是一个智能助手，可帮助员工解决食品检测行业问题。" \
"仅使用以下信息来源中提供的数据来回答问题。" \
"每个源都有一个名称，后跟冒号和实际数据，请引用您在响应中使用的每条数据的源名称。 " \
"例如，如果问题是\"天空是什么颜色的？\"，并且其中一个信息源说\"info123:只要不多云，天空就是蓝色的\"，则回答\"天空是蓝色的[info123]\"。"\
"请务必严格遵循以下格式：源名称在句子末尾的方括号中，并且只能到冒号之前的前缀 （\"：\"）。" \
"如果有多个来源，请在各自的方括号中引用每个来源。例如，使用 \"[info343][ref-76]\" 而不是 \"[info343，ref-76]\"。" \
"切勿引用工具名称作为来源。" \
"如果您无法使用以下来源回答，请说您不知道。" \
"\n\n您可以访问以下工具:"
    
    template_suffix = """
开始!

问题: {input}

思考: {agent_scratchpad}"""    

    CognitiveSearchToolDescription = "可用于搜索食品检测行业信息。"

    def __init__(self, search_client: SearchClient, openai_deployment: str, sourcepage_field: str, content_field: str):
        self.search_client = search_client
        self.openai_deployment = openai_deployment
        self.sourcepage_field = sourcepage_field
        self.content_field = content_field

    def retrieve(self, q: str, overrides: dict) -> any:
        use_semantic_captions = True if overrides.get("semantic_captions") else False
        top = overrides.get("top") or 3
        exclude_category = overrides.get("exclude_category") or None
        filter = "category ne '{}'".format(exclude_category.replace("'", "''")) if exclude_category else None

        if overrides.get("semantic_ranker"):
            r = self.search_client.search(q,
                                          filter=filter, 
                                          query_type=QueryType.SEMANTIC, 
                                          query_language="en-us", 
                                          query_speller="lexicon", 
                                          semantic_configuration_name="default", 
                                          top = top,
                                          query_caption="extractive|highlight-false" if use_semantic_captions else None)
        else:
            r = self.search_client.search(q, filter=filter, top=top)
        if use_semantic_captions:
            self.results = [doc[self.sourcepage_field] + ":" + nonewlines(" -.- ".join([c.text for c in doc['@search.captions']])) for doc in r]
        else:
            self.results = [doc[self.sourcepage_field] + ":" + nonewlines(doc[self.content_field][:250]) for doc in r]
        content = "\n".join(self.results)
        return content
        
    def run(self, q: str, overrides: dict) -> any:
        # Not great to keep this as instance state, won't work with interleaving (e.g. if using async), but keeps the example simple
        self.results = None

        # Use to capture thought process during iterations
        cb_handler = HtmlCallbackHandler()
        cb_manager = CallbackManager(handlers=[cb_handler])
        
        acs_tool = Tool(name = "CognitiveSearch", func = lambda q: self.retrieve(q, overrides), description = self.CognitiveSearchToolDescription)
        inspection_tool = InspectionTool("InspectionTool")
        tools = [acs_tool, inspection_tool]

        prompt = ZeroShotAgent.create_prompt(
            tools=tools,
            prefix=overrides.get("prompt_template_prefix") or self.template_prefix,
            suffix=overrides.get("prompt_template_suffix") or self.template_suffix,
            input_variables = ["input", "agent_scratchpad"])
        llm = AzureOpenAI(deployment_name=self.openai_deployment, temperature=overrides.get("temperature") or 0.3, openai_api_key=openai.api_key)
        chain = LLMChain(llm = llm, prompt = prompt)
        agent_exec = AgentExecutor.from_agent_and_tools(
            agent = ZeroShotAgent(llm_chain = chain, tools = tools),
            tools = tools, 
            verbose = True, 
            callback_manager = cb_manager)
        result = agent_exec.run(q)
                
        # Remove references to tool names that might be confused with a citation
        result = result.replace("[CognitiveSearch]", "").replace("[Employee]", "")

        return {"data_points": self.results or [], "answer": result, "thoughts": cb_handler.get_and_reset_log()}


class InspectionTool(CsvLookupTool):
    content: str = ""

    def __init__(self, content: str):
        super().__init__(filename = "data/sgs-prompts-testing.csv", key_field = "content", name = "Content", description = "useful for answering questions about the inspection information")
        self.func = self.inspection_info
        self.content = content

    def inspection_info(self, unused: str) -> str:
        return self.lookup(self.content)
