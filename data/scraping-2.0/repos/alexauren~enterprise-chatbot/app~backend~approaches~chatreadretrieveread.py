import openai
from approaches.approach import Approach
from azure.search.documents import SearchClient
from azure.search.documents.models import QueryType
from langchain.chat_models import AzureChatOpenAI
from langchain.callbacks.manager import CallbackManager
from langchain.agents import Tool, AgentType, initialize_agent, ConversationalChatAgent
from langchain.memory import ConversationBufferMemory
from langchainadapters import HtmlCallbackHandler
from text import nonewlines
from typing import Any, Sequence


class ChatReadRetrieveReadApproach(Approach):
    """
    Attempt to answer questions by iteratively evaluating the question to see what information is missing, and once all information
    is present then formulate an answer. Each iteration consists of two parts:
     1. use GPT to see if we need more information
     2. if more data is needed, use the requested "tool" to retrieve it.
    The last call to GPT answers the actual question.
    This is inspired by the MKRL paper[1] and applied here using the implementation in Langchain.

    [1] E. Karpas, et al. arXiv:2205.00445
    """
    content: str = ""

    # A message sent from the perspective of the human
    human_message: str = """
TOOLS
------
You can use tools to look up information that may be helpful in answering the users question.

{format_instructions}

Only use information from the sources below. If you need to use information from another source, tell the user you don't know and that they should contact customer support.
{sources}

USER'S INPUT
--------------------
Here is the user's input (remember to respond with a markdown code snippet of a json blob with a single action, and NOTHING else):

{input}"""


    format_instructions = """To use a tool, please use the following format:

    ```
    Thought: Do I need to use a tool? Yes
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ```
    
    When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:
    
    ```
    Thought: Do I need to use a tool? No
    Answer: Your final answer. 
    """
    
    # A message setting the objectives the AI should follow
    system_message: str = "You are an intelligent and helpful assistant. Your name is Floyd. Your job is helping DNB Bank ASA customers with their questions about insurance." \
    "If the question is incomplete, ask the user for more information."  \
    "Only answer questions relevant to DNB house insurance. If the user asks about other insurance providers, say you don't know."  \
    "For information in table format return it as an html table. Do not return markdown format."  \
    "Each source has a name followed by colon and the actual data, quote the source name for each piece of data you use in the response."  \
    "For example, if the question is \ What color is the sky? \ and one of the information sources says \info123: the sky is blue whenever it's not cloudy\, then answer with \The sky is blue [info123]\ "  \
    "It's important to strictly follow the format where the name of the source is in square brackets at the end of the sentence, and only up to the prefix before the colon (\:\)."  \
    "If there are multiple sources, cite each one in their own square brackets. For example, use \[info343][ref-76]\ and not \[info343,ref-76]\."  \
    "Never quote tool names or chat history as sources." \
    f"{human_message}"


    CognitiveSearchToolDescription = "Useful for searching for public information about DNB house insurance."

    def __init__(self, search_client: SearchClient, chatgpt_deployment: str, sourcepage_field: str, content_field: str):
        self.search_client = search_client
        self.chatgpt_deployment = chatgpt_deployment
        self.sourcepage_field = sourcepage_field
        self.content_field = content_field

    def retrieve(self, q: str, overrides: dict[str, Any]) -> Any:
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
        self.content = "\n".join(self.results)
        return self.content
    
    def askUser(self, q: str) -> Any:
        return q
        
    def run(self, history: Sequence[dict[str, str]], overrides: dict[str, Any]) -> Any:
        # Not great to keep this as instance state, won't work with interleaving (e.g. if using async), but keeps the example simple
        self.results = None

        # Use to capture thought process during iterations
        cb_handler = HtmlCallbackHandler()
        cb_manager = CallbackManager(handlers=[cb_handler])
        
        acs_tool = Tool(name="CognitiveSearch", 
                        func=lambda q: self.retrieve(q, overrides), 
                        description=self.CognitiveSearchToolDescription,
                        callbacks=cb_manager)
       
        tools: Sequence = [acs_tool]

        temp_human_message=self.human_message.format(tools=tools, format_instructions=self.format_instructions.format(tool_names=", ".join([t.name for t in tools])), sources=self.sourcepage_field, input=history[-1].get("user"))

        # memory = ConversationBufferMemory(memory_key = "chat_history", 
        #                               input_key = "input",
        #                               output_key = "output", 
        #                               return_messages = True,
        #                               human_prefix=human_message)

        prompt = ConversationalChatAgent.create_prompt(
            system_message=self.system_message,
            human_message=temp_human_message,
            tools=tools,
            input_variables=["input", "agent_scratchpad", "chat_history"])

        print(prompt)

        llm = AzureChatOpenAI(deployment_name=self.chatgpt_deployment, 
                              temperature=0, 
                              openai_api_key=openai.api_key, 
                              openai_api_base=openai.api_base, 
                              openai_api_version=openai.api_version
                              )

        conversational_agent = initialize_agent(
            agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            tools=tools, 
            llm=llm,
            verbose=True,
            max_iterations=5,
            memory=ConversationBufferMemory(memory_key = "chat_history", 
                                      input_key = "input",
                                      output_key = "output", 
                                      return_messages = True),
            agent_kwargs= {
                "system_message": self.system_message,
                "human_message": temp_human_message
                }
            )
        print(history)
        result = conversational_agent.run(history[-1].get("user"))
        
        
        # Remove references to tool names that might be confused with a citation
        result = result.replace("[CognitiveSearch]", "")
        return {"data_points": self.results or [], "answer": result, "thoughts": cb_handler.get_and_reset_log()}
    