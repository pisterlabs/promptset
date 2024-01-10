"""Chat agent with question answering

"""
import os
import re
import logging
# from utils.giphy import GiphyAPIWrapper
from dataclasses import dataclass

from langchain.chains import LLMChain, LLMRequestsChain
from langchain import Wikipedia, OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.agents.react.base import DocstoreExplorer
from langchain.agents import Tool, AgentExecutor, load_tools, initialize_agent, get_all_tool_names
from langchain.prompts import PromptTemplate
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.agents.conversational.base import ConversationalAgent
from datetime import datetime

from core.registry.personalities import get_personality, default_personality
from core.config import get_value

import langchain
from langchain.cache import InMemoryCache
langchain.llm_cache = InMemoryCache()

logger = logging.getLogger(__name__)


# news_api_key = os.environ["NEWS_API_KEY"]
# tmdb_bearer_token = os.environ["TMDB_API_KEY"]


@dataclass
class ChatAgent:
    agent_executor: AgentExecutor = None
    memory: ConversationBufferMemory = None
    personality_name: str = None

    def _get_docstore_agent(self):
        docstore = DocstoreExplorer(Wikipedia())
        docstore_tools = [
            Tool(
                name="Search",
                func=docstore.search,
                description="Search wikipedia"
            ),
            Tool(
                name="Lookup",
                func=docstore.lookup,
                description="Lookup a wikipedia page"
            )
        ]
        docstore_llm = OpenAI(temperature=0, openai_api_key=get_value(
            'providers.openai.apiKey'))
        docstore_agent = initialize_agent(
            docstore_tools, docstore_llm, agent="react-docstore", verbose=True)
        return docstore_agent

    def _get_requests_llm_tool(self):

        template = """
        Extracted: {requests_result}"""

        PROMPT = PromptTemplate(
            input_variables=["requests_result"],
            template=template,
        )

        def lambda_func(input):
            out = chain = LLMRequestsChain(llm_chain=LLMChain(
                llm=OpenAI(temperature=0, openai_api_key=get_value(
                    'providers.openai.apiKey')),
                prompt=PROMPT)).run(input)
            return out.strip()
        return lambda_func

    def __init__(self, *, conversation_chain: LLMChain = None, history_array, personality_name: str = None):
        self.personality_name = personality_name
        date = datetime.today().strftime('%B %d, %Y at %I:%M %p')

        # set up a Wikipedia docstore agent
        docstore_agent = self._get_docstore_agent()

        # giphy = GiphyAPIWrapper()

        # tool_names = get_all_tool_names()
        print(get_all_tool_names())
        tool_names = [
            # 'serpapi',
            # 'wolfram-alpha',
            'llm-math',
            'open-meteo-api',
            'news-api',
            # 'tmdb-api',
            'wikipedia'
        ]

        requests_tool = self._get_requests_llm_tool()

        tools = load_tools(tool_names,
                           llm=OpenAI(temperature=0, openai_api_key=get_value(
                               'providers.openai.apiKey')),
                           news_api_key=get_value('providers.news.apiKey')
                           )
        # news_api_key=news_api_key,
        # tmdb_bearer_token=tmdb_bearer_token)

        # Tweak some of the tool descriptions
        for tool in tools:
            if tool.name == "Search":
                tool.description = "Use this tool exclusively for questions relating to current events, or when you can't find an answer using any of the other tools."
            if tool.name == "Calculator":
                tool.description = "Use this to solve numeric math questions and do arithmetic. Don't use it for general or abstract math questions."

        tools = tools + [
            Tool(
                name="WikipediaSearch",
                description="Useful for answering a wide range of factual, scientific, academic, political and historical questions.",
                func=docstore_agent.run
            ),
            # Tool(
            #     name="GiphySearch",
            #     func=giphy.run,
            #     return_direct=True,
            #     description="useful for when you need to find a gif or picture, and for adding humor to your replies. Input should be a query, and output will be an html embed code which you MUST include in your Final Answer."
            # ),
            Tool(
                name="Requests",
                func=requests_tool,
                description="A portal to the internet. Use this when you need to get specific content from a site. Input should be a specific url, and the output will be all the text on that page."
            )
        ]
        ai_prefix = personality_name or default_personality
        human_prefix = get_value('common.user.name', "Human")

        suffix = f"""
The current date is {date}. Questions that refer to a specific date or time period will be interpreted relative to this date.

Questions that refer to a specific date or time period will be interpreted relative to this date.

After you answer the question, you MUST to determine which langauge your answer is written in, and append the language code to the end of the Final Answer, within parentheses, like this (en-US).

Begin!

Previous conversation history:
{{chat_history}}

New input: {{input}}

{{agent_scratchpad}}
"""

        self.memory = ConversationBufferMemory(memory_key="chat_history")
        for item in history_array:
            self.memory.save_context(
                {f"{ai_prefix}": item["prompt"]}, {f"{human_prefix}": item["response"]})

        llm = OpenAI(temperature=.5, openai_api_key=get_value(
            'providers.openai.apiKey'))
        llm_chain = LLMChain(
            llm=llm,
            prompt=ConversationalAgent.create_prompt(
                tools,
                # prefix=prefix,
                ai_prefix=ai_prefix,
                human_prefix=human_prefix,
                suffix=suffix
            ),
        )

        agent_obj = ConversationalAgent(
            llm_chain=llm_chain, ai_prefix=ai_prefix)

        self.agent_executor = AgentExecutor.from_agent_and_tools(
            agent=agent_obj,
            tools=tools,
            verbose=True,
            max_iterations=5,
            memory=self.memory,)

        # self.agent_executor = AgentExecutor.from_agent_and_tools(
        #     agent=agent,
        #     tools=tools,
        #     max_iterations=5,
        #     verbose=True)

    def express(self, input: str, lang: str = "en-US"):
        personality_llm = ChatOpenAI(temperature=0, openai_api_key=get_value(
            'providers.openai.apiKey'))

        personality = get_personality(self.personality_name)

        personality_prompt = PromptTemplate(
            input_variables=["original_words"],
            partial_variables={
                "lang": lang
            },
            template=personality["name"] +
            " is a " + personality["prompt"] +
            " Restate the following as " +
            personality["name"] +
            " would in {lang}: \n{original_words}\n",
        )

        self.express_chain = LLMChain(
            llm=personality_llm, prompt=personality_prompt, verbose=True, memory=self.memory)

        return self.express_chain.run(input)

    def run(self, input):
        try:
            result = self.agent_executor.run(input)

            pattern = r'\(([a-z]{2}-[A-Z]{2})\)'
            # Search for the local pattern in the string
            match = re.search(pattern, result)

            language = 'en-US'  # defaut
            if match:
                # Get the language code
                language = match.group(1)

                # Remove the language code from the reply
                result = re.sub(pattern, '', result)

            logger.info('Got result from agent: ' + result)

            # TODO: this method is not optimum, but it works for now
            reply = self.express(result, language)
            # reply = self.express_chain.run(result)

            logger.info('Answer from express chain: ' + reply)

            reply = reply.replace('"', '')

        except ValueError as inst:
            print("ValueError: \n\n")
            print(inst)
            reply = "I don't understand what you're saying. Please try again."

        except Exception as e:
            print(e)
            logger.exception(e)
            reply = "I'm sorry, I'm having trouble understanding you. Please try again."

        return reply
