"""
File that contains the logic for the news relevancy check.
"""
import logging
import re
from typing import Literal, Any

from langchain.agents import AgentType, initialize_agent, AgentExecutor
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain.agents import Tool
from langchain import SerpAPIWrapper
from langchain.utilities import WikipediaAPIWrapper

from src.logic.langchain_tools.tool_process_thought import process_thoughts
from src.logic.config import secrets as config_secrets

class RelevancyChecker():
    """
    Class that contains the logic for the news relevancy check.
    """
    
    def __init__(self):
        self.system_template: Literal = """system message template"""
        self.system_message_prompt: SystemMessagePromptTemplate = SystemMessagePromptTemplate.from_template(self.system_template)
        self.few_shot_human: SystemMessagePromptTemplate = SystemMessagePromptTemplate.from_template("human message example template", additional_kwargs={"name": "example_user"})
        self.few_shot_ai: SystemMessagePromptTemplate = SystemMessagePromptTemplate.from_template("""ai message example template""", additional_kwargs={"name": "example_assistant"})
        self.human_template: Literal = """human message template"""
        self.human_message_prompt: HumanMessagePromptTemplate = HumanMessagePromptTemplate.from_template(self.human_template)
        self.chat_prompt: ChatPromptTemplate = ChatPromptTemplate.from_messages(
            [self.system_message_prompt, self.few_shot_human, self.few_shot_ai, self.human_message_prompt]
        )
        self.llm: ChatOpenAI = ChatOpenAI(
            model="gpt-4",
            temperature = 0,
            client = self.chat_prompt,
            openai_api_key = config_secrets.read_openai_credentials()
        )
        self.search: SerpAPIWrapper = SerpAPIWrapper(serpapi_api_key = config_secrets.read_serpapi_credentials())
        self.wikipedia: WikipediaAPIWrapper = WikipediaAPIWrapper()
        self.tools = [
            Tool(
                name = "Search",
                func = self.search.run,
                description = "useful for when you need to answer questions about current events"
            ),
            Tool(
                name = "Thought Processing",
                func = process_thoughts,
                description = """useful for when you have a thought that you want to use in a task, 
                but you want to make sure it's formatted correctly"""
            ),
            Tool(
                name = "Wikipedia",
                func = self.wikipedia.run,
                description = "useful for when you need to detailed information about a topic"
            )
        ]
        self.agent: AgentExecutor = initialize_agent(
            tools = self.tools, 
            llm = self.llm, 
            agent = AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, 
            verbose = True
        )

    def check_relevancy(self, company: str, news: str) -> int:
        """
        Check whether a news article is relevant to a company.

        :param company: The company the news should be relevant for.
        :param news: The news article to check.
        :return: 0 if the news article is relevant, 1 otherwise.
        :raise ValueError: If arg company is not a string or if the string is empty.
        :raise ValueError: If arg news is not a dictionary or if the dictionary is empty.
        """
        if not isinstance(company, str) or not company:
            raise ValueError("Argument company must be a non empty string.")
        if not isinstance(news, str) or not news:
            raise ValueError("Argument news must be a non empty string.")
        try:
            self.template = """You are a risk analyst and your task is to evaluate the relevance of 
            a news article to your company. You will be provided with the company name and the corresponding news article. 
            Your final answer should include two sections: a. Relevancy: True (directly or indirectly relevant to the company) 
            or False. b. Explanation: Provide the reasoning behind your answer. Keep in mind that news mentioning aspects such
            as competitors or their products/services in the same market can still be relevant even when not mentioning the 
            company name directly."""
            few_shot_company: Literal = "Gucci"
            few_shot_article: Literal = """In a recent sighting that has sent fans into a frenzy, the beloved British singer and 
            style icon, Harry Styles, was seen sporting a stunning Gucci tee while expressing his infatuation with the 
            polka dotted franchise. The sighting took place in Los Angeles, where Styles is currently working on his 
            highly anticipated second album."""
            few_shot_answer: Literal = """Relevancy: True. Explanation: The news article is relevant to the company Gucci because 
            it mentions a signature Gucci tee worn by Harry Styles and Gucci is currently active in the news with a 
            partnership in the metaverse-oriented space."""
            self.system_message_prompt: SystemMessagePromptTemplate = SystemMessagePromptTemplate.from_template(self.template)
            self.few_shot_human = SystemMessagePromptTemplate.from_template("""Please rate the relevancy of news: {few_shot_article} for the
            company: {few_shot_company}.""", additional_kwargs={"name": "example_user"})
            self.few_shot_ai = SystemMessagePromptTemplate.from_template("{few_shot_answer}", additional_kwargs={"name": "example_assistant"})
            self.human_template = """Please rate the relevancy of a news article for the company: {company}. 
            News article: {news}"""
            self.human_message_prompt: HumanMessagePromptTemplate = HumanMessagePromptTemplate.from_template(self.human_template)
            self.chat_prompt = ChatPromptTemplate.from_messages(
                [self.system_message_prompt, self.few_shot_human, self.few_shot_ai, self.human_message_prompt]
            )
            relevancy: str = self.agent.run(
                self.chat_prompt.format_messages(company = company, news = news, few_shot_article = few_shot_article, few_shot_company = few_shot_company, few_shot_answer = few_shot_answer)
            )
            is_relevant_result: str | Any = re.search(r"Relevancy: (True|False)", relevancy).group(1)
            if is_relevant_result == "True":
                is_relevant: int = 0
            else:
                is_relevant: int = 1
        except ValueError as e:
            logging.error(e)
            raise ValueError(f"Error: {e}") from e
        return is_relevant
