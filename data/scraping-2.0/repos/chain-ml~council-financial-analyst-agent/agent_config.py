from typing import List
import os
from string import Template

from council.chains import Chain
from council.evaluators import BasicEvaluator
from council.llm import OpenAILLM, LLMMessage
from council.skills import LLMSkill, PromptToMessages
from council.contexts import SkillContext
from council.prompt import PromptBuilder
from council.runners import Parallel

import constants
from config import Config
from skills import (
    DocRetrievalSkill,
    GoogleAggregatorSkill,
    PandasSkill,
    CustomGoogleNewsSkill,
    CustomGoogleSearchSkill,
)
from retrieval import Retriever
from controller import Controller
from filter import LLMFilter

import dotenv

dotenv.load_dotenv()


class AgentConfig:
    def __init__(self):
        # Initializing document retrieval dependencies
        self.config = Config(
            encoding_name=constants.ENCODING_NAME,
            embedding_model_name=constants.EMBEDDING_MODEL_NAME,
        )
        self.index = self.config.initialize()
        self.index_retriever = self.index.as_retriever(
            similarity_top_k=constants.NUM_RETRIEVED_DOCUMENTS
        )
        self.retriever = Retriever(self.config, self.index_retriever)

        # Initializing agent config
        self._llm_skill_model = OpenAILLM.from_env(
            model=constants.DOC_AND_GOOGLE_RETRIEVAL_LLM
        )
        self._controller_model = OpenAILLM.from_env(model=constants.CONTROLLER_LLM)
        self._init_skills()
        self.chains = self._init_chains()
        self.controller = Controller(
            llm=self._controller_model, chains=self.chains, response_threshold=5
        )
        self.evaluator = BasicEvaluator()
        self.filter = LLMFilter(llm=self._controller_model)

    def load_config(self):
        return {
            "controller": self.controller,
            "evaluator": self.evaluator,
            "filter": self.filter,
        }

    def _init_skills(self):
        # Document retrieval skills
        self.doc_retrieval_skill = DocRetrievalSkill(self.retriever)

        # Search skills
        self.google_search_skill = CustomGoogleSearchSkill()
        self.google_news_skill = CustomGoogleNewsSkill()
        self.google_aggregator_skill = GoogleAggregatorSkill()

        # Pandas skills
        self.pandas_skill = PandasSkill(
            api_token=os.getenv("OPENAI_API_KEY"), model=constants.PANDAS_LLM
        )

        # LLM Skill
        self.llm_skill = LLMSkill(
            llm=self._llm_skill_model,
            system_prompt=Template(
                "You are a financial analyst whose job is to answer user questions about $company with the provided context."
            ).substitute(company=constants.COMPANY_NAME),
            context_messages=self._build_context_messages,
        )

    def _init_chains(self) -> List[Chain]:
        self.doc_retrieval_chain = Chain(
            name="doc_retrieval_chain",
            description=f"Information from {constants.COMPANY_NAME} ({constants.COMPANY_TICKER}) 10-K from their 2022 fiscal year, a document that contain important updates for investors about company performance and operations",
            runners=[self.doc_retrieval_skill, self.llm_skill],
        )

        self.search_chain = Chain(
            name="search_chain",
            description=f"Information about {constants.COMPANY_NAME} ({constants.COMPANY_TICKER}) using a Google search",
            runners=[
                Parallel(self.google_search_skill, self.google_news_skill),
                self.google_aggregator_skill,
                self.llm_skill,
            ],
        )

        self.pandas_chain = Chain(
            name="pandas_chain",
            description=f"{constants.COMPANY_NAME} ({constants.COMPANY_TICKER}) historical stock price and trading data information",
            runners=[self.pandas_skill],
        )

        return [self.doc_retrieval_chain, self.search_chain, self.pandas_chain]

    @staticmethod
    def _build_context_messages(context: SkillContext) -> List[LLMMessage]:
        """Context messages function for LLMSkill"""

        prompt = """Use the following pieces of context to answer the query.
        If the answer is not provided in the context, do not make up an answer. Instead, respond that you do not know.

        CONTEXT:
        {{chain_history.last_message}}
        END CONTEXT.

        QUERY:
        {{chat_history.user.last_message}}
        END QUERY.

        YOUR ANSWER:
        """
        context_message_prompt = PromptToMessages(prompt_builder=PromptBuilder(prompt))

        return context_message_prompt.to_user_message(context)
