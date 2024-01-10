from __future__ import annotations
from typing import List

from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains.llm import LLMChain
from langchain.chat_models.base import BaseChatModel
from langchain.schema import (
    AIMessage,
    BaseMessage,
    Document,
    HumanMessage,
    SystemMessage,
)
from langchain.vectorstores.base import VectorStoreRetriever

import faiss

from command_gpt.utils.console_logger import ConsoleLogger
from command_gpt.prompting.ruleset_prompt import RulesetPrompt


class RulesetGeneratorAgent:
    """Driving class for Agent that generates a ruleset for CommandGPT"""

    def __init__(
        self,
        request: str,
        topic: str,
        chain: LLMChain,
        memory: VectorStoreRetriever = None,
    ):
        self.request = request
        self.topic = topic
        self.user_feedback: List[str] = []
        self.generated_ruleset: str = None
        self.memory = memory
        self.full_message_history: List[BaseMessage] = []
        self.next_action_count = 0
        self.chain = chain

    @classmethod
    def from_request_and_topic(
        cls,
        request: str,
        topic: str,
        llm: BaseChatModel,
    ) -> RulesetGeneratorAgent:
        """
        Instantiate RulesetGeneratorAgent from request and topic
        :param request: Inserted as "Generate an input summary that will instruct CommandGPT to {request}: "
        :param topic: Inserted as "...instruct CommandGPT to {request}: {topic}"
        """
        prompt = RulesetPrompt(
            ruleset_that_will=request,
            topic=topic,
            input_variables=["memory", "messages"],
            token_counter=llm.get_num_tokens,
        )

        # Define your embedding model
        embeddings_model = OpenAIEmbeddings()
        # Initialize the vectorstore as empty
        embedding_size = 1536
        index = faiss.IndexFlatL2(embedding_size)
        vectorstore = FAISS(embeddings_model.embed_query,
                            index, InMemoryDocstore({}), {})

        chain = LLMChain(llm=llm, prompt=prompt)
        return cls(
            request=request,
            topic=topic,
            chain=chain,
            memory=vectorstore.as_retriever()
        )

    @classmethod
    def from_request_prompt_for_topic(
        cls,
        request: str,
        prompt_for_topic: str,
        llm: BaseChatModel,
    ) -> RulesetGeneratorAgent:
        """
        Get topic from user and instantiate RulesetGeneratorAgent from request and topic
        :param request: Inserted as "Generate an input summary that will instruct CommandGPT to {request}: "
        :param prompt_for_topic: Input prompt displayed to user for getting topic
        """

        # Get topic from user
        topic = ConsoleLogger.input(f"{prompt_for_topic}: ")

        return RulesetGeneratorAgent.from_request_and_topic(
            request=request,
            topic=topic,
            llm=llm
        )

    @classmethod
    def from_empty_prompt_for_request_and_topic(
        cls,
        llm: BaseChatModel,
    ) -> RulesetGeneratorAgent:
        """
        Get request & topic both from user and instantiate RulesetGeneratorAgent
        """

        # Get request from user
        request = ConsoleLogger.input(
            f"Generate an input summary that will instruct CommandGPT to (e.g. \"research and generate reports on \"): ")
        topic = ConsoleLogger.input(f"Topic (e.g. \"Tardigrades\"): ")

        return RulesetGeneratorAgent.from_request_and_topic(
            request=request,
            topic=topic,
            llm=llm
        )

    def run(self) -> str:
        """
        Kicks off interaction loop with AI
        """
        # Note about ruleset generator
        ConsoleLogger.log("\nNOTE: The ruleset generator is a work in progress. Providing feedback for refined rulesets is experimental; keep it concise for best results. Feedback is appreciated :)\n", color=ConsoleLogger.COLOR_MAGENTA)
        # Interaction Loop
        loop_count = 0
        while True:
            # user_input = ConsoleLogger.input("You: ")
            loop_count += 1

            messages = self.full_message_history
            messages.append(
                SystemMessage(
                    content=f"Loop count: {loop_count}"
                )
            )

            # Get user input beyond initial prompt
            if (loop_count > 1):
                user_input = ConsoleLogger.input(
                    "Type (y) and hit enter if this ruleset is acceptable. Otherwise, provide feedback for a new one: ")
                if user_input == "y":
                    return self.generated_ruleset
                else:
                    self.user_feedback.append(user_input)

            # Update prompt with user feedback if exists
            if self.user_feedback:
                # todo: use user feedback in a more fine tuned way
                prompt = RulesetPrompt(
                    ruleset_that_will=self.request,
                    topic=self.topic,
                    generated_ruleset=self.generated_ruleset,
                    input_variables=["memory", "messages"],
                    token_counter=self.chain.llm.get_num_tokens,
                    user_feedback=user_input
                )
                self.chain.prompt = prompt

            ConsoleLogger.log(
                f"FULL PROMPT:\n\n{self.chain.prompt.construct_full_prompt()}\n",
                color=ConsoleLogger.COLOR_INPUT
            )

            # Set response color for console logger
            ConsoleLogger.set_response_stream_color()
            # Send message to AI, get response
            self.generated_ruleset = self.chain.run(
                messages=messages,
                memory=self.memory,
                user_input="You're doing great. Without any other niceties, provide a ruleset with no additional text before or after. Your message should start with \"You are xxx-GPT...\"",
            )

            # Update message history
            self.full_message_history.append(HumanMessage(
                content="Generate a new ruleset consistent with the original request and the user's feedback."
            ))
            self.full_message_history.append(
                AIMessage(content=self.generated_ruleset))

            self.memory.add_documents([Document(
                page_content=self.generated_ruleset,
                metadata={
                    "ruleset_count": f"{loop_count}"
                }
            )])
