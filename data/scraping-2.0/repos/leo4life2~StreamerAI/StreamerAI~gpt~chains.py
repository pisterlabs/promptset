import logging
import os

from langchain import LLMChain, PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from tenacity import retry, wait_random_exponential, stop_after_attempt
from pathlib import Path

from StreamerAI.database.database import Persona
from StreamerAI.settings import PRODUCT_CONTEXT_SWITCH_SIMILARITY_THRESHOLD, LLM_NAME, LLM_TEMPERATURE
from StreamerAI.gpt.retrieval import Retrieval, SimpleRetrieval


class Chains:
    """A class representing a collection of language model chains used for responding to user queries."""

    chatid_to_chain_prevcontext = {}
    retrieval = SimpleRetrieval()
    
    @classmethod
    def create_chain(cls, temperature=LLM_TEMPERATURE, verbose=False, prompt_type='qa'):
        """Create and return a new language model chain.

        Args:
            temperature (float): the sampling temperature to use when generating responses
            verbose (bool): whether or not to print debugging information

        Returns:
            LLMChain: the newly created language model chain
        """
        persona = Persona.select().where(Persona.current == True).first()

        if prompt_type == "qa":
            prompt_template = persona.qa_prompt
            prompt = PromptTemplate(
                input_variables=["history", "human_input", "product_context", "other_available_products", "audience_name"],
                template=prompt_template
            )
            chatgpt_chain = LLMChain(
                llm=ChatOpenAI(model_name=LLM_NAME, temperature=temperature),
                prompt=prompt,
                verbose=verbose,
                memory=ConversationBufferWindowMemory(k=3, memory_key="history", input_key="human_input"), # only keep the last 3 interactions
            )
            return chatgpt_chain
        elif prompt_type == "conversation":
            prompt_template = persona.conversation_prompt
            prompt = PromptTemplate(
                input_variables=["human_input"],
                template=prompt_template
            )
            chatgpt_chain = LLMChain(
                llm=ChatOpenAI(model_name=LLM_NAME, temperature=temperature),
                prompt=prompt,
                verbose=verbose,
            )
            return chatgpt_chain
        elif prompt_type == "new_viewer":
            prompt_template = persona.new_viewer_prompt
            prompt = PromptTemplate(
                input_variables=["audience_name"],
                template=prompt_template
            )
            chatgpt_chain = LLMChain(
                llm=ChatOpenAI(model_name=LLM_NAME, temperature=temperature),
                prompt=prompt,
                verbose=verbose,
            )
            return chatgpt_chain
        elif prompt_type == "scheduled":
            prompt_template = persona.scheduled_prompt
            prompt = PromptTemplate(
                input_variables=[],
                template=prompt_template
            )
            chatgpt_chain = LLMChain(
                llm=ChatOpenAI(model_name=LLM_NAME, temperature=temperature),
                prompt=prompt,
                verbose=verbose
            )
            return chatgpt_chain

        logging.error(f"create_chain could not handle prompt_type of {prompt_type}")
        return None

    @classmethod
    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(3))
    def get_product_context(cls, message, prev_context):
        """Retrieve product information based on the user's query.

        Args:
            message (str): the user's query
            prev_context (str): the previous product context

        Returns:
            str: the product context
            str: the product name
        """
        # Currently only using embedding retrieval no matter what
        descr, name, score = cls.retrieval.retrieve_with_embedding(message)
        logging.info(f"Score is {score}")
        if prev_context and score < PRODUCT_CONTEXT_SWITCH_SIMILARITY_THRESHOLD:
            logging.info("Using old context")
            return prev_context, name
        return descr, name
    
    @classmethod
    def get_product_list_text(cls, message):
        """Retrieve a list of available product names based on the user's query.

        Args:
            message (str): the user's query

        Returns:
            str: a formatted string containing a list of available product names
        """
        return '\n'.join(cls.retrieval.retrieve_top_product_names_with_embedding(message))

    @classmethod
    def get_chain_prevcontext(cls, chatid):
        """Retrieve the language model chain and previous product context for the given chatid.
        If the chain does not exist, it creates a new one.

        Args:
            chatid (string): the chat uuid

        Returns:
            tuple: a tuple containing the language model chain and the previous product context
        """
        if chatid not in cls.chatid_to_chain_prevcontext:
            chain = cls.create_chain()
            cls.chatid_to_chain_prevcontext[chatid] = (chain, '')
            
        return cls.chatid_to_chain_prevcontext[chatid]
