import json
from logging import Logger
from typing import List, Dict, Optional, Deque

from langchain import LLMChain, PromptTemplate
from langchain.base_language import BaseLanguageModel

from agent import serializer_surrounding_deque, serializer_inventory_deque


class RequirementChecker:
    llm: BaseLanguageModel
    chain: LLMChain
    logger: Optional[Logger]

    prompt1: str = """I will going to tell you what actions you can take in the game (also its requirements too). 
    You'll check to see if the situation met the requirement of the action asks for, based on your priorities, 
    and if you are, give us the name of the action.
    Imagine that you are an expert player who thinks strategically. Choose ONLY from the list of all actions.
    Do not use your background knowledge and select only based on the information given next. 

    You must follow: If you can't determine the correct answer with the information you have, select "Noop".

    Anything not shown in the surrounding is flat ground.
    {context}
    
    Available Actions: 
    {actions}

    Give me only the name of the action:"""

    # select the highest priority action
    prompt2: str = """Select the first action in available actions.

    {context}

    Available Actions: 
    {actions}

    Give me only the name of the action:"""

    def __init__(self, llm: BaseLanguageModel, logger: Logger):
        self.llm = llm
        self.logger = logger

        assert llm is not None, "Must have a language model"

        _prompt = PromptTemplate(
            template=self.prompt2,
            input_variables=["context", "actions"]
        )

        self.chain = LLMChain(
            llm=self.llm,
            prompt=_prompt,
        )

    def run(
            self,
            _available_actions: str,
            _surrounding_deque: Optional[Deque[Dict[str, list]]] = None,
            _inventory_deque: Optional[Deque[Dict[str, int]]] = None,
            **kwargs: Optional[str],
    ) -> str:
        assert len(_available_actions) > 0, "Must have at least one action"

        status_explain = []
        if _surrounding_deque:
            status_explain.append(serializer_surrounding_deque(_surrounding_deque))
        if _inventory_deque:
            status_explain.append(serializer_inventory_deque(_inventory_deque))

        _context = "\n".join(status_explain)
        context_logger = _context.replace('\n', ' ')
        self.logger.info(f"Context: {context_logger}")
        # response = self.chain.predict(context=_context, actions=_actions)
        action = self.chain.predict(context="", actions=_available_actions)
        action = action.lower().replace(" ", "_")
        action_logger = action.replace('\n', ' ')
        self.logger.info(f"Response: {action_logger}")

        return action
