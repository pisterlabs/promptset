import logging
from typing import Optional

from langchain import LLMChain, PromptTemplate
from langchain.llms import OpenAI
from langchain.llms.base import BaseLLM

from nemoguardrails.actions.llm.utils import (
    get_multiline_response,
    llm_call,
    strip_quotes,
)
from nemoguardrails.llm.params import llm_params
from nemoguardrails.llm.taskmanager import LLMTaskManager
from nemoguardrails.llm.types import Task
from nemoguardrails.logging.callbacks import logging_callback_manager_for_chain
from nemoguardrails.rails.llm.config import RailsConfig

log = logging.getLogger(__name__)

HALLUCINATION_NUM_EXTRA_RESPONSES = 2


async def check_hallucination_llama(
    llm_task_manager: LLMTaskManager,
    context: Optional[dict] = None,
    llm: Optional[BaseLLM] = None,
    use_llm_checking: bool = True,
):
    """Checks if the last bot response is a hallucination by checking multiple completions for self-consistency.

    :return: True if hallucination is detected, False otherwise.
    """

    bot_response = context.get("last_bot_message")
    last_bot_prompt_string = context.get("_last_bot_prompt")
    print(bot_response)

    if bot_response and last_bot_prompt_string:
        num_responses = HALLUCINATION_NUM_EXTRA_RESPONSES
        # Use beam search for the LLM call, to get several completions with only one call.
        # At the current moment, only OpenAI LLM engines are supported for computing the additional completions.
        # if type(llm) != OpenAI:
        #     log.warning(
        #         f"Hallucination rail can only be used with OpenAI LLM engines."
        #         f"Current LLM engine is {type(llm).__name__}."
        #     )
        #     print(f"Hallucination rail can only be used with OpenAI LLM engines."
        #         f"Current LLM engine is {type(llm).__name__}.")
        #     return False

        # Use the "generate" call from langchain to get all completions in the same response.
        last_bot_prompt = PromptTemplate(template="{text}", input_variables=["text"])
        chain = LLMChain(prompt=last_bot_prompt, llm=llm, llm_kwargs={"temperature": 1.0})
        
        extra_responses = []
        for i in range(num_responses):
            result = chain.run(last_bot_prompt)    
            result = get_multiline_response(result)
            result = strip_quotes(result)
            extra_responses.append(result)
        
        print(extra_responses)

        if len(extra_responses) == 0:
            # Log message and return that no hallucination was found
            # log.warning(
            #     f"No extra LLM responses were generated for '{bot_response}' hallucination check."
            # )
            print(f"No extra LLM responses were generated for '{bot_response}' hallucination check.")
            # return False
        elif len(extra_responses) < num_responses:
            # log.warning(
            #     f"Requested {num_responses} extra LLM responses for hallucination check, "
            #     f"received {len(extra_responses)}."
            # )
            print(f"Requested {num_responses} extra LLM responses for hallucination check, "
                f"received {len(extra_responses)}.")
            
        if use_llm_checking:
            # Only support LLM-based agreement check in current version
            prompt = llm_task_manager.render_task_prompt(
                task=Task.CHECK_HALLUCINATION,
                context={
                    "statement": bot_response,
                    "paragraph": " ".join(extra_responses),
                },
            )
            print(prompt)

            with llm_params(llm):#, temperature=0.0):
                agreement = await llm_call(llm, prompt)

            agreement = agreement.lower().strip()
            # log.info(f"Agreement result for looking for hallucination is {agreement}.")
            print(f"Agreement result for looking for hallucination is {agreement}.")
            # Return True if the hallucination check fails
            return "no" in agreement
        else:
            # TODO Implement BERT-Score based consistency method proposed by SelfCheckGPT paper
            # See details: https://arxiv.org/abs/2303.08896
            return False

    return False