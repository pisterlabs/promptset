from typing import Any
from aimstack.langchain_debugger.callback_handlers import GenericCallbackHandler
from langchain.schema import LLMResult
from langchain.callbacks.utils import flatten_dict
from aimstack.langchain_debugger.types.action import (
    LLMEndAction,
)

class LocalGenericCallbackHandler(GenericCallbackHandler):
    # Currently the AimOS generic handler only supports OpenAI which returns results in
    # response.llm_output but the local
    # models seem to respod with response.generations instead, so overriding this method
    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """
        Handle the event when a Language Model (LLM) ends.

        Args:
            response (LLMResult): The result object from the LLM.
            kwargs (Any): Additional keyword arguments.
        """
        # Collect and serialize LLM outputs/generations
        generations_log = []
        for generations in response.generations:
            for generation in generations:
                generations_log.append(flatten_dict(generation.dict()))

        # This is the modified part
        if response.llm_output:
            llm_output = response.llm_output
            token_usage_res = llm_output.get('token_usage', {})
            token_usage = {
                'prompt_tokens': token_usage_res.get('prompt_tokens', 0),
                'completion_tokens': token_usage_res.get('completion_tokens',
                                                        0),
                'total_tokens': token_usage_res.get('total_tokens', 0),
            }
            model_name = llm_output.get('model_name', 'Unknown')

            # Tokens calculation
            self.step_total_tokens_count += token_usage['prompt_tokens'] + \
                                            token_usage['completion_tokens']

            # Cost calculation
            prompt_cost_val = 0
            output_cost_val = 0
            unknown_cost_val = False
            if model_name.startswith('gpt-3.5'):
                prompt_cost_val = 0.002
                output_cost_val = 0.002
            elif model_name.startswith('gpt-4-32K'):
                prompt_cost_val = 0.12
                output_cost_val = 0.06
            elif model_name.startswith('gpt-4'):
                prompt_cost_val = 0.03
                output_cost_val = 0.06
            else:
                unknown_cost_val = True

            if not unknown_cost_val:
                input_price = token_usage[
                                'prompt_tokens'] * prompt_cost_val / 1000
                output_price = token_usage[
                                'completion_tokens'] * output_cost_val / 1000
                total_price = input_price + output_price
                self.step_cost += total_price

            # Tokens usage tracking
            self.tokens_usage_input.track(token_usage['prompt_tokens'])
            self.tokens_usage_output.track(token_usage['completion_tokens'])
            self.tokens_usage.track(token_usage['total_tokens'])
        else:
            model_name = generations_log[0].get('generation_info_model', 'Unknown')
            token_usage = {}

        action = LLMEndAction(model_name, token_usage, generations_log)
        self.actions.append(action)
