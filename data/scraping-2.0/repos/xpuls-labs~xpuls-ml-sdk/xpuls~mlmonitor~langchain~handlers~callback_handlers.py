import time
import logging
from typing import Any, Dict, List, Union

import pydantic
from langchain.callbacks.base import AsyncCallbackHandler
from langchain.schema.output import LLMResult
from langchain.schema.messages import BaseMessage
from langchain.schema.agent import AgentAction, AgentFinish

from xpuls.mlmonitor.langchain.profiling.prometheus import LangchainChainMetrics, LangchainPrometheusMetrics, \
    LangchainChatModelMetrics, LLMTokens, LangchainToolMetrics
from xpuls.mlmonitor.utils.common import get_safe_dict_value

from . import constants as c
# Set the tracer provider and a console exporter


class CallbackHandler(AsyncCallbackHandler):
    log = logging.getLogger()

    def __init__(self, ln_metrics: LangchainPrometheusMetrics, chain_run_id: str,
                 override_labels: Dict[str, str]) -> None:
        self.llm_start_time = None
        self.llm_end_time = None

        self.chat_start_time = None
        self.chat_end_time = None
        self.chat_model_start_metrics = None

        self.tool_start_time = None
        self.tool_end_time = None
        self.tool_metrics = None

        self.ln_metrics = ln_metrics
        self.chain_run_id = chain_run_id

        self.chain_end_time = None
        self.chain_start_time = None
        self.chain_start_metrics = None

        self.override_labels = override_labels

    def _get_model_name(self, data, model_type):
        if model_type == c.WORD_CHAT_MODELS:
            model_info = get_safe_dict_value(get_safe_dict_value(
                get_safe_dict_value(data, 'kwargs', {}),
                'llm',
                {}
            ), 'kwargs', {})
            return model_info['model_name'] if 'model_name' in model_info else ''
        elif model_type == c.WORD_LLM:
            return 'default'  ## TODO: Improve
        else:
            return 'default'

    def on_llm_start(
            self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> Any:
        """Run when LLM starts running."""

        self.log.debug(f"on_llm_start, {serialized}, {prompts}, {kwargs}")

    def on_chat_model_start(
            self, serialized: Dict[str, Any], messages: List[List[BaseMessage]], **kwargs: Any
    ) -> Any:
        """Run when Chat Model starts running."""

        tags = get_safe_dict_value(kwargs, 'tags')
        parent_run_id = get_safe_dict_value(kwargs, 'parent_run_id')
        ml_model_name = get_safe_dict_value(get_safe_dict_value(serialized, 'kwargs', {}),
                                            'model_name', '')

        self.chat_model_start_metrics = LangchainChatModelMetrics(
            lc=str(get_safe_dict_value(serialized, 'lc')),
            type=get_safe_dict_value(serialized, 'type'),
            execution_step='on_chat_model_start',
            agent_type=tags[0] if len(tags) > 0 else c.WORD_UNDEFINED,
            ml_model_type='chat_models',
            ml_model_name=ml_model_name,
            other_tags=','.join(list(tags)),
        )
        self.chat_start_time = time.time()  # Record start time
        if self.llm_start_time is None:
            self.llm_start_time = time.time()  # Record start time
        self.ln_metrics.add_chat_model_counter(self.chat_model_start_metrics, self.override_labels)

        self.log.debug(f"on_chat_model_start, {serialized}, {messages}, {kwargs}")

    def on_llm_new_token(self, token: str, **kwargs: Any) -> Any:
        """Run on new LLM token. Only available when streaming is enabled."""
        self.log.debug(f"on_llm_new_token, {token}")

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        """Run when LLM ends running."""
        tags = get_safe_dict_value(kwargs, 'tags')
        token_usage = get_safe_dict_value(response.llm_output,
                                          'token_usage') if response.llm_output is not None else None
        execution_step = "on_llm_end"
        if token_usage is not None:
            llm_tokens = LLMTokens(
                execution_step=execution_step,
                ml_model_type='llm',
                ml_model_name=get_safe_dict_value(response.llm_output,
                                                  'model_name') if response.llm_output is not None else "fake_chat_model",
                other_tags=','.join(list(tags)),
            )
            self.ln_metrics.add_llm_tokens_usage(llm_tokens, 'prompt_tokens',
                                                 int(dict(token_usage)['prompt_tokens']), self.override_labels)
            self.ln_metrics.add_llm_tokens_usage(llm_tokens, 'total_tokens',
                                                 int(dict(token_usage)['total_tokens']), self.override_labels)
            self.ln_metrics.add_llm_tokens_usage(llm_tokens, 'completion_tokens',
                                                 int(dict(token_usage)['completion_tokens']), self.override_labels)
        if self.chat_model_start_metrics is not None:

            llm_end_metrics = LangchainChatModelMetrics(
                lc=self.chat_model_start_metrics.lc,
                type=self.chat_model_start_metrics.type,
                execution_step=execution_step,
                agent_type=tags[0] if len(tags) > 0 else c.WORD_UNDEFINED,
                ml_model_type=self.chat_model_start_metrics.ml_model_type,
                ml_model_name=self.chat_model_start_metrics.ml_model_name,
                other_tags=','.join(list(tags)),
            )
            elapsed_time = time.time() - self.chat_start_time  # Record start time
            self.ln_metrics.add_chat_model_counter(llm_end_metrics, self.override_labels)
            self.ln_metrics.observe_chat_model_latency(llm_end_metrics, elapsed_time, self.override_labels)

        elif self.llm_start_time is not None:
            llm_end_metrics = LangchainChatModelMetrics(
                lc='-1',
                type=c.WORD_UNDEFINED,
                execution_step='on_llm_end',
                agent_type=tags[0] if len(tags) > 0 else c.WORD_UNDEFINED,
                ml_model_type=self.chat_model_start_metrics.ml_model_type,
                ml_model_name=self.chat_model_start_metrics.ml_model_name,
                other_tags=','.join(list(tags)),
            )
            elapsed_time = time.time() - self.llm_start_time  # Record start time
            self.ln_metrics.add_chat_model_counter(llm_end_metrics, self.override_labels)
            self.ln_metrics.observe_chat_model_latency(llm_end_metrics, elapsed_time, self.override_labels)

        self.log.debug(f"on_llm_end, {response}, {kwargs}")

    def on_llm_error(
            self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> Any:
        """Run when LLM errors."""
        self.log.debug("on_llm_error")

    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any) -> Any:
        """Run when chain starts running."""
        tags = get_safe_dict_value(kwargs, 'tags')
        parent_run_id = get_safe_dict_value(kwargs, 'parent_run_id')
        model_type = 'chat_models' if 'chat_models' in get_safe_dict_value(get_safe_dict_value(
            get_safe_dict_value(serialized, 'kwargs', {}),
            'llm',
            {}
        ), 'id', []) else 'llm'

        self.chain_start_metrics = LangchainChainMetrics(
            lc=str(get_safe_dict_value(serialized, 'lc')),
            type=get_safe_dict_value(serialized, 'type'),
            agent_type=tags[0] if len(tags) > 0 else c.WORD_UNDEFINED,
            # run_id=str(get_safe_dict_value(kwargs, 'run_id')),
            # chain_run_id=self.chain_run_id,
            # parent_run_id=str(parent_run_id) if parent_run_id is not None else None,
            ml_model_type=model_type,
            other_tags=','.join(list(tags)),
            ml_model_name=self._get_model_name(serialized, model_type),
            execution_step='on_chain_start',

        )

        self.chain_start_time = time.time()  # Record start time
        self.ln_metrics.add_chain_counter(self.chain_start_metrics, self.override_labels)

        self.log.debug(f"on_chain_start, {serialized}, {inputs}, {kwargs}")

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> Any:
        """Run when chain ends running."""
        output_chars = get_safe_dict_value(outputs, 'text')
        tags = get_safe_dict_value(kwargs, 'tags')
        parent_run_id = get_safe_dict_value(kwargs, 'parent_run_id')

        if self.chain_start_metrics is not None:
            if pydantic.__version__.startswith("2"):
                chain_end_metrics = self.chain_start_metrics.model_copy(
                    update={'execution_step': 'on_chain_end'}, deep=True
                )
            else:
                chain_end_metrics = self.chain_start_metrics.copy(
                    update={'execution_step': 'on_chain_end'}, deep=True
                )
        else:
            chain_end_metrics = LangchainChainMetrics(
                lc='-1',
                type=c.WORD_UNDEFINED,
                agent_type=tags[0] if len(tags) > 0 else c.WORD_UNDEFINED,
                ml_model_type='llm',
                other_tags=','.join(list(tags)),
                ml_model_name='undefined',
                execution_step='on_chain_end',

            )
        elapsed_time = time.time() - self.chain_start_time

        self.ln_metrics.add_chain_counter(chain_end_metrics, self.override_labels)
        self.ln_metrics.observe_chain_latency(chain_end_metrics, elapsed_time, self.override_labels)
        self.log.debug(f"on_chain_end, {outputs}, {kwargs}, {self.chat_start_time}")

    def on_chain_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> Any:
        """Run when chain errors."""
        tags = get_safe_dict_value(kwargs, 'tags')
        parent_run_id = get_safe_dict_value(kwargs, 'parent_run_id')

        if self.chain_start_metrics is not None:
            if pydantic.__version__.startswith("2"):
                chain_err_metrics = self.chain_start_metrics.model_copy(
                    update={'execution_step': 'on_chain_error'}, deep=True
                )
            else:
                chain_err_metrics = self.chain_start_metrics.copy(
                    update={'execution_step': 'on_chain_error'}, deep=True
                )
        else:
            chain_err_metrics = LangchainChainMetrics(
                lc='-1',
                type=c.WORD_UNDEFINED,
                agent_type=tags[0] if len(tags) > 0 else c.WORD_UNDEFINED,
                # run_id=str(get_safe_dict_value(kwargs, 'run_id')),
                # parent_run_id=str(parent_run_id) if parent_run_id is not None else None,
                # chain_run_id=self.chain_run_id,
                ml_model_type='llm',
                other_tags=','.join(list(tags)),
                ml_model_name=c.WORD_UNDEFINED,
                execution_step='on_chain_error',

            )

        elapsed_time = time.time() - self.chain_start_time

        self.ln_metrics.add_chain_counter(chain_err_metrics, self.override_labels)
        self.ln_metrics.observe_chain_latency(chain_err_metrics, elapsed_time, self.override_labels)
        self.log.debug("on_chain_error")

    def on_tool_start(
            self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> Any:
        """Run when tool starts running."""
        self.tool_start_time = time.time()
        self.tool_metrics = LangchainToolMetrics(
            execution_step='on_tool_start',
            action=get_safe_dict_value(serialized, 'name'),
            # run_id=str(get_safe_dict_value(kwargs, 'run_id')),
            # parent_run_id=str(get_safe_dict_value(kwargs, 'parent_run_id')),
            agent_type=self.chain_start_metrics.agent_type,
            other_tags=','.join(get_safe_dict_value(kwargs, 'tags'))
        )

        self.ln_metrics.add_tools_usage_counter(self.tool_metrics, self.override_labels)
        self.log.debug(f"on_tool_start, {serialized}, {input_str}, {kwargs}")

    def on_tool_end(self, output: str, **kwargs: Any) -> Any:
        """Run when tool ends running."""

        tool_metrics = LangchainToolMetrics(
            execution_step='on_tool_end',
            action=self.tool_metrics.action,
            # run_id=str(get_safe_dict_value(kwargs, 'run_id')),
            # parent_run_id=str(get_safe_dict_value(kwargs, 'parent_run_id')),
            agent_type=self.chain_start_metrics.agent_type,
            other_tags=','.join(get_safe_dict_value(kwargs, 'tags'))
        )

        elapsed_time = time.time() - self.tool_start_time

        self.ln_metrics.add_tools_usage_counter(tool_metrics, self.override_labels)
        self.ln_metrics.observe_tool_latency(tool_metrics, elapsed_time, self.override_labels)

        self.log.debug(f"on_tool_end, {output}")

    def on_tool_error(
            self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> Any:
        """Run when tool errors."""

        tool_metrics = LangchainToolMetrics(
            execution_step='on_tool_error',
            action=self.tool_metrics.action,
            # run_id=str(get_safe_dict_value(kwargs, 'run_id')),
            # parent_run_id=str(get_safe_dict_value(kwargs, 'parent_run_id')),
            agent_type=self.chain_start_metrics.agent_type,
            other_tags=','.join(get_safe_dict_value(kwargs, 'tags'))
        )

        elapsed_time = time.time() - self.tool_start_time

        self.ln_metrics.add_tools_usage_counter(tool_metrics, self.override_labels)
        self.ln_metrics.observe_tool_latency(tool_metrics, elapsed_time, self.override_labels)

        self.log.debug("on_tool_error")

    def on_text(self, text: str, **kwargs: Any) -> Any:
        """Run on arbitrary text."""
        pass

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        """Run on agent action."""
        self.log.debug(f"on_agent_action, {action}")

    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> Any:
        """Run on agent end."""
        self.log.debug(f"on_agent_finish, {finish}")
