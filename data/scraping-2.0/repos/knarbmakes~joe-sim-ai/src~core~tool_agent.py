import json
import openai
import os
import traceback
import logging

from dotenv import load_dotenv
from core.function_call_handler import FunctionCallHandler
from core.message_stack_builder import MessageStackBuilder
from core.cost_helper import (
    calculate_cost,
)
from typing import Any
from datetime import datetime
from core.joe_types import TextConfig, ObjectConfig

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

load_dotenv()
openai_client = openai.OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
)
class ToolAgent:
    def __init__(self, text_config: TextConfig, object_config: ObjectConfig):
        # Every agent needs a key, so generate one if it is not provided.
        self.agent_key = text_config.agent_key or os.urandom(16).hex()

        # text_config contains llm or user defined parameters
        self.text_config = text_config

        # object_config contains system level configuration objects and variables.
        # things that only our code needs to know about.
        self.object_config = object_config
        self.agent_service = object_config.agent_service
        self.bank_account = object_config.bank_account

        self.message_stack_builder = MessageStackBuilder(
            self.agent_key, text_config, object_config
        )

        self.function_call_handler = FunctionCallHandler(
            self.agent_key, text_config, object_config
        )

    def track_usage(self, usage):
        # Calculate cost from usage and subtract from bank account
        cost = calculate_cost(dict(usage), self.text_config.model)
        if not self.bank_account.subtract_balance(cost, self.agent_key):
            logger.warning(
                f"Insufficient funds for agent {self.agent_key}. Stopping execution."
            )
            return False
        return True

    def run(self, input_messages=None, sys_message_suffix=None) -> dict[str, Any]:
        try:
            if not self.bank_account.get_balance() > 0:
                return {"status": "error", "error": "Budget limit exceeded"}

            # Update context memory with input messages if any, before starting the loop
            if input_messages:
                self.agent_service.update_context_memory(memory_elements=input_messages)

            # Keep track of tool executions
            tool_execution_log = []

            # Loop until we stop getting function calls or we exceed the budget.
            while True:
                if not self.bank_account.get_balance() > 0:
                    logger.warning(
                        f"Budget exceeded for agent {self.agent_key}. Stopping execution."
                    )
                    return {"status": "error", "error": "Budget limit exceeded"}

                messages = self.message_stack_builder.build_message_stack(sys_message_suffix)
                kwargs = self.text_config.kwargs or {}

                # Check if the last message is a tool_call and skip completion if so
                if messages and "tool_calls" in messages[-1] and messages[-1]["tool_calls"]:
                    # Directly use the tool_calls as they are already in dictionary format
                    tool_calls_serializable = messages[-1]["tool_calls"]
                    current_log = self.function_call_handler.handle_fn_calls(tool_calls_serializable)
                    tool_execution_log.extend(current_log)
                    continue

                # Add functions to the kwargs if they are available
                available_function_definitions = (
                    self.function_call_handler.get_available_fn_defn()
                )
                if available_function_definitions:
                    kwargs["tools"] = available_function_definitions

                # Log the completion request to a file in tmp/logs
                logfile = f"tmp/logs/completion_requests{self.agent_key}_{datetime.now().isoformat()}.json"
                # ensure dir exists
                os.makedirs(os.path.dirname(logfile), exist_ok=True)
                with open(logfile, "a") as f:
                    f.write(
                        json.dumps(
                            {
                                "model": self.text_config.model,
                                "messages": messages,
                                "kwargs": kwargs,
                            },
                            indent=2,
                        )
                        + "\n"
                    )

                # TODO: If the last message on the stack, we should skip the completion and call the function(s) directly.
                completion = openai_client.chat.completions.create(
                    model=self.text_config.model,
                    messages=messages,
                    timeout=90,
                    **kwargs,
                )

                # Track spending
                if not self.track_usage(dict(completion).get("usage")):
                    return {"status": "error", "error": "Insufficient funds"}

                # Handle explicit function calls
                output = dict(completion.choices[0].message)
                if "tool_calls" in output and output["tool_calls"] is not None:
                    # Convert tool_calls to serializable format
                    tool_calls_serializable = [
                        tool_call.dict() for tool_call in output["tool_calls"]
                    ]

                    # Update context memory with serializable tool calls
                    self.agent_service.update_context_memory(
                        [
                            {
                                "role": "assistant",
                                "content": output.get("content"),
                                "tool_calls": tool_calls_serializable,
                            }
                        ],
                    )

                    # Log the content, if any.
                    if output.get("content"):
                        logger.info(f"Intermediate Response:\n{output.get('content')}")

                    # Process the tool calls
                    current_log = self.function_call_handler.handle_fn_calls(tool_calls_serializable)
                    tool_execution_log.extend(current_log)
                else:
                    # Save the output to context memory and return.
                    self.agent_service.update_context_memory([{"role": "assistant", "content": output.get("content")}],
                    )
                    return {"status": "success", "output": output.get("content"), "tool_execution_log": tool_execution_log}
        except Exception as e:
            traceback.print_exc()  # Print the stack trace
            logger.exception(f"An error occurred in agent {self.agent_key}: {e}")
            return {"status": "error", "error": f"{e}"}
