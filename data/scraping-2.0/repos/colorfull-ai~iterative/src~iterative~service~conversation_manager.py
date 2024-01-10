from openai import OpenAI
from iterative import get_config as _get_config
from iterative import get_all_actions as _get_all_actions
from tqdm import tqdm
from logging import getLogger as _getLogger
import json
import time
from typing import Dict, List


logger = _getLogger(__name__)



class ConversationManager:
    def __init__(self, client, assistant_id):
        self.client: OpenAI = client
        self.assistant_id = assistant_id
        self.current_thread = _get_config().get("assistant_conversation_thread_id")
        self.current_run = None
        self.actions = _get_all_actions()

    def create_conversation(self):
        self.current_thread = self.client.beta.threads.create()
        return self.current_thread

    def add_message(self, message: str):
        if not self.current_thread:
            raise Exception("No active conversation thread.")
        return self.client.beta.threads.messages.create(
            thread_id=self.current_thread.id,
            content=message,
            role="user"
        )

    def process_conversation(self):
        logger.info("Processing conversation...")

        if not self.current_thread:
            raise Exception("No active conversation thread.")

        self.current_run = self.client.beta.threads.runs.create(
            thread_id=self.current_thread.id,
            assistant_id=self.assistant_id
        )

        # Continuously check the status of the conversation
        while True:
            self.current_run = self.client.beta.threads.runs.retrieve(
                thread_id=self.current_thread.id,
                run_id=self.current_run.id
            )

            if self.current_run.status == "requires_action":
                self.handle_required_action()
            elif self.current_run.status in ["cancelled", "cancelling", "completed", "failed", "expired"]:
                break

            time.sleep(1)  # Avoid too frequent polling

        messages = self.client.beta.threads.messages.list(
            thread_id=self.current_thread.id
        )
        return messages

    def handle_required_action(self):
        # Assuming the tool call details are in current_run.required_action
        required_action = self.current_run.required_action
        outputs = required_action.submit_tool_outputs
        actual_action_outputs = []
        for tool_call in outputs.tool_calls:
            tool_call_id = tool_call.id
            
            function_name, args = tool_call.function.name, tool_call.function.arguments
            # Execute the action
            try:
                action_output = self.execute_action(function_name, **json.loads(args))
                actual_action_outputs.append({"tool_call_id": tool_call_id, "output": json.dumps(action_output), "status": "success"})
            except Exception as e:
                actual_action_outputs.append({"tool_call_id": tool_call_id, "output": str(e), "status": "failure"})

        # Submit the tool output back to OpenAI
        self.submit_tool_outputs(actual_action_outputs)


    def execute_action(self, action_name: str, **kwargs):
        action = self.actions.get(action_name)
        if not action:
            raise Exception(f"Action {action_name} not found.")
        action_result = action.get_function()(**kwargs)
        return action_result

    def submit_tool_outputs(self, tool_outputs: List[Dict]):
        logger.info("Submitting tool outputs...")

        for _ in tqdm(range(300), desc="Submitting...", leave=False):
            time.sleep(0.03)

        run = self.client.beta.threads.runs.submit_tool_outputs(
            thread_id=self.current_thread.id,
            run_id=self.current_run.id,
            tool_outputs=tool_outputs
        )
        return run