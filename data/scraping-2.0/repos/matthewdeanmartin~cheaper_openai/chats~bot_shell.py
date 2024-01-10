"""
Code for AI
"""
import json
import time
from typing import Any, List, Optional

import untruncate_json
from openai import AsyncOpenAI
from openai.types.beta import Assistant, Thread, ThreadDeleted
from openai.types.beta.threads import Run, ThreadMessage, run_create_params

from chats.tool_code.pypi_info import PyPIChecker
from chats.tool_code.text_shorteners import count_tokens, readability_scores, word_count
from chats.utils import SetEncoder, write_json_to_logs


class Bot:
    def __init__(self, assistant_id: str = None, model: str = "gpt-3.5-turbo"):
        self.client = AsyncOpenAI()
        self.model = model

        self.assistant_id = assistant_id
        self.assistant: Optional[Assistant] = None

    async def populate_assistant(self) -> Assistant:
        """Refresh the assistant object from remote store"""
        if self.assistant_id and not self.assistant:
            self.assistant = await self.retrieve_assistant()
        write_json_to_logs(self.assistant, "assistant")
        return self.assistant

    async def retrieve_assistant(self) -> Assistant:
        self.assistant = await self.client.beta.assistants.retrieve(assistant_id=self.assistant_id)
        write_json_to_logs(self.assistant, "assistant")
        return self.assistant

    async def create_assistant(self, bot_name: str, instructions: str) -> Assistant:
        self.assistant = await self.client.beta.assistants.create(
            name=bot_name,
            instructions=instructions,
            # model="gpt-4-1106-preview", EXPENSIVE
            model=self.model,  # Cheap and smart enough
        )
        write_json_to_logs(self.assistant, "assistant")
        return self.assistant

    async def update_instructions(self, instuctions: str):
        assistant = await self.client.beta.assistants.update(
            self.assistant.id,
            instructions=instuctions,
        )
        self.assistant = assistant
        return assistant

    async def enable_code_intern(self):
        assistant = await self.client.beta.assistants.update(
            self.assistant.id,
            tools=[{"type": "code_interpreter"}],
        )
        self.assistant = assistant
        return assistant

    async def enable_file(self, file_ids: list[str]):
        """Upload one file"""
        # Update Assistant
        assistant = await self.client.beta.assistants.update(
            self.assistant.id,
            tools=[{"type": "retrieval"}],
            file_ids=file_ids,
        )
        write_json_to_logs(self.assistant, "update_assistant")
        self.assistant = assistant
        return assistant


class BotConversation:
    def __init__(self, assistant: Assistant, thread: Optional[Thread] = None):
        self.client = AsyncOpenAI()
        self.model = "gpt-3.5-turbo"
        self.assistant: Assistant = assistant

        self.thread: Optional[Thread] = thread
        if self.thread:
            self.thread_id = thread.id
        else:
            self.thread_id = None

    async def populate_thread(self) -> Thread:
        """Refresh the thread object from remote store"""
        self.thread = await self.client.beta.threads.retrieve(thread_id=self.thread_id)
        write_json_to_logs(self.assistant, "thread")
        return self.thread

    async def create_thread(self) -> Thread:
        """Create thread. It will persist with messages for 30 days."""
        self.thread = await self.client.beta.threads.create()
        self.thread_id = self.thread.id
        write_json_to_logs(self.thread, "thread")
        return self.thread

    async def delete_thread(self) -> ThreadDeleted:
        """Clean up thread because we can't list them later!"""
        result = await self.client.beta.threads.delete(self.thread.id)
        write_json_to_logs(result, "delete_thread")
        self.thread = None
        self.thread_id = None
        return result

    async def create_run(self, tools: Optional[List[run_create_params.Tool]] = None) -> Run:
        """This is a request to chatbot where the chatbot might make some call backs before
        responding with the final new message"""
        run = await self.client.beta.threads.runs.create(
            thread_id=self.thread_id,
            assistant_id=self.assistant.id,
            tools=tools,
        )
        write_json_to_logs(run, "run")
        return run

    async def poll_the_run(self, run: Run, tool: Optional[str] = None) -> Run:
        """Handle polling.

        TODO: Need pointers to the tool handlers!
        """
        if tool:
            tool_tag = f"_{tool}"
        else:
            tool_tag = ""

        while True:
            if run.status in ("queued", "in_progress"):
                # Don't log this. Too noisy.
                print(".", end="")
                time.sleep(1)
            elif run.status == "failed":
                write_json_to_logs(run, f"run_poll_{run.status}{tool_tag}")
                raise Exception(run.last_error)
            elif run.status == "completed":
                # Don't log this. Too noisy, it only has timing & repeats other run info
                # write_json_to_logs(run, f"run_poll_{run.status}{tool_tag}")
                break
            elif run.status == "cancelling":
                print("Cancelling, this isn't going well.")
                write_json_to_logs(run, f"run_poll_{run.status}{tool_tag}")
                time.sleep(1)
            elif run.status == "cancelled":
                print("Cancelled, did you do that?")
                write_json_to_logs(run, f"run_poll_{run.status}{tool_tag}")
                exit()
            elif run.status == "expired":
                print("Some sort of time out")
                write_json_to_logs(run, f"run_poll_{run.status}{tool_tag}")
                exit()
            elif run.status == "requires_action":
                write_json_to_logs(run, f"run_poll_{run.status}{tool_tag}")
                await self.process_tool_calls(run)
            else:
                raise Exception(f"Out of bounds, don't know what to do with run.status {run.status}")
            # poll
            run = await self.check_run(run)

        return run

    async def check_run(self, run: Run) -> Run:
        """This is a request to chatbot where the chatbot might make some call backs before
        responding with the final new message"""
        run = await self.client.beta.threads.runs.retrieve(thread_id=self.thread_id, run_id=run.id)
        return run

    async def add_user_message(self, content: str) -> ThreadMessage:
        """Add a persistent message. Messages persist 30 days, same as thread"""
        message = await self.client.beta.threads.messages.create(
            thread_id=self.thread_id,
            role="user",
            content=content,
        )
        return message

    async def display_current_threads_messages(self):
        messages = await self.client.beta.threads.messages.list(thread_id=self.thread_id, order="asc")
        write_json_to_logs(messages, "messages")
        return messages

    async def display_most_recent_bot_message(self) -> Optional[Any]:
        messages = await self.client.beta.threads.messages.list(thread_id=self.thread_id, order="desc")
        # all_messages = []
        async for message in messages:
            write_json_to_logs(message, "messages")
            return message
            # all_messages.append(message)
        return None
        # return all_messages

    async def process_tool_calls(self, run: Run) -> Run:
        results = []
        for tool_call in run.required_action.submit_tool_outputs.tool_calls:
            name = tool_call.function.name
            # TODO: Sometimes this isn't really json!
            args_text = tool_call.function.arguments
            try:
                arguments = json.loads(args_text)
            except json.decoder.JSONDecodeError:
                print("Truncated json!")
                arguments = json.loads(untruncate_json.complete(args_text))

            print(arguments)
            print(name)
            if name == "package_exists":
                result = await PyPIChecker().package_exists(arguments["package_name"])
            elif name == "packages_or_variant_exist":
                result = await PyPIChecker().packages_or_variant_exist(arguments["package_names"])
            elif name == "package_is_stdlib":
                result = PyPIChecker().package_is_stdlib(arguments["package_names"])
            elif name == "packages_exists":
                result = await PyPIChecker().packages_exist(arguments["package_names"])
            elif name == "describe_packages":
                result = await PyPIChecker().describe_packages(arguments["package_names"])
            elif name == "count_tokens":
                result = count_tokens(arguments["text"])
            elif name == "readability_scores":
                result = readability_scores(arguments["text"])
            elif name == "word_count":
                result = word_count(arguments["text"])
            else:
                raise Exception(f"Unknown function name {name}")
            print(result)
            print("-----")
            tool_result = {
                "tool_call_id": tool_call.id,
                "output": json.dumps(result, cls=SetEncoder),
            }
            write_json_to_logs(tool_call, f"tool_call_{name}")
            write_json_to_logs(result, f"tool_result_{name}")

            results.append(tool_result)

        # submit all tool calls as batch to runs, not the *run*
        post_tool_run = await self.client.beta.threads.runs.submit_tool_outputs(
            thread_id=self.thread.id,
            run_id=run.id,
            tool_outputs=results,
        )
        post_poll_run = await self.poll_the_run(post_tool_run)
        return post_poll_run

    async def submit_tool_results(self, thread_id: str, run: Run, tool_responses: dict[str, Any]):
        run = await self.client.beta.threads.runs.submit_tool_outputs(
            thread_id=thread_id,
            run_id=run.id,
            tool_outputs=[
                {
                    "tool_call_id": tool_responses["tool_call_id"],
                    "output": json.dumps(tool_responses["response"], cls=SetEncoder),
                }
            ],
        )
        return run
