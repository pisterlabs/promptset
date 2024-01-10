# code_example = '''
# import time
# from openai import OpenAI
# from openai.types.beta.assistant import Assistant
# from openai.types.beta.thread import Thread
# from openai.types.beta.threads.run import Run
# from openai.types.beta.threads.thread_message import ThreadMessage, Content
# from openai.types.beta.threads.message_content_text import MessageContentText
# from openai.types.beta.assistant_create_params import Tool
# from openai._types import NotGiven, NOT_GIVEN
# from openai.pagination import SyncCursorPage

# from prompts.summarization_prompts import (
#     SUMMARIZER_DEFAULT_INSTRUCTIONS,
#     SUMMARIZER_DEFAULT_DESCRIPTION,
#     summary_prompt_list,
# )


# class Summarizer:
#     def __init__(
#         self,
#         assistant: Assistant | None = None,
#     ) -> None:
#         self.client = OpenAI()
#         self.prompt_list: list[str] = summary_prompt_list
#         self.default_prompt: str = self.prompt_list[0]
#         self.assistant: Assistant = assistant if assistant else self._create_assistant()
#         self.assistant_id: str = self.assistant.id

#     def _create_assistant(
#         self,
#         *,
#         name: str = "Python Code Summarizer",
#         model: str = "gpt-4-1106-preview",
#         instructions: str = SUMMARIZER_DEFAULT_INSTRUCTIONS,
#         description: str = SUMMARIZER_DEFAULT_DESCRIPTION,
#         tools: list[Tool] | NotGiven = NOT_GIVEN,
#     ) -> Assistant:
#         try:
#             return self.client.beta.assistants.create(
#                 name=name,
#                 model=model,
#                 instructions=instructions,
#                 description=description,
#                 tools=tools,
#             )
#         except Exception as e:
#             raise Exception(f"Error creating assistant (OpenAI): {e}")

#     def _delete_assistant(self) -> None:
#         try:
#             self.client.beta.assistants.delete(self.assistant_id)
#         except Exception as e:
#             print(f"Error deleting assistant (OpenAI): {e}")

#     def _create_thread(self) -> Thread:
#         return self.client.beta.threads.create()

#     def _delete_thread(self, thread_id: str) -> None:
#         try:
#             self.client.beta.threads.delete(thread_id)
#         except Exception as e:
#             print(f"Error deleting thread (OpenAI): {e}")

#     def _interpolate_prompt(self, code: str, custom_prompt: str | None = None) -> str:
#         """
#         Returns the prompt for the code snippet.

#         Args:
#             code (str): The code snippet.
#             custom_prompt (str | None): Custom prompt to be used. Defaults to None.

#         Returns:
#             str: The formatted prompt.

#         Notes:
#             - If custom_prompt is not provided, the default prompt will be used.
#             - If custom_prompt contains "{code}", it will be replaced with the code snippet.
#             - If custom_prompt does not contain "{code}", the code snippet will be appended below the custom_prompt.
#         """

#         if not custom_prompt:
#             return self.default_prompt.format(code=code)

#         else:
#             if "{code}" in custom_prompt:
#                 return custom_prompt.format(code=code)
#             else:
#                 return f"{custom_prompt}\n\n{code}"

#     def _add_message_to_thread(self, thread_id: str, message: str) -> None:
#         try:
#             self.client.beta.threads.messages.create(
#                 thread_id, content=message, role="user"
#             )
#         except Exception as e:
#             raise Exception(f"Error adding message to thread (OpenAI): {e}")

#     def _run_thread(self, thread_id: str) -> Run:
#         try:
#             return self.client.beta.threads.runs.create(
#                 thread_id, assistant_id=self.assistant_id
#             )
#         except Exception as e:
#             raise Exception(f"Error running thread (OpenAI): {e}")

#     def _get_response(self, thread_id: str, run_id: str) -> list[str]:
#         run: Run = self._run_thread(thread_id)

#         while True:
#             run_retrieval: Run = self.client.beta.threads.runs.retrieve(
#                 thread_id=thread_id, run_id=run.id
#             )
#             print(f"Run status: {run_retrieval.status}")
#             if run_retrieval.status == "completed":
#                 break
#             time.sleep(1)

#         messages: SyncCursorPage[
#             ThreadMessage
#         ] = self.client.beta.threads.messages.list(thread_id=thread_id)
#         return [
#             item.text.value
#             for content in messages
#             for item in content.content
#             if type(item) == MessageContentText and content.role == "assistant"
#         ]

#     def summarize_code(self, code: str, file_path: str) -> list[str] | str | None:
#         try:
#             thread: Thread = self._create_thread()
#             self._add_message_to_thread(thread.id, self._interpolate_prompt(code))
#             summary: list[str] = self._get_response(thread.id, code)
#             self._delete_thread(thread.id)
#             return summary
#         except Exception as e:
#             return f"An error occurred while summarizing '{file_path}' (OpenAI): {e}"

#     def print_assistants_list(self) -> None:
#         print(f"Assistants list: {self.client.beta.assistants.list()}")'''
