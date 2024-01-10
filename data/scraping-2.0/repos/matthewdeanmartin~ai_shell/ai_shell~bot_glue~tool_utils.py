"""
This module contains the loop_tools function, which is used to process
tools in a thread.
"""
# pylint: disable=wrong-import-position, using-constant-test
import logging
import time

if True:
    import openai_multi_tool_use_parallel_patch

    if not openai_multi_tool_use_parallel_patch:
        print("Needs to not move!")

import openai
from openai.types.beta import Thread
from openai.types.beta.threads import Run

from ai_shell.ai_logs.log_to_markdown import DialogLoggerWithMarkdown
from ai_shell.openai_toolkit import ToolKit

logger = logging.getLogger(__name__)


async def loop_tools(
    client: openai.AsyncClient, kit: ToolKit, run: Run, thread: Thread, dialog_logger_md: DialogLoggerWithMarkdown
) -> int:
    """
    Loop over the tools in a thread.

    Args:
        client (openai.AsyncClient): The OpenAI client.
        kit (ToolKit): The toolkit.
        run (Run): The run object.
        thread (Thread): The thread object.
        dialog_logger_md (DialogLoggerWithMarkdown): The dialog logger.

    Returns:
        int: The number of tools used.
    """
    waiting = False
    tool_use_count = 0
    while True:
        if run.status in ("queued", "in_progress"):
            print(".", end="")
            waiting = True
            time.sleep(1)
        elif run.status in ("failed", "cancelling", "cancelled", "expired"):
            raise Exception(run.last_error)
        elif run.status == "completed":
            logger.info("completed")
            if waiting:
                print()
            break
        elif run.status == "requires_action":
            logger.info("requires_action")
            if waiting:
                print()
                waiting = False
            if not run.required_action:
                # Why would this happen? Mypy says it is possible.
                raise TypeError("Missing required_action")
            for tool_call in run.required_action.submit_tool_outputs.tool_calls:
                tool_use_count += 1
                dialog_logger_md.add_tool(tool_call.function.name, tool_call.function.arguments)
            results = await kit.process_tool_calls(run, print)
            dialog_logger_md.add_tool_result(results)
            # submit results
            run = await client.beta.threads.runs.submit_tool_outputs(
                thread_id=thread.id,
                run_id=run.id,
                tool_outputs=results,
            )
            run = await client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
            continue
        else:
            raise Exception(run.status)
        # poll
        run = await client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
    return tool_use_count
