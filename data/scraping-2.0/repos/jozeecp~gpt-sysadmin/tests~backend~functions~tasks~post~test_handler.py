"""Test handler function"""

import unittest
from unittest.mock import MagicMock

import openai

from app.backend.functions.tasks.post.handler import handler
from app.backend.models.task import (
    EngineEnum,
    GPTMessage,
    HostMessage,
    StatusEnum,
    Task,
)


class TestHandler(unittest.TestCase):
    """Test handler function"""

    def test_handler(self):
        """Test handler function"""

        task = Task(
            taskId="some_task_id",
            engine=EngineEnum.GPT_3_5,
            status=StatusEnum.RUNNING,
            taskDescription="Sample task description",
            hostDescription="Sample host description",
            host="Sample host",
            user="Sample user",
            supervised=True,
        )

        # Mock TaskService
        task_service = MagicMock()
        task_service.create_task.return_value = task
        task_service.get_task.return_value = task

        # Mock MessageService
        message_service = MagicMock()

        # Mock GenerativeCmdService
        gpt_message = GPTMessage(
            role="assistant",
            human_msg="Sample command",
            machine_msg="Sample output",
        )
        generative_cmd_service = MagicMock()
        generative_cmd_service.generate_cmd.return_value = gpt_message

        # Mock CmdService
        cmd_service = MagicMock()
        cmd_service.execute_command.return_value = "Sample output"

        # Mock OpenAI
        chat_completion_response = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677652288,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": '{\
                            "human_msg": "Sample command",\
                            "machine_msg": "Sample output"\
                        }',
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 9, "completion_tokens": 12, "total_tokens": 21},
        }

        with unittest.mock.patch.object(
            openai.ChatCompletion, "create", return_value=chat_completion_response
        ), unittest.mock.patch(
            "app.backend.functions.tasks.post.handler.TaskService",
            return_value=task_service,
        ), unittest.mock.patch(
            "app.backend.functions.tasks.post.handler.MessageService",
            return_value=message_service,
        ), unittest.mock.patch(
            "app.backend.functions.tasks.post.handler.GenerativeCmdService",
            return_value=generative_cmd_service,
        ), unittest.mock.patch(
            "app.backend.functions.tasks.post.handler.CmdService",
            return_value=cmd_service,
        ):
            result = handler(task)

            # Verify task creation
            task_service.create_task.assert_called_once_with(task)

            # Verify command generation
            generative_cmd_service.generate_cmd.assert_called_once_with(task)

            # Verify adding gpt message
            message_service.add_message.assert_any_call(task, gpt_message)

            # Verify adding host message
            print("Mock calls:", message_service.add_message.mock_calls)
            message_service.add_message.assert_called_with(
                task, HostMessage(role="user", machine_msg="Sample output")
            )

            # Verify command execution
            cmd_service.execute_command.return_value = "Sample output"

            # Verify getting the latest task object
            task_service.get_task.assert_called_once_with(task.taskId)

            self.assertIsInstance(result, Task)


if __name__ == "__main__":
    unittest.main()
