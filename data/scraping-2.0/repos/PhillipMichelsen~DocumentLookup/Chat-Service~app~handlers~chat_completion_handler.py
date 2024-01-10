import json

from app.schemas.service_tasks.chat_completion_schemas import ChatCompletionRequest, ChatCompletionResponse
from app.schemas.task_schemas import TaskRequest
from app.utils.openai_utils import openai_utils
from app.utils.service_utils import send_handler_messages


def handle_chat_completion(decoded_message_body):
    task_request = TaskRequest.model_validate(decoded_message_body)
    job_data = json.loads(task_request.job_data)
    chat_completion_request = ChatCompletionRequest.model_validate(job_data)

    chat_completion_message = openai_utils.form_messages(chat_completion_request.query, chat_completion_request.ranked_entries)
    chat_completion = openai_utils.chat_completion(chat_completion_message)

    chat_completion_response = ChatCompletionResponse(chat_completion=chat_completion)

    send_handler_messages(task_request.task_id, job_data, chat_completion_response)
