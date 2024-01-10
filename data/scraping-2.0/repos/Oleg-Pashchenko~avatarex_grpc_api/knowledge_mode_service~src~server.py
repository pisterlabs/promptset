import asyncio
import json
import time
from concurrent import futures

import grpc
import openai

from knowledge_mode_service.proto import knowledge_mode_pb2 as pb2
from knowledge_mode_service.proto import knowledge_mode_pb2_grpc as pb2_grpc

description = (
    "Ищет соотвтествующий вопрос если не нашел соотвтествия - возвращает пустоту"
)


async def get_function_by_rules(rules: list):
    properties = {}
    for rule in rules:
        properties[rule.question] = {
            "type": "boolean",
            "description": "Вопрос соответствует заданному?",
        }

    return [
        {
            "name": "get_question_by_context",
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": list(properties.keys()),
            },
        }
    ]


async def complete_openai(message, function, model, max_tokens, temperature, api_token):
    api_key = ""
    messages = [
        {"role": "system", "content": description},
        {"role": "assistant", "content": message},
    ]

    async with openai.AsyncOpenAI(api_key=api_key) as client:
        completion = await client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            functions=function,
            function_call={"name": "get_question_by_context"},
        )
        function_args = json.loads(
            completion.choices[0].message.function_call.arguments
        )
        if len(function_args.keys()) == 0:
            return None
        else:
            return list(function_args.keys())[0]


class OpenAIKnowledgeService(pb2_grpc.OpenAIKnowledgeServiceServicer):
    async def CompleteKnowledge(self, request, context):
        start_time = time.time()
        # Implement your server logic here
        # Access request.rules, request.openai_settings, etc.
        func = await get_function_by_rules(request.rules)
        print(request)
        print(func)
        result = await complete_openai(
            request.message,
            func,
            request.openai_settings.model,
            request.openai_settings.max_tokens,
            request.openai_settings.temperature,
            request.openai_settings.api_token,
        )
        execution_time = round(time.time() - start_time, 2)
        if not result:
            response_data = pb2.ResponseData(message=None, error="Nothing found!")
            return pb2.OpenAIKnowledgeResponse(
                success=False, data=response_data, execution_time=execution_time
            )

        for r in request.rules:
            if r.question == result:
                response_data = pb2.ResponseData(message=r.answer)
                return pb2.OpenAIKnowledgeResponse(
                    success=True, data=response_data, execution_time=execution_time
                )


async def run_server():
    server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=10))
    pb2_grpc.add_OpenAIKnowledgeServiceServicer_to_server(
        OpenAIKnowledgeService(), server
    )
    listen_addr = "localhost:50051"
    server.add_insecure_port(listen_addr)
    print(f"Starting server on {listen_addr}")
    await server.start()
    await server.wait_for_termination()


if __name__ == "__main__":
    asyncio.run(run_server())
