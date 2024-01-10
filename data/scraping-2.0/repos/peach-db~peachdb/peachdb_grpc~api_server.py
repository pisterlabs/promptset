"""
Module docstring:
This module is a streaming gRPC server for the bot service.
It serves to handle requests related to bot creation, conversation creation and continuation.
"""

import asyncio
import functools
import traceback
from typing import AsyncIterable, Iterator

import api_pb2 as api_pb2  # type: ignore
import api_pb2_grpc as api_pb2_grpc  # type: ignore
import grpc  # type: ignore
import openai

from peachdb.bots.qa import BadBotInputError, ConversationNotFoundError, QABot, UnexpectedGPTRoleResponse


def grpc_error_handler_async_fn(fn):
    """
    A decorator to handle any unhandled errors and map them to appropriate gRPC status codes.
    :param fn: The function to be decorated.
    :return: Decorated function.
    """

    @functools.wraps(fn)
    async def wrapper(self, request, context):
        try:
            return await fn(self, request, context)
        except BadBotInputError as e:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(e))
        except openai.error.RateLimitError:
            await context.abort(
                grpc.StatusCode.RESOURCE_EXHAUSTED, "OpenAI's servers are currently overloaded. Please try again later."
            )
        except openai.error.AuthenticationError:
            await context.abort(
                grpc.StatusCode.UNAUTHENTICATED, "There's been an authentication error. Please contact the team."
            )
        except openai.error.ServiceUnavailableError:
            await context.abort(
                grpc.StatusCode.RESOURCE_EXHAUSTED, "OpenAI's servers are currently overloaded. Please try again later."
            )
        except UnexpectedGPTRoleResponse:
            await context.abort(grpc.StatusCode.INTERNAL, "GPT-3 responded with a role that was not expected.")
        except ConversationNotFoundError:
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT, "Conversation not found. Please check `conversation_id`"
            )
        except Exception as e:
            traceback.print_exc()
            await context.abort(grpc.StatusCode.UNKNOWN, "An unknown error occurred. Please contact the team.")

    return wrapper


def grpc_error_handler_async_gen(fn):
    """
    A decorator to handle any unhandled errors and map them to appropriate gRPC status codes.
    :param fn: The function to be decorated.
    :return: Decorated function.
    """

    @functools.wraps(fn)
    async def wrapper(self, request, context):
        try:
            async for response in fn(self, request, context):
                yield response
        except BadBotInputError as e:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(e))
        except openai.error.RateLimitError:
            await context.abort(
                grpc.StatusCode.RESOURCE_EXHAUSTED, "OpenAI's servers are currently overloaded. Please try again later."
            )
        except openai.error.ServiceUnavailableError:
            await context.abort(
                grpc.StatusCode.RESOURCE_EXHAUSTED, "OpenAI's servers are currently overloaded. Please try again later."
            )
        except openai.error.AuthenticationError:
            await context.abort(
                grpc.StatusCode.UNAUTHENTICATED, "There's been an authentication error. Please contact the team."
            )
        except UnexpectedGPTRoleResponse:
            await context.abort(grpc.StatusCode.INTERNAL, "GPT-3 responded with a role that was not expected.")
        except ConversationNotFoundError:
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT, "Conversation not found. Please check `conversation_id`"
            )
        except Exception as e:
            traceback.print_exc()
            await context.abort(grpc.StatusCode.UNKNOWN, "An unknown error occurred. Please contact the team.")

    return wrapper


class BotServiceServicer(api_pb2_grpc.BotServiceServicer):
    @grpc_error_handler_async_fn
    async def CreateBot(self, request: api_pb2.CreateBotRequest, context) -> api_pb2.CreateBotResponse:
        """
        RPC method to create a new bot instance.
        :param request: Request instance for CreateBot.
        :param context: Context instance for CreateBot.
        :return: CreateBotResponse instance.
        """
        bot = QABot(
            bot_id=request.bot_id,
            system_prompt=request.system_prompt,
            llm_model_name=request.llm_model_name if request.HasField("llm_model_name") else "gpt-3.5-turbo",
            embedding_model=request.embedding_model_name if request.HasField("embedding_model_name") else "openai_ada",
        )
        bot.add_data(documents=list(request.documents))
        return api_pb2.CreateBotResponse(status="Bot created successfully.")

    @grpc_error_handler_async_gen
    async def CreateConversation(
        self, request: api_pb2.CreateConversationRequest, context
    ) -> AsyncIterable[api_pb2.CreateConversationResponse]:
        """
        RPC method to create a new conversation for a bot.
        :param request: Request instance for CreateConversation.
        :param context: Context instance for CreateConversation.
        :return: CreateConversationResponse instance stream.
        """
        await self._check(request, "bot_id", context)
        await self._check(request, "query", context)

        bot_id = request.bot_id
        query = request.query

        bot = QABot(bot_id=bot_id)
        generator = bot.create_conversation_with_query(query=query, stream=True)
        assert isinstance(generator, Iterator)

        for cid, response in generator:
            yield api_pb2.CreateConversationResponse(conversation_id=cid, response=response)

    @grpc_error_handler_async_gen
    async def ContinueConversation(
        self, request: api_pb2.ContinueConversationRequest, context
    ) -> AsyncIterable[api_pb2.ContinueConversationResponse]:
        """
        RPC method to continue a conversation for a bot.
        :param request: Request instance for ContinueConversation.
        :param context: Context instance for ContinueConversation.
        :return: ContinueConversationResponse instance stream.
        """
        await self._check(request, "bot_id", context)
        await self._check(request, "conversation_id", context)
        await self._check(request, "query", context)

        bot_id = request.bot_id
        conversation_id = request.conversation_id
        query = request.query

        bot = QABot(bot_id=bot_id)
        response_gen = bot.continue_conversation_with_query(conversation_id=conversation_id, query=query, stream=True)
        for response in response_gen:
            yield api_pb2.ContinueConversationResponse(response=response)

    async def _check(self, request, field, context):
        if not getattr(request, field):
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, f"{field} must be specified.")


async def serve() -> None:
    """
    Start a gRPC server.
    :return: None
    """
    server = grpc.aio.server()
    api_pb2_grpc.add_BotServiceServicer_to_server(BotServiceServicer(), server)
    server.add_insecure_port("[::]:50051")
    await server.start()
    await server.wait_for_termination()


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(serve())
