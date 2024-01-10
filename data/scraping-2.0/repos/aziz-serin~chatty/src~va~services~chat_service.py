from flask import json, Response
from src.va.openai_tools.ai_chat import OpenAIChat
from src.va.context.context import Context
from src.va.openai_tools.error import InvalidMessageError, TokenLimitError, \
    NullResponseError, VAError, OpenAIAPIKeyError
from .service import Service
import logging

class ChatService(Service):
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger("chatty")

    def chat(self, content:dict) -> Response:
        try:
            prompt = content["prompt"]
            model = content["model"]
            token_limit = content["token_limit"]
        except KeyError as err:
            self.logger.debug(err)
            return Response(
                response=json.dumps({
                    "reason": "Invalid/Bad Request"
                }),
                status=400,
                mimetype='application/json'
            )
        openai_chat = OpenAIChat(
            model=model,
            config=self.system_config,
            token_limit=token_limit
        )
        try:
            system = content["system_config"]
            openai_chat.system_config = system
        except KeyError:
            pass
        try:
            response = openai_chat.send_message(prompt, False)
            data = {
                "response": response,
                "token_count": openai_chat.get_current_token_count(reply=response)
            }
            return Response(
                response=json.dumps(data),
                status=200,
                mimetype='application/json'
            )
        except (InvalidMessageError | TokenLimitError) as err:
            self.logger.debug(err)
            return Response(
                response=json.dumps({
                    "reason": "Invalid/Bad Request"
                }),
                status=400,
                mimetype='application/json'
            )
        except (OpenAIAPIKeyError | NullResponseError, VAError) as err:
            self.logger.debug(err)
            return Response(
                response=json.dumps({
                    "reason": "Internal Server Error"
                }),
                status=500,
                mimetype='application/json'
            )

    def conversation(self, content:dict) -> Response:
        connection = self.factory.get_context_connection()
        if connection is None:
            self.logger.debug("Could not establish connection with database")
            return Response(
                response=json.dumps({
                    "reason": "Internal Server Error"
                }),
                status=500,
                mimetype='application/json'
            )
        try:
            prompt = content["prompt"]
        except KeyError as err:
            self.logger.debug(err)
            return Response(
                response=json.dumps({
                    "reason": "Invalid/Bad Request"
                }),
                status=400,
                mimetype='application/json'
            )
        context_id = None
        try:
            context_id = content["context_id"]
        except KeyError:
            pass

        if context_id is not None:
            context_document = connection.get_document_by_id(context_id)
            if context_document is not None:
                context = Context()
                context.load_from_json(context_document)
            else:
                context = Context(
                    config={},
                    chat_model="gpt-3.5-turbo",
                    token_limit= 4000,
                    default=True
                )
        else:
            context = Context(
                config={},
                chat_model="gpt-3.5-turbo",
                token_limit=4000,
                default=True
            )

        openai_chat = OpenAIChat(
            model=context.chat_model,
            config=context.config,
            token_limit=context.token_limit,
            initial_messages=context.messages
        )
        try:
            system = content["system_config"]
            openai_chat.system_config = system
        except KeyError:
            pass
        try:
            response = openai_chat.send_message(prompt, True)
            context.messages = openai_chat.messages
            if context.default:
                # We are writing it, not default anymore
                context.default = False
                context_id = connection.insert_document(context.jsonify())
            else:
                connection.update_document(context_id, context.jsonify())
            data = {
                "response": response,
                "token_count": openai_chat.get_current_token_count(),
                "context_id": str(context_id)
            }
            return Response(
                response=json.dumps(data),
                status=200,
                mimetype='application/json'
            )
        except (InvalidMessageError | TokenLimitError) as err:
            self.logger.debug(err)
            return Response(
                response=json.dumps({
                    "reason": "Invalid/Bad Request"
                }),
                status=400,
                mimetype='application/json'
            )
        except (OpenAIAPIKeyError | NullResponseError, VAError) as err:
            self.logger.debug(err)
            return Response(
                response=json.dumps({
                    "reason": "Internal Server Error"
                }),
                status=500,
                mimetype='application/json'
            )

    def get_all_contexts(self) -> Response:
        context_connection = self.factory.get_context_connection()
        contexts = context_connection.get_all_documents()
        return Response(
            response=json.dumps(contexts),
            status=200,
            mimetype='application/json'
        )
