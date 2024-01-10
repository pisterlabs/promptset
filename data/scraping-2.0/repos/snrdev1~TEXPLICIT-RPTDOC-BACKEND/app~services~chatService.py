import os
from datetime import datetime

import openai
from bson import ObjectId
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

from app import socketio
from app.config import Config
from app.models.mongoClient import MongoClient
from app.utils.common import Common
from app.utils.enumerator import Enumerator
from app.utils.pipelines import PipelineStages
from app.utils.vectorstore.base import VectorStore

openai.api_key = Config.OPENAI_API_KEY


class ChatService:
    @staticmethod
    def get_chat_response(user_id, chat_type, question):
        """
        The function `get_chat_response` takes in a user ID, chat type, and question as input, and
        generates a response based on the chat type and question.

        Args:
          user_id: The user ID is a unique identifier for each user in the chat system. It helps to keep
        track of the user's chat history and personalize the responses if needed.
          chat_type: The `chat_type` parameter is used to determine the type of chat being conducted. It
        can have two possible values:
          question: The `question` parameter is the user's input or query that needs to be processed and
        responded to. It is the main input for the chatbot to generate a response.
        """
        try:
            if chat_type == int(Enumerator.ChatType.External.value):
                chat = ChatOpenAI(temperature=0)
                template = """ Answer the users question as best as possible.\
                        - If the answer has multiple paragraphs then format the answer as introduction, content, conclusion in html paragraph.\
                        - Don't preface answer with 'Introduction' or 'Content' or 'Conclusion'.\
                        - If the answer has a table then return an html table.\
                        - If the answer has a numbered list or bullets then return an html list.\
                        - If the answer has a piece of code logic then return the code logic as HTML code block.\
                        - Use HTML Code formatting when necessary.
                    """
                system_message_prompt = SystemMessagePromptTemplate.from_template(
                    template
                )
                human_template = "{question}"
                human_message_prompt = HumanMessagePromptTemplate.from_template(
                    human_template
                )
                chat_prompt = ChatPromptTemplate.from_messages(
                    [system_message_prompt, human_message_prompt]
                )

                # get a chat completion from the formatted messages
                response = chat(
                    chat_prompt.format_prompt(question=question).to_messages()
                ).content

            else:
                response = ChatService._answer_question(
                    user_id, question=question, max_tokens=300, chat_type=chat_type
                )

            # If the chat type is of type document chat then extract both the response and the sources
            if response and (
                chat_type
                in [
                    int(Enumerator.ChatType.Document.value),
                    int(Enumerator.ChatType.KnowledgeItem.value),
                ]
            ):
                data = response["response"]
                sources = response["sources"]
            else:
                data = response
                sources = []

            chat_dict = {
                "prompt": question,
                "response": data,
                "sources": sources,
                "timestamp": datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
                "chatType": chat_type,
            }

            if data:
                chat_event = "chat_" + str(user_id)
                socketio.emit(chat_event, [chat_dict])

            # Update user chat history
            ChatService()._update_user_chat_info(user_id, chat_dict)

        except Exception as e:
            Common.exception_details("ChatSerice.get_chat_response", e)

    def get_all_user_related_chat(self, user_id, limit=10, offset=0):
        """
        The function `get_all_user_related_chat` retrieves chat data related to a specific user from a
        MongoDB database.

        Args:
          user_id: The `user_id` parameter is the unique identifier of the user for whom you want to
        retrieve the related chat.
          limit: The `limit` parameter specifies the maximum number of chat records to retrieve. By
        default, it is set to 10, but you can change it to any positive integer value to retrieve a
        different number of records. Defaults to 10
          offset: The offset parameter is used to specify the starting point of the query results. It
        determines how many documents to skip before returning the results. By default, the offset is
        set to 0, meaning it starts from the beginning of the collection. Defaults to 0

        Returns:
          a dictionary containing the chat related to the specified user. If no chat is found, it
        returns None.
        """
        try:
            m_db = MongoClient.connect()
            project_keys = ["chat"]
            unset_keys = ["_id"]
            pipeline = [
                PipelineStages.stage_match(
                    {"user._id": ObjectId(user_id), "user.ref": "user"}
                ),
                PipelineStages.stage_project(project_keys),
                PipelineStages.stage_unset(unset_keys),
            ]

            result = m_db[Config.MONGO_CHAT_MASTER_COLLECTION].aggregate(pipeline)

            print("result : ", result)

            response = []
            if result:
                print("Found result!")
                response = Common.cursor_to_dict(result)

            if len(response) > 0:
                return response[0]
            else:
                return None

        except Exception as e:
            Common.exception_details("chatService.get_all_user_related_chat", e)
            return None

    def delete_chats(self, user_id):
        """
        The delete_chats function deletes the chat history of a user.
            Args:
                user_id (str): The id of the user whose chat history is to be deleted.

        Args:
            self: Represent the instance of the class
            user_id: Identify the user in the database

        Returns:
            The response of the update_one function
        """
        m_db = MongoClient.connect()

        delete_filter = {"user._id": ObjectId(user_id)}
        delete_field = {"$unset": {"chat": 1}}
        response = m_db[Config.MONGO_CHAT_MASTER_COLLECTION].update_one(
            delete_filter, delete_field
        )

        if response:
            return True
        else:
            return False

    @staticmethod
    def _customer_service_response(question, max_tokens=100):
        """
        The function `_customer_service_response` takes a question as input and uses the OpenAI GPT-3
        model to generate a response.

        Args:
          question: The `question` parameter is the input question or query that you want to ask the
        customer service. It will be used as the prompt for the OpenAI completion model.
          max_tokens: The `max_tokens` parameter specifies the maximum number of tokens that the model
        can generate in its response. Tokens are chunks of text that the model processes, and the total
        number of tokens affects the cost and response time of the API call. By setting a value for
        `max_tokens`, you can control. Defaults to 100

        Returns:
          The function `_customer_service_response` returns the response generated by the OpenAI language
        model.
        """
        try:
            question = question + " ->"
            response = openai.Completion.create(
                model="davinci:ft-personal-2023-08-07-13-08-23",
                prompt=question,
                presence_penalty=0,
                frequency_penalty=0.3,
                max_tokens=max_tokens,
                temperature=0.3,
                stop=["END", "->"],
            )

            return response["choices"][0]["text"]

        except Exception as e:
            Common.exception_details("chatService._customer_service_response", e)
            return ""

    @staticmethod
    def _answer_question(
        user_id,
        question="Am I allowed to publish model outputs to Twitter, without a human review?",
        max_tokens=150,
        chat_type=int(Enumerator.ChatType.Document.value),
    ):
        """
        The function `_answer_question` returns a chat response based on the user's question, using
        different chat types and a maximum number of tokens.

        Args:
          user_id: The user_id parameter is used to identify the user who is asking the question. It can
        be any unique identifier for the user, such as their username or user ID in your system.
          question: The "question" parameter is a string that represents the question or query that the
        user wants to ask. It is an optional parameter with a default value of "Am I allowed to publish
        model outputs to Twitter, without a human review?". Defaults to Am I allowed to publish model
        outputs to Twitter, without a human review?
          max_tokens: The `max_tokens` parameter specifies the maximum number of tokens allowed in the
        generated response. Tokens are units of text, such as words or characters. Defaults to 150
          chat_type: The `chat_type` parameter is used to specify the type of chat or conversation. It
        can take one of the following values:

        Returns:
          The function `_answer_question` returns a response based on the given question and chat type.
        The specific response depends on the logic implemented in the
        `VectorStore().get_document_chat_response`, `VectorStore().get_knowledgeitem_chat_response`, and
        `ChatService._customer_service_response` methods.
        """
        if chat_type == int(Enumerator.ChatType.Document.value):
            return VectorStore().get_document_chat_response(user_id, question)
        elif chat_type == int(Enumerator.ChatType.KnowledgeItem.value):
            return VectorStore().get_knowledgeitem_chat_response(question)
        else:
            return ChatService._customer_service_response(question, max_tokens)

    def _update_user_chat_info(self, user_id, chat_dict):
        """
        The function `_update_user_chat_info` updates the chat information for a user in a MongoDB
        database.

        Args:
          user_id: The user_id parameter is the unique identifier of the user for whom the chat
        information is being updated.
          chat_dict: The `chat_dict` parameter is a dictionary that contains the following keys and
        values:

        Returns:
          the number of documents modified in the database.
        """
        try:
            m_db = MongoClient.connect()

            query = {"user._id": ObjectId(user_id), "user.ref": "user"}

            update_data = {
                "$push": {
                    "chat": {
                        "$each": [
                            {
                                "role": "user",
                                "content": chat_dict["prompt"],
                                "timestamp": chat_dict["timestamp"],
                                "chatType": chat_dict["chatType"],
                            },
                            {
                                "role": "system",
                                "content": chat_dict["response"],
                                "sources": chat_dict["sources"],
                                "timestamp": chat_dict["timestamp"],
                                "chatType": chat_dict["chatType"],
                            },
                        ]
                    }
                }
            }

            response = m_db[Config.MONGO_CHAT_MASTER_COLLECTION].update_one(
                query, update_data, upsert=True
            )

            if response:
                return response.modified_count

            return None

        except Exception as e:
            Common.exception_details("ChatSerice._update_user_chat_info", e)
