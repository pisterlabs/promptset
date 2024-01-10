import json

from asgiref.sync import sync_to_async
from channels.db import database_sync_to_async
from channels.generic.websocket import AsyncJsonWebsocketConsumer
from langchain.memory import ConversationBufferMemory

from .chatbot.schemas import ChatResponse
from .chatbot.tools import AppointmentJSONException, create_appointment_from_json_str

from .chatbot.callback import (
    QuestionGenCallbackHandler,
    StreamingLLMCallbackHandler,
)
from .chatbot.chains import (
    get_appointment_chain,
    get_general_chat_chain,
    get_intents_chain,
    get_symptoms_chain,
)

from .chatbot.utils import init_retriever


# TODO(murat): Use a single chat history
# TODO(murat): Use a vectorstore for storing chat history
chat_history = []
memory = ConversationBufferMemory()


@database_sync_to_async
def get_profile(user):
    prof = user.healthprofile
    data = f"""User health data:
    Gender: {prof.gender};
    Age: {prof.age};
    Weight: {prof.weight} kilograms;
    Height: {prof.height} centimeters;
    Health condition notes: {prof.health_conditions_notes}"""
    return data


class ChatRoomConsumer(AsyncJsonWebsocketConsumer):
    async def connect(self):
        # TODO(murat): check if user is authenticated.
        # TODO(murat): create a chat session and use session id as chat_box_name.
        self.health_data = await get_profile(self.scope["user"])
        self.chat_box_name = self.scope["url_route"]["kwargs"]["chat_box_name"]
        self.group_name = "chat_%s" % self.chat_box_name

        await self.channel_layer.group_add(self.group_name, self.channel_name)

        await self.accept()
        resp = ChatResponse(username="bot", message="Loading stuff...", type="info")
        await self.send(text_data=json.dumps(resp.dict()))

        question_handler = QuestionGenCallbackHandler(self)
        stream_handler = StreamingLLMCallbackHandler(self)
        retriever = await sync_to_async(init_retriever)()
        self.intents_chain = await sync_to_async(get_intents_chain)()
        self.symptopms_qa_chain = await sync_to_async(get_symptoms_chain)(
            retriever, question_handler, stream_handler, tracing=True
        )
        self.general_chat_chain = await sync_to_async(get_general_chat_chain)(
            stream_handler, memory=memory
        )
        self.appointment_chain = await sync_to_async(get_appointment_chain)(memory)
        resp = ChatResponse(
            username="bot", message="Ready to accept questions", type="info"
        )
        await self.send(text_data=json.dumps(resp.dict()))

    async def disconnect(self, close_code):
        await self.channel_layer.group_discard(self.group_name, self.channel_name)

    async def receive(self, text_data):
        data = await self.decode_json(text_data)
        message = data.get("message", "")
        type_of_msg = data.get("type", "")

        intent = await self.intents_chain.arun(input=message)

        if intent == "appointment" or type_of_msg == "clarification":
            chat_hanlder = "appointment_message"
        elif intent == "symptom":
            chat_hanlder = "symptom_message"
        else:
            chat_hanlder = "general_chat_message"

        await self.channel_layer.group_send(
            self.group_name,
            {
                "type": chat_hanlder,
                "message": message,
                "username": "you",  # TODO: get username from session
            },
        )

    async def appointment_message(self, event):
        print("IN APPOINTMENT MESSAGE")
        message = event["message"]
        username = event["username"]
        # send message and username of sender to websocket
        resp = ChatResponse(username=username, message=message, type="stream")
        await self.send(text_data=json.dumps(resp.dict()))

        # Construct a response
        start_resp = ChatResponse(username="bot", message="", type="start")
        await self.send(text_data=json.dumps(start_resp.dict()))

        result = await self.appointment_chain.arun(input=message)
        print("@@@@@@@@@@= RESULT: ", result)
        try:
            output = await sync_to_async(create_appointment_from_json_str)(result)
            output_msg = f'Created an appointment with title "{output.name}" on {output.date} at {output.time}.'
            chat_history.append((message, output_msg))

            resp = ChatResponse(username="bot", message=output_msg, type="stream")
            await self.send(text_data=json.dumps(resp.dict()))

            end_resp = ChatResponse(username="bot", message="", type="end")
            await self.send(text_data=json.dumps(end_resp.dict()))

        except AppointmentJSONException as e:
            error_msg = str(e)
            is_field_error = any([x in error_msg for x in ["date", "time", "name"]])
            if is_field_error:
                resp = ChatResponse(
                    username="bot",
                    message=error_msg,
                    type="clarification",
                )
                # Make sure that the bot knows there was
                # a missing value and that we asked a user to provide it.
                serialized_result = ""
                appointment_dict = json.loads(result)
                for key, value in appointment_dict.items():
                    serialized_result += f"{key}: {value}\n"
                await sync_to_async(memory.save_context)(
                    {"input": serialized_result}, {"ouput": error_msg}
                )
                chat_history.append((message, error_msg))
                await self.send(text_data=json.dumps(resp.dict()))
            else:
                resp = ChatResponse(
                    username="bot",
                    message="Sorry, something went wrong. Please try again.",
                    type="stream",
                )
                await self.send(text_data=json.dumps(resp.dict()))

                end_resp = ChatResponse(username="bot", message="", type="end")
                await self.send(text_data=json.dumps(end_resp.dict()))

    async def general_chat_message(self, event):
        print("IN GENERAL CHAT MESSAGE")
        message = event["message"]
        username = event["username"]
        # send message and username of sender to websocket
        resp = ChatResponse(username=username, message=message, type="stream")
        await self.send(text_data=json.dumps(resp.dict()))

        # Construct a response
        start_resp = ChatResponse(username="bot", message="", type="start")
        await self.send(text_data=json.dumps(start_resp.dict()))

        result = await self.general_chat_chain.acall(
            {"text": message, "chat_history": chat_history}
        )
        chat_history.append((message, result["text"]))

        end_resp = ChatResponse(username="bot", message="", type="end")
        await self.send(text_data=json.dumps(end_resp.dict()))

    async def symptom_message(self, event):
        print("IN SYMPTOM MESSAGE")
        message = event["message"]
        username = event["username"]
        # send message and username of sender to websocket
        resp = ChatResponse(username=username, message=message, type="stream")
        await self.send(text_data=json.dumps(resp.dict()))

        # Construct a response
        start_resp = ChatResponse(username="bot", message="", type="start")
        await self.send(text_data=json.dumps(start_resp.dict()))

        question = (
            f"Original question: {message}.\nPatient health data: {self.health_data}"
        )
        result = await self.symptopms_qa_chain.acall(
            {"question": question, "chat_history": chat_history}
        )
        chat_history.append((question, result["answer"]))

        end_resp = ChatResponse(username="bot", message="", type="end")
        await self.send(text_data=json.dumps(end_resp.dict()))
