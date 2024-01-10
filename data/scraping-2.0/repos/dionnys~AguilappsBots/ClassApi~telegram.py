import os
import re
import time
import random
import string
import logging
import asyncio
import validators
from typing import Optional
from datetime import datetime
from ClassApi.openai import OpenAI
from aiogram.utils import executor
from aiogram import Bot, Dispatcher, types
from ClassApi.spacy import SpacyProcessor
from Tools.managers_tasks import TaskHandler
from aiogram.contrib.middlewares.logging import LoggingMiddleware
from ConnectionDao.mongodb_connection import MongoDBConnection


class TelegramBot:
    def __init__(self, token, openai_api_key, spacy_model_default):
        self.token = token
        self.bot = Bot(token=self.token)
        self.dp = Dispatcher(self.bot)
        self.dp.middleware.setup(LoggingMiddleware())
        self.chatgpt_active_users = {}
        self.user_conversations = {}
        logging.basicConfig(level=logging.INFO)
        self.dp.register_message_handler(self.start, commands=['start'])
        self.dp.register_message_handler(self.handle_echo)
        self.spacy_model_name_default = spacy_model_default
        # Create an instance of the MongoDBConnection class
        self.db_connection = MongoDBConnection()
        # Crear instancia de la clase OpenAI y configurar la API key
        self.openai_instance = openai_api_key


    async def handle_echo(self, message: types.Message):
        user_id = message.from_user.id
        received_text = message.text.lower()
        user_first_name = message.from_user.first_name
        user_last_name = message.from_user.last_name
        user_username = message.from_user.username
        self.messagedefault = self.db_connection.find_one_documents("messagedefault", {"applications": "telegram"})


        # Verificar si hay una conversación existente para el usuario
        if user_id not in self.user_conversations:
            self.user_conversations[user_id] = []

        # Añadir el mensaje del usuario a la conversación
        self.user_conversations[user_id].append(f"User: {received_text}")

         # Construir la cadena de conversación
        conversation_history = "\n".join(self.user_conversations[user_id]) + "\nChatGPT:"


        # Verificar si el usuario ya tiene datos almacenados en la base de datos
        user_data = self.db_connection.find_one_documents("users", {"user_id": user_id})
        if user_data:
            # Si ya existe un documento para el usuario, actualizar los datos del usuario
            user_data_update = {
                "user_name": user_first_name,
                "user_last_name": user_last_name,
                "user_username": user_username,
                "last_interaction_time": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
            }
            self.db_connection.update_one("users", {"user_id": user_id}, {"$set": user_data_update})
        else:
            # Si no existe un documento para el usuario, insertar uno nuevo
            user_data = {
                "user_id": user_id,
                "chatgpt_active": False,
                "spacy_model": self.spacy_model_name_default,
                "banned_gpt": False,
                "banned_chat": False,
                "user_name": user_first_name,
                "user_last_name": user_last_name,
                "user_username": user_username,
                "activation_time": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                "last_interaction_time": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
            }
            self.db_connection.insert_one("users", user_data)

        # Mapea comandos a funciones.
        command_functions = {
            '/hola': self.activate_chatgpt,
            '/chao': self.deactivate_chatgpt

        }

        # Obtener la función asociada al comando recibido.
        command_function = command_functions.get(received_text)

        if command_function:
            if asyncio.iscoroutinefunction(command_function):
                task = asyncio.create_task(command_function(user_id, message))
                response = await task
            else:
                response = command_function(user_id, message)

            await self.typing_indicator(user_id)
            #Obtener respuesta de OpenAI
            response = self.openai_instance.get_response(conversation_history)

        # Verificar si el mensaje contiene una palabra o frase de activación para generar una imagen
        if "/imagen" in received_text:
            # Llame al método generate_image de la instancia de OpenAI
            await self.typing_indicator(user_id)
            image_url = self.openai_instance.generate_image(conversation_history)
            # Espere a que se genere la imagen y luego envíela como respuesta al usuario
            await self.typing_indicator(user_id)
            await message.answer_photo(photo=image_url)
        else:
            print('ChatGPT desactivado.', user_id)
            if received_text == "/task":
                await self.typing_indicator(user_id)
                response = self.messagedefault['messages']['new_msg_features']
            if received_text == "/hola":
                await self.typing_indicator(user_id)
                # Activa la función de ChatGPT
                await self.activate_chatgpt(user_id,conversation_history)
            else:

                await self.typing_indicator(user_id)
                response = self.openai_instance.get_response(conversation_history)



        user_document = self.db_connection.find_one_documents("users", {"user_id": user_id})
        user_spacy_model = user_document.get("spacy_model", self.spacy_model_name_default) if user_document is not None else self.spacy_model_name_default
        # Save the conversation in MongoDB
        conversation = {
            "user_id": user_id,
            "user_username": user_username,
            "user_name": user_first_name,
            "user_last_name": user_last_name,
            "user_message": received_text,
            "bot_message": response,
            "model": user_spacy_model if user_id not in self.chatgpt_active_users else "ChatGPT",
            "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

        }
        self.db_connection.insert_one("conversations", conversation)
        # Add ChatGPT's response to the conversation
        self.user_conversations[user_id].append(f"ChatGPT: {response}")
        # Send the response to the chat
        await self.typing_indicator(user_id)
        await message.answer(response)

    async def activate_chatgpt(self, user_id, message):
        # Verificar si el chat ya está activo para el usuario
        user = self.db_connection.find_one_documents("users", {"user_id": user_id})
        if user and user.get("chatgpt_active"):
            await self.typing_indicator(user_id)
            status_message =  self.messagedefault['messages']['status_message']
            return status_message
        elif user and user.get("banned_gpt"):
            await self.typing_indicator(user_id)
            banned_message = self.messagedefault['messages']['banned_message']
            return banned_message
        else:
            # Si el usuario no está activo y no está baneado, actualizar el campo chatgpt_active.
            user_data = {"chatgpt_active": True}
            self.db_connection.update_one("users", {"user_id": user_id}, {"$set": user_data})
            self.chatgpt_active_users[user_id] = True

            # Verificar que el campo chatgpt_active esté actualizado en la base de datos
            user = self.db_connection.find_one_documents("users", {"user_id": user_id})
            if not user or not user.get("chatgpt_active"):
                # Si el campo no está actualizado, imprimir un mensaje de error y actualizar el campo chatgpt_active en la base de datos
                print(f"Error: chatgpt_active not updated for user {user_id}")
                self.db_connection.update_one("users", {"user_id": user_id}, {"$set": {"chatgpt_active": True}})

            # Mensaje de saludo del chatGPT
            await self.typing_indicator(user_id)
            user_first_name = message.from_user.first_name
            activate_message = f"¡Hola {user_first_name}! Soy ChatGPT, tu amigable asistente. Estoy aquí para ayudarte con tus preguntas. Por favor, adelante, pregúntame cualquier cosa y estaré encantado de responder. Cuando quieras despedirte, simplemente escribe /chao."
            return activate_message


    async def deactivate_chatgpt(self, user_id, message):
        # Cambiar el estado de activación del chat a False para el usuario en la base de datos
        self.db_connection.update_one("users", {"user_id": user_id}, {"$set": {"chatgpt_active": False}})
        self.chatgpt_active_users.pop(user_id, None)
        await self.typing_indicator(user_id)
        farewell_message = self.messagedefault['messages']['farewell_message']
        return farewell_message

    async def get_last_relevant_message(self, user_id, current_message):
        conversation = self.db_connection.find_all_documents("conversations", {"user_id": user_id}, sort_by="timestamp", sort_direction=-1)

        if not conversation:
            return None

        last_relevant_message = None
        max_similarity = 0

        # Procesar el mensaje actual con spaCy
        current_doc = self.spacy_processor.analyze_message(current_message)

        for message in conversation:
            # Procesar el mensaje previo con spaCy
            user_message = message.get("user_message", "").lower()
            if user_message.startswith(("/hola", "/chao", "/task", "/chatgpt")):
                continue  # Saltar mensajes de comandos
            doc = self.spacy_processor.analyze_message(user_message)

            # Calcular la similitud semántica entre el mensaje actual y el mensaje previo
            if 'doc' in current_doc:
                similarity = current_doc['doc'].similarity(doc)
            else:
                    similarity = None

            if similarity is not None and similarity > max_similarity:
                max_similarity = similarity
                most_similar = current_doc


        return last_relevant_message


    async def typing_indicator(self, user_id):
        await self.bot.send_chat_action(chat_id=user_id, action=types.ChatActions.TYPING)


    async def start(self, message: types.Message):
        # Obtener el primer nombre del usuario del objeto de mensaje
        user_first_name = message.from_user.first_name
        # Construir el mensaje de bienvenida con el nombre del usuario
        welcome_message = f"¡Hola! {user_first_name} Soy un bot de Telegram equipado con ChatGPT y spaCy, un asistente inteligente basado en inteligencia artificial. Para activar ChatGPT, escribe /hola. Si deseas salir, simplemente escribe /chao."
        # Enviar el mensaje de bienvenida al usuario
        await message.answer(welcome_message)
    async def run(self):
        # Método que inicia el polling del bot
        await self.dp.start_polling()
