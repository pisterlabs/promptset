from django.conf import settings
from channels.db import database_sync_to_async
from channels.generic.websocket import AsyncWebsocketConsumer
from django.contrib.auth import authenticate
from asgiref.sync import async_to_sync
from .tools.embeddings import crear_embeddings_docs
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
from .tools.base import ejecutar_sql
import os
import json
import asyncio
import threading


class ChatConsumer(AsyncWebsocketConsumer):

    def __init__(self, *args, **kwargs):
        self.sql_prompt = """Eres un experto en convertir texto a SQL.
        No des niguna explicación solo la consulta o consultas sql.
        {contexto}
        Pregunta:{input}
        """

        path = os.path.join(settings.BASE_DIR, "app", "valle_ia", "info_db", "informacion.txt")
        #self.rt = crear_embeddings_docs(file_db=path)
        #self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        
        super().__init__(*args, **kwargs)

    def enviar_mensaje_sync(self, mensaje, tipo="text"):
        # Enviar mensaje al grupo
        async_to_sync(self.channel_layer.group_send)(
            self.group_name,
            {
                'type': 'send_message',
                'message': mensaje,
                "tipo":tipo,
            }
        )


    @database_sync_to_async
    def async_authenticate(self, user, token):
        user = authenticate(pk=user, token=token)
        return user
    
    async def connect(self):
       self.user = self.scope['url_route']['kwargs']['user']
       self.group_name = settings.EMPRESA + "__gestion_ia__" + self.user
       
       await self.channel_layer.group_add(
            self.group_name,
            self.channel_name
        )

       await self.accept()

    async def disconnect(self, close_code):
        await self.channel_layer.group_discard(
            self.group_name,
            self.channel_name
        )

    async def receive(self, text_data):
        text_data_json = json.loads(text_data)
        token = text_data_json.get("token", {"user":0, "token": ""})
        user = await self.async_authenticate(token["user"], token["token"])
        
        if user is None:
             await self.channel_layer.group_send(
                self.group_name,
                {
                    'type': 'send_message_text',
                    'message': "Error en la autentificacion. usuario no valido."
                }
            )
        else:
            message = text_data_json["message"]["query"]
            tabla = text_data_json["message"]["tabla"]
        
            # Ejecutar acciones en segundo plano en un hilo
            loop = asyncio.get_running_loop()
            thread = threading.Thread(target=self.background_task, args=(message, tabla,))
            thread.start()
            await asyncio.sleep(0)
             
    # Receive message from room group
    async def send_message(self, event):
        # Send message to WebSocket
        await self.send(text_data=json.dumps({
            'type': "anwser",
             event["tipo"]: event['message'],
        }))

    
    def background_task(self, message, tabla):
        sql_prompt = PromptTemplate(input_variables=["contexto", "input"], template=self.sql_prompt)
        preguntaHuman = HumanMessage(content=sql_prompt.format(input=message, contexto=self.rt.get_relevant_documents(message)))
        respuesta = self.llm(messages=[preguntaHuman])
        print("Respuesta ia ", respuesta.content)
        respuesta = ejecutar_sql(respuesta.content)
        print("Respuesa ord ", respuesta)
        respuesta = self.llm(messages=[
            HumanMessage(content=f"""Imagina que a la pregunta {message} la respuesta de un ordenado ha sido {respuesta}. 
                                     Podrias darme una respuesta simplifícada?. Muchisimas gracias."""),
            ])
        self.enviar_mensaje_sync(respuesta.content)
    