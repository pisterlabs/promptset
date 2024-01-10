from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from .models import CustomUser,MessageModel
from .serializers import MessageModelSerializer
from rest_framework import status
from langchain.schema import (AIMessage,HumanMessage,SystemMessage)
from langchain.chat_models import ChatOpenAI
from .getkeys import getKeys,chatInfo
from langchain.llms import OpenAI
import json
from langchain.memory import ConversationSummaryBufferMemory


# Create your views here.
class GetBase(APIView):
    def get(self,request):
        test = MessageModel.objects.get(id=12)
        serializer = MessageModelSerializer(test)
        data={
            "data":serializer.data
        }
        return Response(data,status=200)


class GetUserChats(APIView):
    def get(self,request,*args, **kwargs):
        username = kwargs['username']
        try:
            user = CustomUser.objects.get(username=username)
        except CustomUser.DoesNotExist:
            user = CustomUser.objects.create(username=username)

        messages = MessageModel.objects.filter(user=user)

        message_serializer = MessageModelSerializer(messages, many=True)


        data = {
            'messages': message_serializer.data
        }

        return Response(data, status=status.HTTP_200_OK)



class AskModelChat(APIView):
    def post(self,request,*args,**kwargs):
        openAikey = getKeys()
        username = kwargs['username']

        try:
            user = CustomUser.objects.get(username=username)
        except CustomUser.DoesNotExist:
            user = CustomUser.objects.create(username=username)
        
        messages = MessageModel.objects.filter(user=user)
        message_serializer = MessageModelSerializer(messages, many=True)
        user_messages = message_serializer.data

        try:
            data = json.loads(request.body.decode('utf-8'))
            print(data)
        except UnicodeDecodeError:
            return Response('Invalid request body encoding', status=400)
        except json.JSONDecodeError:
            return Response('Invalid JSON data', status=400)
        
        prompt = data['text']
        

        llm = OpenAI(temperature=0.7,openai_api_key=openAikey)
        chat = ChatOpenAI(openai_api_key=openAikey)
        memory = ConversationSummaryBufferMemory(llm=llm,max_token_limit=10)
        placeholder = []

        for message in user_messages:
            if(message['type'] == 'human'):
                placeholder.append({'input':message['message']})
            else:
                placeholder.append({'output':message['message']})
                print(placeholder[0],placeholder[1])
                memory.save_context(placeholder[0],placeholder[1])
                placeholder =[]
        history = memory.load_memory_variables({})['history']
        print(history)

        messages =[
            SystemMessage(content=chatInfo),
            SystemMessage(content=f"Current user name is {username}"),
            SystemMessage(content=history),
            HumanMessage(content=prompt)
        ]
        result = chat(messages)
        print(result)
        res = result.to_json()
        response = ""
        text_string = res['kwargs']['content']
        if text_string.startswith("AI:"):
            response = text_string[len("AI:"):].lstrip()
        elif text_string.startswith("An AI:"):
            response = text_string[len("An AI:"):].lstrip()
            pass
        else:
            response = text_string
        # MessageModel.objects.create(user = user,type="human",message=prompt)
        # ai_reply = MessageModel.objects.create(user = user,type="ai",message=response)
        # serializer = MessageModelSerializer(ai_reply)
        return Response("workinf",status=status.HTTP_200_OK)