import requests
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication, QMainWindow, QTextEdit
from rest_framework import viewsets, status
from rest_framework.response import Response
from .serializers import EmailSerializer
import time
from email_msg_generator import email_generator
from rest_framework.permissions import IsAuthenticated
from email_msg_generator.models import OpenAiUserModel,OpenAiAdminModel
from email_msg_generator.serializers import OpenAiModelSerializers
from django.shortcuts import get_list_or_404,get_object_or_404
import os
import openai
import time
import csv
from email_msg_generator.models import EmailGeneratorModel
import json
from django.core.mail import EmailMultiAlternatives
from rest_framework import status
from rest_framework.response import Response
from rest_framework import viewsets
from .serializers import EmailSerializer
from .models import EmailGeneratorModel

import os
from channels.generic.websocket import AsyncWebsocketConsumer
from channels.routing import ProtocolTypeRouter, URLRouter
from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync
from django.core.asgi import get_asgi_application
from django.urls import re_path
from rest_framework import status
from rest_framework.response import Response
from rest_framework.viewsets import ViewSet
import openai
from django.core.mail import EmailMessage
from django.conf import settings
import requests
from django.utils import timezone
from pathlib import Path
import random
from .serializers import EmailMessageSerializer

class EmailGeneratorViewSet(viewsets.ViewSet):
    http_method_names = ['get','post']
    permission_classes = [IsAuthenticated]
    serializer_class = EmailSerializer

    def create(self, request):
        serializer = self.serializer_class(data=request.data)
        
        serializer.is_valid(raise_exception=True)
        self.number_of_mail = serializer.validated_data['number_of_mail']
        self.promt = serializer.validated_data['prompt']
        access_token=serializer.validated_data['access_token']

        self.email_keywords_list=['follow_up','top_up','advert','register reminder']
        
        
        import os
        
        import requests

        # Load the .env file
        
        # file_path = 'D:/contractwork/emailaigenerator/current_user_token.txt'
        
        # with open(file_path, 'r') as file:
        #     access_token = str(file.read())
        # Make the API request and retrieve the OpenAI key
        url = 'http://localhost:8000/api/openaiusers/'
        headers = {
            'Authorization': f'Bearer {str(access_token)}',
            'Content-Type': 'application/json'
        }

        response = requests.get(url, headers=headers)
       
        data = response.json()
        openai_key = data.get('open_ai_key')
        content = response.content.decode('utf-8')
        print(content)
        openai.api_key=openai_key
        self.follow_up_prompt=["Please generate a follow-up message for this website link.",
        "I need a message to tell user to folow, can you help?","Could you write a message to go along with this website link?", "I'm sharing a website link and need a short message for users to follow it.","Can you compose a follow-up message for this website link, please?","I require a brief message for the following website link.","Hey, can you help me out with a message for this website link?","Please assist me in creating a follow-up message for this link"]
        self.software_sales_prompt=['Please generate a message to market my software providing my app site url','help me to generate a message to market my software providing my app site url']
        # self.register_up_promt=['action message to tell user to register with a particular link']

        from core.wallet.models import UsdModel
        user=request.user
        if UsdModel.objects.get(user=user).amount<10:
            return Response({"unique_email_text":"Top up at least 10USD to make use of the application"})
        
        else:
            if str(self.promt)=='follow_up'.lower():
                self.follow_up_prompt=random.choice(self.follow_up_prompt)
                
                response = openai.Completion.create(
                engine="text-davinci-003",
                prompt=self.follow_up_prompt,
                temperature=1.0,
                max_tokens=45,
                top_p=1,
                n=self.number_of_mail,
                frequency_penalty=2.0,
                presence_penalty=2.0,
                    )
                
            
            
                
                emails_list = []
                for unique_email  in range(self.number_of_mail):
                    email_entry = {
                        f"id_{unique_email}": unique_email,
                        f"message_body_{unique_email}": f'{str(response["choices"][unique_email]["text"])}'
                    }
                    emails_list.append(email_entry)
                emails_list=emails_list   
                    
                    # data = [{'id': f'{unique_email}', f'message_body_{unique_email}': f'{response["choices"][unique_email]["text"]}'}]
                    # # Generate the list of dictionaries in the desired format
                    # formatted_data = self.generate_emails_data(data)

                    # Get the path to the "Downloads" directory
                downloads_dir = Path.home() / "Downloads"

                # Create the file path
                file_path = downloads_dir / "emails.json"
                current_user=UsdModel.objects.get(user=user)
                current_user_amount=current_user.amount
                current_user_balance=current_user_amount-(self.number_of_mail*0.01)
                current_user.amount=current_user_balance
                current_user.save()
                # Write the formatted data to the file
                with open(file_path, 'w') as file:
                    json.dump(emails_list, file, indent=2)
                
                
                    
                return Response({'unique_email_text':emails_list},status=status.HTTP_200_OK)

            if str(self.promt)=='software sales'.lower():
                self.software_sales_prompt=random.choice(self.software_sales_prompt)
                
                response = openai.Completion.create(
                engine="text-davinci-003",
                prompt=self.follow_up_prompt,
                temperature=1.0,
                max_tokens=45,
                top_p=1,
                n=self.number_of_mail,
                frequency_penalty=2.0,
                presence_penalty=2.0,
                    )
                
            
            
                    
                emails_list = []
                for unique_email  in range(self.number_of_mail):
                    email_entry = {
                        f"id_{unique_email}": unique_email,
                        f"message_body_{unique_email}": f'{str(response["choices"][unique_email]["text"])}'
                    }
                    emails_list.append(email_entry)
                emails_list=emails_list   
                    
                    # data = [{'id': f'{unique_email}', f'message_body_{unique_email}': f'{response["choices"][unique_email]["text"]}'}]
                    # # Generate the list of dictionaries in the desired format
                    # formatted_data = self.generate_emails_data(data)

                    # Get the path to the "Downloads" directory
                downloads_dir = Path.home() / "Downloads"
                current_user=UsdModel.objects.get(user=user)
                current_user_amount=current_user.amount
                current_user_balance=current_user_amount-(self.number_of_mail*0.01)
                current_user.amount=current_user_balance
                current_user.save()
                # Create the file path
                file_path = downloads_dir / "emails.json"

                # Write the formatted data to the file
                with open(file_path, 'w') as file:
                    json.dump(emails_list, file, indent=2)
                
                
                    
                return Response({'unique_email_text':emails_list},status=status.HTTP_200_OK)
        
                # self.email_terminal.setText(self.email_terminal.toPlainText()+f"\n {unique_email}"+str(response["choices"][unique_email]["text"]))
                # self.email_terminal.setText(self.email_terminal.toPlainText()+"\n"+str(response["choices"][1]["text"]))
                # self.email_terminal.setText(self.email_terminal.toPlainText()+"\n"+str(response["choices"][2]["text"]))
                # self.email_terminal.setText(self.email_terminal.toPlainText()+"\n"+str(response["choices"][3]["text"]))
    #         for unique_mail in range(self.number_of_messages):
                

        # elif str(self.promt)=='top_up'.lower():
        #     response = openai.Completion.create(
        #     engine="text-davinci-003",
        #     prompt=self.promt,
        #     temperature=1.0,
        #     max_tokens=50,
        #     top_p=1,
        #     n=self.number_of_mail,
        #     frequency_penalty=2.0,
        #     presence_penalty=2.0,
        #     )
        #     pass
        
        
            
            

    
    def get_queryset(self):
        return EmailGeneratorModel.objects.all()
   
   

from .serializers import OpenAiUserSerializer
from rest_framework import viewsets
from rest_framework.permissions import IsAuthenticated

class OpenAiUserViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = OpenAiUserModel.objects.all()
    serializer_class = OpenAiUserSerializer
    permission_classes = [IsAuthenticated]
    http_method_names=['get','post']

    def get_queryset(self):
        user = self.request.user
        return OpenAiUserModel.objects.filter(user=user)

    # def list(self, request, *args, **kwargs):
    #     queryset = self.get_queryset()
    #     serializer = self.serializer_class(queryset, many=True)
    #     open_ai_key = queryset.first().open_ai_key if queryset.exists() else None
    #     response_data = {
    #         'open_ai_key': open_ai_key,
    #         'users': serializer.data
    #     }
    #     return Response(response_data)
    def list(self, request):
        try:
            user = request.user
            current_user = OpenAiUserModel.objects.filter(user=user).latest('id')    # Retrieve the UsdModel based on the user's ID
            open_api_key = current_user.open_ai_key
            return Response({"open_ai_key":open_api_key},status=status.HTTP_200_OK)
        except OpenAiUserModel.DoesNotExist:
        # Create a new UsdModel for the user
            unassigned_keys = OpenAiAdminModel.objects.filter(assigned=False).first()

            if unassigned_keys:
                
                unassigned_keys.assigned = True
                
                open_api_key = unassigned_keys.open_ai_key
                
                unassigned_keys.save()
                
                OpenAiUserModel.objects.create(custom_user_key_id=unassigned_keys.custom_user_key_id,
                open_ai_key=open_api_key,time_of_assigning=timezone.now(),user=user)
                print("User assigned key successful")
                user = request.user
                current_user = OpenAiUserModel.objects.filter(user=user).latest('id')    # Retrieve the UsdModel based on the user's ID
                open_api_key = current_user.open_ai_key
                return Response({"open_ai_key":open_api_key},status=status.HTTP_200_OK)

from .models import EmailMessageModel 
class EmailMessageViewsets(viewsets.ModelViewSet):
    serializer_class=EmailMessageSerializer
    http_method_names=['post','get']
    permission_classes=[IsAuthenticated,]

    def create(self,request):
        user = request.user
        serializer=self.serializer_class(data=request.data)
        serializer.is_valid(raise_exception=True)
        title_of_message=serializer.validated_data['title']
        message_body=serializer.validated_data['message_body']
        maillist=serializer.validated_data['maillist']
        
        site_url=serializer.validated_data['site_url']
        

    
        EmailMessageModel.objects.create(user=user,title=title_of_message,message_body=message_body,maillist=maillist,site_url=site_url)

    

        subject, from_email, to = "hello", f"{settings.EMAIL_HOST_USER}", f"kezechristian@gmail.com"
        text_content = "This is an important message."
        html_content = """<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Mail Template</title>
    <style>div.messagebody{
            height: 100vh;
            width: 100vw;
            margin: 0;
            
            background:white;
        }
        div.footer{
            
            height: 2rem;
            background-color: rgb(91, 54, 169);;
        }
        div.footer p,div.footer p a{
            color:white;
            text-align: center;
            text-decoration: none;
        }
        div.navdiv{
            height: 2.5rem;
            padding: 0;
            color:white;
            text-align: center;
        }
        @media only screen and (max-width: 600px)  {
        body{
            width: 100vw;
            padding:0;
            margin:0;
        }
        
        div.messagebody{
            width: 100%;
            height: 80vh;
            background-color: white;
        }
        div.messagebody p{
            word-wrap: break-word;
            text-align: center;
        }
        }
        
        
    </style>
</head>
<body style="background-color:rgb(91, 54, 169);">
    <div class="navdiv" >
        <p>Codeblaze Academy</p>
        <p>Unlocking potentials through learning</p>
    </div>

    
    <div class="messagebody">
        <p>"""+f"""{message_body}</a> </p>
    </div>
    <div class="footer">
        <p><a  href="">support@codeblazeacademy.net</a></p>
    </div>
</body>
</html>"""

        msg = EmailMultiAlternatives(subject, text_content, from_email, [to])
        msg.attach_alternative(html_content, "text/html")

        msg.send()


        return Response({"message_sent_response":"message_sent successfully"},status=status.HTTP_200_OK)
        

    def get_queryset(self):
        return EmailMessageModel.objects.all()
        
   