import openai
import os
from pathlib import Path
from dotenv import load_dotenv
from django.core.mail import BadHeaderError, send_mail
from smtplib import SMTPException
from pathlib import Path

from backend.settings import EMAIL_HOST_USER
from rest_framework.response import Response
from rest_framework import status

file_path = '{}'.format(Path(__file__).resolve().parent.parent.parent)


env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

openai.api_key = os.getenv("OPENAI_API_KEY")

DEBUG_MODE = False

def handle_uploaded_file(filename, file):
    filename = os.path.join(file_path, filename)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open("{}".format(filename), "wb+") as destination:
        for chunk in file.chunks():
            destination.write(chunk)


def send_an_email(subject, message, email_from, recipient_list):
    try:
        send_mail(subject, message, email_from, recipient_list)
    except BadHeaderError:
        return Response({
            'msg': "email header is wrong",
        }, status=status.HTTP_406_NOT_ACCEPTABLE)
    except SMTPException as e:  # It will catch other errors related to SMTP.
        print('There was an error sending an email.' + e)
    except:  # It will catch All other possible errors.
        print("Mail Sending Failed!")

def send_translate_completion_email(username, user_email, filename):
    print("Sending email")
    # Sends email to the user
    subject = 'Translation is completed'
    message = f"""Hi {username}, 
            
        Your {filename} has been translated! Please go to page: https://astoryai.xyz/file_download_converted.html to download the file. 
            
Thanks,
AStory.ai
    """
    email_from = f'AStory.ai <{EMAIL_HOST_USER}>'
    recipient_list = [user_email]
    return send_an_email(subject, message, email_from, recipient_list)

def send_translate_failure_email(username, user_email, filename):
    print("Sending failure email")
    # Sends email to the user
    subject = 'Translation is failed'
    message = f"""Hi {username}, 
            
        We're sorry that your {filename} hasn't been translated. Please try again. 
            
Thanks,
AStory.ai
    """
    email_from = f'AStory.ai <{EMAIL_HOST_USER}>'
    recipient_list = [user_email]
    return send_an_email(subject, message, email_from, recipient_list)
