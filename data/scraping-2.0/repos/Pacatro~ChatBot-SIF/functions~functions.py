# Note: you need to be using OpenAI Python v0.27.0 for the code below to work
import os
import openai
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from dotenv import load_dotenv

def bot_response(message):
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")
    
    description = """
        Eres un chatbot dedicado a la atención al cliente de una empresa llamada FixNexus dedicada a la venta 
        y reparación de productos tecnológicos como mandos de consola, ordenadores, periféricos, etc. Tu labor 
        será la de atender todas las dudas de los clientes de la empresa de la manera más educada y eficiente
        posible.
    """
    
    response = openai.ChatCompletion.create(
        model = "gpt-3.5-turbo",
        messages = [
            {"role": "system", "content": description},
            {"role": "user", "content": message}
        ]
    )
        
    return response['choices'][0]['message']['content']

def send_email(username, user_email, user_message, receiver_email):
    load_dotenv()
    
    mail_content = ""
    subject = ""

    if receiver_email == os.getenv("FIXNEXUS_ADDRESS"):
        subject = f"El usuario {username} quiere contactar con nosotros"
        mail_content = f"El usuario {username} con email {user_email} quiere contactar con nosotros por:\n{user_message}"
        
    else:
        subject = "Hemos recibido tu mensaje!"
        mail_content = f"Hola {username}!\n Hemos enviado tu sugerencia a nuestro equipo de soporte y te responderemos lo antes posible."
        
    sender_email = os.getenv("FIXNEXUS_ADDRESS")
    sender_pass = os.getenv("FIXNEXUS_PASS")
    
    message = MIMEMultipart()
    message['From'] = sender_email
    message['To'] = receiver_email
    message['Subject'] = subject
    
    message.attach(MIMEText(mail_content, 'plain'))
    
    session = smtplib.SMTP('smtp-mail.outlook.com', 587)
    session.starttls()
    session.login(sender_email, sender_pass)
    text = message.as_string()
    session.sendmail(sender_email, receiver_email, text)
    session.quit()    
    