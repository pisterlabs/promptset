import random
import os
from validate_email import validate_email
import smtplib
import pymongo
import os
from dotenv import load_dotenv
import datetime
from datetime import datetime as dt
import openai

load_dotenv()

project_server = pymongo.MongoClient(os.getenv("pymongo_client"))
db_users = project_server['users']
users_column = db_users['users_list']
tweet_column = db_users['users_tweet']
db_content = project_server['contents']
home_content_column = db_content['home_content']
chat_time_line = db_users['chat_history']

doc_column = db_users['doc_column']

def generate_otp(user_email):
    new_otp = random.randint(100000, 999999)
    sender_email = os.getenv("email_id")
    sender_password = os.getenv("email_password")
    receiver_email = user_email
    message_text = f"The OTP is {new_otp}"
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(sender_email, sender_password)
    if validate_email(receiver_email):
        server.sendmail(sender_email, receiver_email, message_text)
        return new_otp
    else:
        return 0
    

def add_user(id, name,  email, phone, password, dob):
    print(dob)
    dob_date = dt.strptime(dob, "%Y-%m-%d")
    users_column.insert_one({'User_Id':id, "Name":name, "Email":email , "Phone":phone , "Password":password, "DOB":dob_date, "Role":'user', "Date_Added":datetime.datetime.utcnow()})

def send_mail(mail_id, message):
    sender_email = os.getenv("email_id")
    sender_password = os.getenv("email_password")
    receiver_email = mail_id
    message_text = message
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(sender_email, sender_password)
    if validate_email(receiver_email):
        server.sendmail(sender_email, receiver_email, message_text)
    else:
        return 0
    
def add_to_oms(image_link , link_oms , outsourcing_type):
    home_content_column.insert_one({'img':image_link, 'link':link_oms, 'type':outsourcing_type})

def add_asmr_links(asmr_links):
    print("added")
    print(home_content_column.find_one({'Block':'asmr_link'}))
    print(asmr_links)
    home_content_column.update_one({'Block':'asmr_link'},{'$set':{'asmr_links' : asmr_links}})

def add_tweet(message, name):
    print('HAI da Parama')
    tweet_column.insert_one({'Name':name, 'Message':message, "Date_Posted":datetime.datetime.utcnow()})

def chat_time(p1, p2, message):
    if p1>p2:
        p = p1+p2
    else:
        p = p2+p1
    chat_time_line.insert_one({'Time_line':p, 'sender':p1, 'message':message})


def get_chat_time_line(p1, p2):
    if p1>p2:
        p = p1+p2
    else:
        p = p2+p1
    chats = chat_time_line.find({'Time_line':p})
    return chats

openai.api_key = "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"

def chatbot(message):
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=message,
        temperature=0.5,
        max_tokens=1024,
        n=1,
        stop=None,
    )
    response_text = response.choices[0].text.strip()
    return response_text

def add__doc(id, name, specialist, year, city, address, phone, img_binary, clinic, con_fee):
    doc_column.insert_one({'Doc_Id':id, 'Name':name ,'Specialist':specialist, 'year':year, 'City':city, 'Address':address, 'Phone':phone, 'Image':img_binary ,'Clinic':clinic, 'Consultation_Fees':con_fee})
