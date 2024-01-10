import openai
import datetime
import time
import ssl
from email.message import EmailMessage
import smtplib

# import openai API key

def sendEmail(text):
    senderMail = ""
    receiverMail = ""
    password = ""

    subject = "your daily task is here"
    body = text

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = senderMail
    msg["To"] = receiverMail
    msg.set_content(body)
    
#   Add ssl certificate for additional security this step is completely optional
    context = ssl.create_default_context()

    with smtplib.SMTP_SSL("smtp.gmail.com", 465 , context = context) as smtp:
        smtp.login(senderMail, password)
        smtp.sendmail(senderMail, receiverMail, msg.as_string())


def get_completion_from_messages(messages, model="gpt-3.5-turbo", temperature=0):
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,  # this is the degree of randomness of the model's output
    )
    #     print(str(response.choices[0].message))
    return response.choices[0].message["content"]


while True:
    currentTime = datetime.datetime.now()
    curHourTime = currentTime.strftime("%H:%M:%S")
    print(curHourTime)
    if str(curHourTime) == "20:00:00":
        print("Time to fire the function")
        content = "Hey can you please suggest me a medium to hard level golang problem that should cover all the " \
                  "important topics of golang language"
        message = [
            {'role': 'system', 'content': 'You are An intelligent software developer proficient in golang and you '
                                          'like to educate your juniors about golang by giving them the problems of '
                                          'difficulty ranging from medium to insane which covers each and every topic '
                                          'of golang language including the net/http module, concurrency, '
                                          'datastructures, structs,pointers the questions given by you will contain '
                                          'the mixture of all these topics to make the student/junior more skillful '
                                          'with some advanced industry best practices.'},
            {'role': 'user', 'content': content}]
        actualResponse = get_completion_from_messages(message)
        sendEmail(actualResponse)
        break
    time.sleep(1)
