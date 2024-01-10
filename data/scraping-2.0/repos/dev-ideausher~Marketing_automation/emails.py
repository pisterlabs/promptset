import pandas as pd
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import imaplib
import email
import time
import csv
import os
import schedule
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import re
import openai
import datetime
# import sys
# from API_KEY import api_sec

scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
credentials = ServiceAccountCredentials.from_json_keyfile_name('C:/Users/shal1/OneDrive/Desktop/extra/pr/marketing-automations-401806-124a6a502fd9.json', scope)
gc = gspread.authorize(credentials)
spreadsheet = gc.open('Automated_Data')  
worksheet = spreadsheet.worksheet('Sheet1')  

data = worksheet.get_all_records()
df = pd.DataFrame(data)

smtp_server = 'smtp.gmail.com'
smtp_port = 587  
smtp_username = 'username'  
smtp_password = 'pass'


openai.api_key="API key here"
messages=[{"role":"system","content":"act as an email outreach expert and write an email with the following description."}]

def CustomChatGPT(user_input):
   messages.append({"role":"user","content":user_input})
   response=openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=messages
   )

   ChatGPT_Reply=response["choices"][0]["message"]["content"]
   messages.append({"role":"assistant","content":"ChatGPT_Reply"})
   return ChatGPT_Reply

#Function to send  the first mail
def sendmail_first(to_email,name,company_name,short_desc):
  chat=f"Write an opening introduction\nfor a {company_name} using below info\n{short_desc}\nEmail Body \nClosure"

  subject = f"{company_name}"
  body = CustomChatGPT(chat)
    

  message = MIMEMultipart()
  message['From'] = smtp_username
  message['To'] = to_email
  message['Subject'] = subject
  message.attach(MIMEText(body, 'plain'))

    
  try:
      server = smtplib.SMTP(smtp_server, smtp_port)
      server.starttls()  
      server.login(smtp_username, smtp_password)

   
      server.sendmail(smtp_username, to_email, message.as_string())
      print(f"Email sent to {to_email}")
      server.quit()
  except Exception as e:
      print(f"Failed to send email to {to_email}: {str(e)}")


  attempt=1
  print(f"Sent email for Attempt no. {attempt}")

#Function to send  the second mail
def sendmail_second(to_email,name,company_name,short_desc):
  chat=f"Write an opening introduction\nfor a {company_name} using below info\n{short_desc}\nEmail Body \nClosure"

  subject = f"{company_name}"
  body = CustomChatGPT(chat)
    

  message = MIMEMultipart()
  message['From'] = smtp_username
  message['To'] = to_email
  message['Subject'] = subject
  message.attach(MIMEText(body, 'plain'))

    
  try:
    server = smtplib.SMTP(smtp_server, smtp_port)
    server.starttls()  
    server.login(smtp_username, smtp_password)

   
    server.sendmail(smtp_username, to_email, message.as_string())
    print(f"Email sent to {to_email}")
    server.quit()
  except Exception as e:
    print(f"Failed to send email to {to_email}: {str(e)}")

  attempt=2
  print(f"Sent email for Attempt no. {attempt}")

#Funtion to send the third mail
def sendmail_third(to_email,name,company_name,short_desc):
  chat=f"Write an opening introduction\nfor a {company_name} using below info\n{short_desc}\nEmail Body \nClosure"

  subject = f"{company_name}"
  body = CustomChatGPT(chat)
    

  message = MIMEMultipart()
  message['From'] = smtp_username
  message['To'] = to_email
  message['Subject'] = subject
  message.attach(MIMEText(body, 'plain'))

    
  try:
    server = smtplib.SMTP(smtp_server, smtp_port)
    server.starttls()  
    server.login(smtp_username, smtp_password)

   
    server.sendmail(smtp_username, to_email, message.as_string())
    print(f"Email sent to {to_email}")
    server.quit()
  except Exception as e:
    print(f"Failed to send email to {to_email}: {str(e)}")

  attempt=3
  print(f"Sent email for Attempt no. {attempt}")


#Function to check for response
def check_response(email_search):
   response_list = []  
   imap_server="imap.gmail.com"
   mail = imaplib.IMAP4_SSL(imap_server)
   mail.login(smtp_username,smtp_password)
   mail.select("Inbox")

   search_criteria = f'UNSEEN FROM "{email_search}"'

    
   result, email_ids = mail.search(None, search_criteria)

    
   for email_id in email_ids[0].split():
       result, data = mail.fetch(email_id, "(RFC822)")
       msg = email.message_from_bytes(data[0][1])

       found_text_plain = False  
       response_content = ""
    
       if msg.is_multipart():
           for part in msg.walk():
               if part.get_content_type() == "text/plain":
                   response_content = part.get_payload(decode=True).decode("utf-8")
                   payload = part.get_payload(decode=True)
                   if payload is not None:
                       response_content = payload.decode("utf-8")
                       print("Received response for", email_search, ":", response_content)
                       found_text_plain = True
                       response_list.append(response_content)  
                       break
                   
       if not found_text_plain:
            payload = msg.get_payload(decode=True)
            if payload is not None:
                response_content = payload.decode("utf-8")
                response_list.append(response_content)
   return response_list

#function to mark the timestamps for all the sent emails 
def mark_timestamp(to_email,Type):
   cell = worksheet.find(to_email)
   timestamp_column_name = f'{Type} email time'
   timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

   
   if timestamp_column_name not in df.columns:
        worksheet.add_cols(1)
        df[timestamp_column_name] = ''


   cell = df[df['Email'] == to_email]
   if not cell.empty:
        cell_index = cell.index[0]
        df.loc[cell_index, timestamp_column_name] = timestamp
        update_worksheet(df)

#function to check if mail if already sent
def email_already_sent(to_email, email_type):
    sent_column_name = f'sent{email_type}'
    if sent_column_name not in df.columns:
        worksheet.add_cols(1)
        df[sent_column_name] = 'False'
    cell = df[df['Email'] == to_email]
    if cell.empty:
        return False 
    return cell[sent_column_name].iloc[0] == 'True'

#function to mark email as sent
def mark_email_as_sent(to_email, email_type):
    df.loc[df['Email'] == to_email, f'sent{email_type}'] = 'True'
    update_worksheet(df)
    mark_timestamp(to_email,email_type)

def update_worksheet(df):
    worksheet.update([df.columns.values.tolist()] + df.values.tolist())

def update_response_to_sheet(to_email, response):
    header = list(data[0].keys()) 
    values = [list(row.values()) for row in data] 

    if "Response" not in header:
        header.append("Response")
        for row in values:
            row.append("")

    for row in values:
        if row[header.index("Email")] == to_email:
             if isinstance(response, list):
                sanitized_response = ' '.join(response)
             else:
                sanitized_response = response
             sanitized_response = re.sub(r'\r\n|\r|\n', ' ', sanitized_response)
             row[header.index("Response")] = sanitized_response

    worksheet.update([header] + values)
    # worksheet.format("B2:B", {"wrapStrategy": "WRAP"})

  
def main():
   response = None
   for index, row in df.iterrows():
      to_email = row['Email']
      name=row['Name']
      company_name=row['Company Name']
      short_desc=row['Short Description']
      current_date = datetime.datetime.now()

      if not email_already_sent(to_email, "First"):
         sendmail_first(to_email,name,company_name,short_desc)
         mark_email_as_sent(to_email, "First")
      else:
        if response is None:
         response = check_response(to_email)
        #  print("response check 1")
         if response:
          update_response_to_sheet(to_email,response)
          # print("response block")
          break
         else:
          if not email_already_sent(to_email, "Second"):
            #  for record in worksheet.get_all_records():
              sent_time = datetime.datetime.strptime(row['First email time'], '%Y-%m-%d %H:%M:%S')
              if current_date - sent_time >= datetime.timedelta(minutes=3):
                 sendmail_second(to_email, name,company_name,short_desc)
                 mark_email_as_sent(to_email, "Second")
              else:
                 print("Not enough time has passed to send the second email")
          else:
            if response is None:
             response = check_response(to_email)
             print("response check 2")
             if response:
              update_response_to_sheet(to_email,response)
              break
             else:
              if not email_already_sent(to_email, "Third"):
              #  for row in worksheet.get_all_records():
                sent_time = datetime.datetime.strptime(row['Second email time'], '%Y-%m-%d %H:%M:%S')
                if current_date - sent_time >= datetime.timedelta(minutes=3):
                 sendmail_third(to_email, name,company_name,short_desc)
                 mark_email_as_sent(to_email, "Third")
                else:
                 print("Not enough time has passed to send the third email")
              else:
                print("Third email already sent")
                print("All follow-up emails sent")

schedule.every(1).minutes.do(main)
while True:
    schedule.run_pending()
    time.sleep(1)