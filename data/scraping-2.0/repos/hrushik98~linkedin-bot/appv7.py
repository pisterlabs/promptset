import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from googlesearch import search
import openai
import pandas as pd
import regex as re
import time
import csv
import requests
import json
import ast
import os
import wget
import streamlit as st


global employee_export_id
global auto_connect_id
global search_export_data_download_link
global employee_export_data_download_link
global auto_connect_data_download_link
global search_export_id
global phantom_key
global hunter_api_key
global clearbit_api_key
global size_of_company
global company_size_array
global generated_target_profiles


openai.api_key = st.secrets["openai_api_key"]

phantom_key = "hh2UOaFpWrojhw458ZzxSKxuBA5xGs6TVmG7CPnKZfY"
employee_export_id = "843885985636439"
auto_connect_id = "7205681370771394"
search_export_id = "215072976871448"
search_export_data_download_link = "https://cache1.phantombooster.com/xNdp30g5TSk/Fu1Js2zQCMEF1LJEvVkh5A/result.csv"
employee_export_data_download_link = "https://cache1.phantombooster.com/xNdp30g5TSk/hpQJt3BoutjPGrbh3MyWow/result.csv"
auto_connect_data_download_link = "https://cache1.phantombooster.com/xNdp30g5TSk/IefBSFr0yMDFv4PPOVyonA/database-linkedin-network-booster.csv"

hunter_api_key = "ebfe5b9f28930e4b72e722cdbfbde2c3720a00e2" #mail
clearbit_api_key = "sk_6fa9397ac4655b5c9a0ef456dcab8057" #phone

mailjet_api_key = "dc2f95a12cfe7df90165aaf6b7bac8f2"
mailjet_api_secret = "796dc22c8c52d3c589b9f552e4d98429"



global locations
locations = {'canada': '22101174742',
 'usa': '22103644278',
 'india': '22102713980',
 'australia': '22101452733',
 'united kingdom': '22101165590',
 'uk': '22101165590',
 'mexico': '22103323778',
 'philippines': '22103121230',
 'germany': '22101282230',
 'luxembourg': '22104042105',
 'england': '22102299470',
 'france': '22105015875',
 'spain': '22105646813',
 'netherlands': '22102890719'}

def extract_location_id(link, location):
  global locations
  locations[f'{location.lower()}'] = link[68:79]
  st.write(link[68:79])

size_of_company = {
    "small": "5B%22B%22%2C%22C%22%2C%22D%22",
    "medium": "5B%22E%22%2C%22F%22%2C%22G%22",
    "large": "5B%22H%22%2C%22I%22"
}

def search_link_generator(location, keyword, company_size):
  global size_of_company
  total_company_size = "5B%"
  for size in company_size:
    total_company_size = total_company_size + size_of_company[f'{size}'][3:] + "%2C%"
  location_id = locations[f'{location.lower()}']
  return f"https://www.linkedin.com/search/results/companies/?companyHqGeo=%5B%{location_id}%22%5D&companySize=%{total_company_size[0:-4]}%5D&keywords={keyword}&origin=FACETED_SEARCH&sid=9YG"


global ps,pdn
ps = ""
pdn = ""

def reset():
  if os.path.exists("profile_list.csv"):
    
     os.remove("profile_list.csv")
  if os.path.exists("profiles_with_email_phone.csv"):
      os.remove("profiles_with_email_phone.csv")
  if os.path.exists("profiles_with_emails.csv"):
      os.remove("profiles_with_emails.csv")

  if os.path.exists("company.csv"):
    os.remove("company.csv")
  if os.path.exists("companies.csv"):
    os.remove("companies.csv")
  if os.path.exists("profile.csv"):
    os.remove("profile.csv")
  if os.path.exists("profiles.csv"):
    os.remove("profiles.csv")




def domain_finder(company_name):
  global hunter_api_key
  import re
  search_results = search(company_name, num_results=1, lang="en")
  for i, result in enumerate(search_results, start=1):
      domain = result
  try:
    domain = domain.replace("www.","")
  except:
    pass
  pattern = r'https://([\w.-]+)'
  matches = re.findall(pattern, domain)
  if matches:
      url = matches[0]
  return url


def email_finder(first_name, last_name, company_name):
  import json
  domain= domain_finder(company_name)
  url = "https://api.hunter.io/v2/email-finder"

  api_key = f"{hunter_api_key}"   #Hunter.io api_key

  params = {
      "domain": f"{domain}",
      "first_name": f"{first_name}",
      "last_name": f"{last_name}",
      "api_key": api_key
  }
  response = requests.get(url, params=params)

  final = response.content.decode()
  parsed_data = json.loads(final)
  email = parsed_data['data']['email']
  score = parsed_data['data']['score']

  url = "https://api.hunter.io/v2/email-verifier"

  params = {
      "email": email,
      "api_key": api_key
  }

  # Send the GET request
  response = requests.get(url, params=params)
  final = response.content.decode()
  parsed_data = json.loads(final)
  parsed_data['data']['status'] == "valid"

  return email, score

def phone_number_finder(email):
  global clearbit_api_key
  import json
  url = f"https://person-stream.clearbit.com/v2/combined/find?email={email}"
  api_key = f"{clearbit_api_key}"        #clearbit.com api_key
  headers = {
      "Authorization": f"Bearer {api_key}"
  }
  response = requests.get(url, headers=headers)
  final = response.content.decode()
  parsed_data = json.loads(final)

  return parsed_data['company']['site']['phoneNumbers']

def target_persona_decider(user_response):
  system_content = """
  You will be given a user response, you have to figure out the target profiles the user is looking for. Once you have found them return them using the word OR as a seperator.
  Only limit the target persona to 6. Any more than that please avoid and dont include them. Only consider the first six.
  Example:
  user: I'm thinking data scientist and maybe one full stack engineer
  target_persona: Data Scientist OR Full Stack Engineer

  Example:
  user: I guess ceo, cto and a HR manager.
  target_persona: CEO OR CRO OR HR manager

  Example:
  user: ceo, cto, cfo, head of sals, vp sales, sales enablement and data scientist and full stack
  target_persona: CEO OR CTO OR CFO OR Head of Sales OR VP Sales ORSales Enablement
  """

  user_content = f"{user_response} \target_persona: "

  m1 = [{"role": "system", "content": f"{system_content}"},
        {"role": "user", "content": f"{user_content}"}]
  for i in range(0,10):
      try:
          
          result = openai.ChatCompletion.create(
            model="gpt-4",
            max_tokens = 100,
            temperature =0.8,
            messages=m1)
          response = result["choices"][0]['message']['content']
          break
      except:
        continue
  target_persona = str(response)
  return target_persona

def display_profile_list(profiles):
    
    try:
        profiles = profiles[pd.isna(profiles['error'])].reset_index(drop=True)
    except:
        profiles = profiles
    st.header("Profiles:")
    for index, i in enumerate(range(len(profiles))):
        firstName = profiles['firstName'][i]
        lastName = profiles['lastName'][i]
        profileUrl = profiles['profileUrl'][i]
        st.write(index+1, " " ,firstName,lastName, "-" , profileUrl)

def decide_number_of_profiles(user_response):
  system_content = """
  You will be given a user response, you have to figure the number the user is mentioning. Only return this number and nothing else.
  Example:
  user: around 30
  number: 30

  Example:
  usee: sixty maybe
  number: 60
  """

  user_content = f"{user_response} \nnumber: "

  m1 = [{"role": "system", "content": f"{system_content}"},
        {"role": "user", "content": f"{user_content}"}]
  for i in range(0,10):
      try:
          
          result = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            max_tokens = 5,
            temperature =0.8,
            messages=m1)
          response = result["choices"][0]['message']['content']
          break
      except:
        continue
  number = str(response)
  return int(number)


def decide(user_response):
  system_content = """
  You will be given a user response. You have to return the intention either "YES" or "NO" by analysing what the user has responded with.
 For "NO" st.write "n"

   For "YES" st.write "y"
  Example:
  User: Nah.. I dont like it
  Intention: n

  User: sounds good.
  Intention: y
  """

  user_content = f"{user_response} \nIntention: "

  m1 = [{"role": "system", "content": f"{system_content}"},
        {"role": "user", "content": f"{user_content}"}]
  for i in range(0,10):
      try:
          
          result = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            max_tokens = 5,
            temperature =0.8,
            messages=m1)
          response = result["choices"][0]['message']['content']
          break
      except:
        continue
  answer = str(response)
  return answer


def outreach_1():
    if 'ran_profile_validator' in st.session_state and st.session_state.ran_profile_validator == "":
        profile_validator()
        st.session_state.ran_profile_validator = "yes"
    if st.session_state.ran_profile_validator == "yes":
        profiles = pd.read_csv("profiles.csv")
        try:
            profiles = profiles[pd.isna(profiles['error'])].reset_index(drop=True)
        except:
            profiles = pd.read_csv("profiles.csv")
        display_profile_list(profiles)
        connect_or_not = st.text_input("Would you like to connect with these people?\n")
        if connect_or_not == "":
            pass
        else:
            if 'connect_or_not' in st.session_state and st.session_state.connect_or_not != "":
                st.write(st.session_state.connect_or_not)
            else:
                # Make the API call and store the response in session state
                st.session_state.connect_or_not = decide(connect_or_not)
                connect_or_not = st.session_state.connect_or_not
                st.write(connect_or_not)

            if connect_or_not == "n" or connect_or_not == "N" or connect_or_not == "NO" or connect_or_not == "no":
                profiles_list = profiles
                profiles_list.to_csv("profile_list.csv")
                st.success("profile_list.csv has been created")
                st.download_button(label="Download", data=profiles_list.to_csv(), file_name="profile_list.csv", mime="text/csv")
            else:
                send_connection_message_or_not = st.text_input("Would you like to send a connection message?\n")
                if send_connection_message_or_not == "":
                    pass
                else:
                    if 'send_connection_message_or_not' in st.session_state and st.session_state.send_connection_message_or_not != "":
                        st.write(st.session_state.send_connection_message_or_not)
                        send_connection_message_or_not = st.session_state.send_connection_message_or_not
                    else:
                        st.session_state.send_connection_message_or_not = decide(send_connection_message_or_not)
                        send_connection_message_or_not = st.session_state.send_connection_message_or_not
                        st.write(send_connection_message_or_not)

                    if send_connection_message_or_not == "y":
                        if 'auto_generated_connection_message' in st.session_state and st.session_state.auto_generated_connection_message == "":
                            auto_generated_connection_message = generate_connection_message(ps, pdn, changes="no changes")
                            st.session_state.auto_generated_connection_message = auto_generated_connection_message
                            connection_message = st.text_area("Auto-generated connection message", st.session_state.auto_generated_connection_message, height=300)
                        else:
                            connection_message = st.text_area("Auto-generated connection message", st.session_state.auto_generated_connection_message, height=300)
                        if st.button("send"):
                            if len(connection_message) > 300:
                                st.warning("Please keep the connection message under 300 characters")
                                st.write("Current length: ", len(connection_message))
                            else:
                                if len(st.session_state.connection_message) <= 300:
                                    st.success("Connection message under 300 characters ✅")
                                    st.session_state.connection_message = connection_message
                                    for linkedin_url in profiles['profileUrl']:
                                        send_connection_request(linkedin_url, message=st.session_state.connection_message)
                    if send_connection_message_or_not == "n":
                        st.write("sending connection requests...")
                        for linkedin_url in profiles['profileUrl']:
                            send_connection_request(linkedin_url, message="")



def outreach_2():
    if 'ran_profile_validator' in st.session_state and st.session_state.ran_profile_validator == "":
        profile_validator()
        st.session_state.ran_profile_validator = "yes"

    if st.session_state.ran_profile_validator == "yes":
        profiles = pd.read_csv("profiles.csv")
        try:
            profiles = profiles[pd.isna(profiles['error'])].reset_index(drop=True)
        except:
            profiles = pd.read_csv("profiles.csv")
        
        display_profile_list(profiles)
        email_to_csv(profiles)
        connect_or_not = st.text_input("Would you like to connect with these people?\n")

        if connect_or_not == "":
            pass
        else:
            if 'connect_or_not' in st.session_state and st.session_state.connect_or_not != "":
                st.write(st.session_state.connect_or_not)
            else:  # Make the API call and store the response in session state
                st.session_state.connect_or_not = decide(connect_or_not)
                connect_or_not = st.session_state.connect_or_not
                st.write(connect_or_not)

            if connect_or_not == "n" or connect_or_not == "N" or connect_or_not == "NO" or connect_or_not == "no":
                profiles_list = profiles
                profiles_list.to_csv("profile_list.csv")
                st.success("profile_list.csv has been created")
                st.download_button(label="Download", data=profiles_list.to_csv(), file_name="profile_list.csv", mime="text/csv")
            else:
                send_connection_message_or_not = st.text_input("Would you like to send a connection message? \n")

                if send_connection_message_or_not == "":
                    pass
                else:
                    if 'send_connection_message_or_not' in st.session_state and st.session_state.send_connection_message_or_not != "":
                        st.write(st.session_state.send_connection_message_or_not)
                        send_connection_message_or_not = st.session_state.send_connection_message_or_not
                    else:
                        st.session_state.send_connection_message_or_not = decide(send_connection_message_or_not)
                        send_connection_message_or_not = st.session_state.send_connection_message_or_not
                        st.write(send_connection_message_or_not)

                    if send_connection_message_or_not == "y":
                        if 'auto_generated_connection_message' in st.session_state and st.session_state.auto_generated_connection_message == "":
                            auto_generated_connection_message = generate_connection_message(ps, pdn, changes="no changes")
                            st.session_state.auto_generated_connection_message = auto_generated_connection_message
                            connection_message = st.text_area("Auto-generated connection message",
                                                             st.session_state.auto_generated_connection_message,
                                                             height=300)
                        else:
                            connection_message = st.text_area("Auto-generated connection message",
                                                             st.session_state.auto_generated_connection_message,
                                                             height=300)
                        if st.button("send"):
                            if len(connection_message) > 300:
                                st.warning("Please keep the connection message under 300 characters")
                                st.write("Current length: ", len(connection_message))
                            else:
                                if len(st.session_state.connection_message) <= 300:
                                    st.success("Connection message under 300 characters ✅")
                                    st.session_state.connection_message = connection_message
                                    for linkedin_url in profiles['profileUrl']:
                                        send_connection_request(linkedin_url, message=st.session_state.connection_message)

                    if send_connection_message_or_not == "n":
                        st.write("sending connection requests...")
                        for linkedin_url in profiles['profileUrl']:
                            send_connection_request(linkedin_url, message="")


  # linkedin + email

def outreach_3():
    if 'ran_profile_validator' in st.session_state and st.session_state.ran_profile_validator == "":
        profile_validator()
        st.session_state.ran_profile_validator = "yes"
    if st.session_state.ran_profile_validator == "yes":

        profiles = pd.read_csv("profiles.csv")
        try:
            profiles = profiles[pd.isna(profiles['error'])].reset_index(drop=True)
        except:
            profiles = pd.read_csv("profiles.csv")
        display_profile_list(profiles)
        email_to_csv(profiles)
        phone_to_csv(profiles)
        connect_or_not = st.text_input("Would you like to connect with these people?\n")
        if connect_or_not == "":
            pass
        else:
            if 'connect_or_not' in st.session_state and st.session_state.connect_or_not != "":
                st.write(st.session_state.connect_or_not)
            else:  # Make the API call and store the response in session state
                st.session_state.connect_or_not = decide(connect_or_not)
                connect_or_not = st.session_state.connect_or_not
                st.write(connect_or_not)

            if connect_or_not == "n" or connect_or_not == "N" or connect_or_not == "NO" or connect_or_not == "no":
                profiles_list = profiles
                profiles_list.to_csv("profile_list.csv")
                st.success("profile_list.csv has been created")
                st.download_button(label="Download", data=profiles_list.to_csv(), file_name="profile_list.csv",
                                   mime="text/csv")
            else:
                send_connection_message_or_not = st.text_input("Would you like to send a connection message? \n")
                if send_connection_message_or_not == "":
                    pass
                else:
                    if 'send_connection_message_or_not' in st.session_state and st.session_state.send_connection_message_or_not != "":
                        st.write(st.session_state.send_connection_message_or_not)
                        send_connection_message_or_not = st.session_state.send_connection_message_or_not
                    else:
                        st.session_state.send_connection_message_or_not = decide(send_connection_message_or_not)
                        send_connection_message_or_not = st.session_state.send_connection_message_or_not
                        st.write(send_connection_message_or_not)

                    if send_connection_message_or_not == "y":
                        if 'auto_generated_connection_message' in st.session_state and st.session_state.auto_generated_connection_message == "":
                            auto_generated_connection_message = generate_connection_message(ps, pdn, changes="no changes")
                            st.session_state.auto_generated_connection_message = auto_generated_connection_message
                            connection_message = st.text_area("Auto-generated connection message",
                                                             st.session_state.auto_generated_connection_message,
                                                             height=300)
                        else:
                            connection_message = st.text_area("Auto-generated connection message",
                                                             st.session_state.auto_generated_connection_message,
                                                             height=300)
                        if st.button("send"):
                            if len(connection_message) > 300:
                                st.warning("Please keep the connection message under 300 characters")
                                st.write("Current length: ", len(connection_message))
                            else:
                                if len(st.session_state.connection_message) <= 300:
                                    st.success("Connection message under 300 characters ✅")
                                    st.session_state.connection_message = connection_message
                                    for linkedin_url in profiles['profileUrl']:
                                        send_connection_request(linkedin_url, message=st.session_state.connection_message)

                    if send_connection_message_or_not == "n":
                        st.write("sending connection requests...")
                        for linkedin_url in profiles['profileUrl']:
                            send_connection_request(linkedin_url, message="")


  # linkedin + email + phone

def outreach_4():
    if 'ran_profile_validator' in st.session_state and st.session_state.ran_profile_validator == "":
        profile_validator()
        st.session_state.ran_profile_validator = "yes"
    if st.session_state.ran_profile_validator == "yes":
        profiles = pd.read_csv("profiles.csv")
        try:
            profiles = profiles[pd.isna(profiles['error'])].reset_index(drop=True)
        except:
            profiles = pd.read_csv("profiles.csv")
        email_to_csv(profiles)
  #email

def outreach_5():
    
    if 'ran_profile_validator' in st.session_state and st.session_state.ran_profile_validator == "":
        profile_validator()
        st.session_state.ran_profile_validator = "yes"
    if st.session_state.ran_profile_validator == "yes":
        profiles = pd.read_csv("profiles.csv")
        try:  
            profiles = profiles[pd.isna(profiles['error'])].reset_index(drop=True)
        except:
            profiles = pd.read_csv("profiles.csv")  
        email_to_csv(profiles)
        phone_to_csv(profiles)
  #email+phone



def generate_email_content(ps,pdn):
    system_content = f"""You are given a problem statement and a description of the product that our company has built to solve the problem.
    We're trying to connect with professioanls via their email to promote our product. Generate a email content to send to them.
    Start every email with "Hi, [name]" , I will add the name later.
    This is my calendly link: [calendly_link]
    End the email with "Thank you".
    """
    
    user_content = f"""Problem Statement: {ps} \nProduct description: {pdn} \nEmail Content: """
    m1 = [{"role": "system", "content": f"{system_content}"},
        {"role": "user", "content": f"{user_content}"}]
    for i in range(0,10):
        try:
            
            result = openai.ChatCompletion.create(
                model="gpt-4",
                max_tokens = 500,
                temperature =0.8,
                messages=m1)
            email_content = result["choices"][0]['message']['content']
            break
        except:
            continue
    return email_content

def generate_email_subject(ps,pdn):
    system_content = f"""You are given a problem statement and a description of the product that our company has built to solve the problem.
    We're trying to connect with professioanls via their email to promote our product. Generate a email subject line to send to them.
    Make sure you keep the subject short and simple.
    """
    user_content = f"""Problem Statement: {ps} \nProduct description: {pdn} \nEmail subject: """
    m1 = [{"role": "system", "content": f"{system_content}"},
        {"role": "user", "content": f"{user_content}"}]
    for i in range(0,10):
        try:
            
            result = openai.ChatCompletion.create(
                model="gpt-4",
                max_tokens = 100,
                temperature =0.8,
                messages=m1)
            email_subject = result["choices"][0]['message']['content']
            break
        except:
            continue
    return email_subject
    
        

def send_emails():
  global mailjet_api_key
  global mailjet_api_secret
  valid_email_list = pd.read_csv("profiles_with_emails.csv")[pd.read_csv("profiles_with_emails.csv")['email'] != "not found"].reset_index(drop=True)
  valid_email_list['name'] = "-"
  sender_email = "ece21735017@matrusri.edu.in"
  sender_name = "Phani Hrushik"
  st.header("Email List")
  for i in range(0, len(valid_email_list)):
    st.write(valid_email_list['email'][i])

  # send_emails_decision = st.text_input("Would you like to send emails to these people?\n")
  send_emails_decision = st.text_input("Would you like to send emails to these people?\n")
  if send_emails_decision == "":
    pass
  else:
    if 'send_emails_decision' in st.session_state and st.session_state.send_emails_decision != "":
      send_emails_decision = st.session_state.send_emails_decision
      st.write(st.session_state.send_emails_decision)
    else:# Make the API call and store the response in session state
      st.session_state.send_emails_decision = decide(send_emails_decision)
      send_emails_decision = st.session_state.send_emails_decision
      st.write(send_emails_decision)
  if str(send_emails_decision.lower()) == "y":
      
      if ps == "" and pdn == "":
        subject = st.text_input("Subject: ")
        email_content = st.text_area("Enter your email content: \n", height = 400)
        if st.button("send"):
            for i in range(len(valid_email_list)):

                receiver_email = str(valid_email_list['email'][i])
                receiver_name = str(valid_email_list['name'][i])
                email_content.replace("[name]", receiver_name)
                from mailjet_rest import Client
                import os
                mailjet_api_key = f"{mailjet_api_key}"
                mailjet_api_secret = f"{mailjet_api_secret}"
                mailjet = Client(auth=(mailjet_api_key, mailjet_api_secret), version='v3.1')
                data = {
                'Messages': [
                        {
                            "From": {
                                "Email": f"{sender_email}",
                                "Name": f"{sender_name}"
                            },
                            "To": [
                                {
                                    "Email": f"{receiver_email}",
                                    "Name": f"{receiver_name}"
                                }
                            ],
                            "Subject": f"{subject}",
                            # "TextPart": f"{email_content}",  because some email clients or recipients may not support HTML content, so the plain text version serves as a fallback that ensures the message can still be read and understood.
                            "HTMLPart": f"<h3>{email_content}</h3>"
                        }
                    ]
                }

                result = mailjet.send.create(data=data)
                st.write("Mail: ", receiver_email)
                if str(result.status_code) == "200":
                    st.write("Success")
      if ps!="" and pdn!="":
          if 'auto_generated_email_subject' in st.session_state and st.session_state.auto_generated_email_subject == "":
              auto_generated_email_subject = generate_email_subject(ps,pdn)
              st.session_state.auto_generated_email_subject = auto_generated_email_subject
              email_subject= st.text_area("Auto-generated email subject", st.session_state.auto_generated_email_subject)
              subject = email_subject
          else:
             email_subject= st.text_area("Auto-generated email subject", st.session_state.auto_generated_email_subject)
             subject = email_subject
             
          
          if 'auto_generated_email_content' in st.session_state and st.session_state.auto_generated_email_content == "":
              auto_generated_email_content = generate_email_content(ps,pdn)
              st.session_state.auto_generated_email_content = auto_generated_email_content
              email_content= st.text_area("Auto-generated email content", st.session_state.auto_generated_email_content, height=400)
          else:
             email_content= st.text_area("Auto-generated email content", st.session_state.auto_generated_email_content, height = 400)
          if st.button("send", key = "send_emails"):
              st.session_state.email_content = email_content
              for i in range(len(valid_email_list)):
                receiver_email = str(valid_email_list['email'][i])
                receiver_name = str(valid_email_list['name'][i])

                from mailjet_rest import Client
                import os
                mailjet_api_key = f"{mailjet_api_key}"
                mailjet_api_secret = f"{mailjet_api_secret}"
                mailjet = Client(auth=(mailjet_api_key, mailjet_api_secret), version='v3.1')
                data = {
                'Messages': [
                        {
                            "From": {
                                "Email": f"{sender_email}",
                                "Name": f"{sender_name}"
                            },
                            "To": [
                                {
                                    "Email": f"{receiver_email}",
                                    "Name": f"{receiver_name}"
                                }
                            ],
                            "Subject": f"{subject}",
                            # "TextPart": f"{email_content}",  because some email clients or recipients may not support HTML content, so the plain text version serves as a fallback that ensures the message can still be read and understood.
                            "HTMLPart": f"<h3>{email_content}</h3>"
                        }
                    ]
                }

                result = mailjet.send.create(data=data)
                st.write("Mail: ", receiver_email)
                if str(result.status_code) == "200":
                    st.write("Success")
              
           #############################################################################################################################
          
      else:
        return



def generate_connection_message(ps,pdn,changes):
  system_content = f"""You are given a problem statement and a description of the product that our company has built to solve the problem.
  We're trying to connect with professioanls on Linkedin to promote our product. Generate a connection message to send to them.
  The message should generate interest for our prodcut.
  Linkedin only allows 300 characters, so keep the message short. Instead of using [name], use #firstName#
  {changes}
  Example:
  Hi #firstName#, 
  <message content>
  Thank you.
  """
  user_content = f"""Problem Statement: {ps} \nProduct description: {pdn} \nConnection Message: """
  m1 = [{"role": "system", "content": f"{system_content}"},
      {"role": "user", "content": f"{user_content}"}]
  for i in range(0,10):
      try:
          
          result = openai.ChatCompletion.create(
            model="gpt-4",
            max_tokens = 100,
            temperature =0.8,
            messages=m1)
          connection_message = result["choices"][0]['message']['content']
          break
      except:
        continue
  return connection_message


def industries_path():
  global user_decided_industries
  user_decided_industries = st.text_input("Great! What kind of industries or companies you have in mind? \n")
  if user_decided_industries == "":
    pass
  else:
    if 'user_decided_industries' in st.session_state and st.session_state.user_decided_industries != "":
      st.write(st.session_state.user_decided_industries)
      user_decided_industries = st.session_state.user_decided_industries
    else:# Make the API call and store the response in session state

      m1 = [{"role": "system", "content": "You will be given a answer from user. You have to figure out what companies the user has mentioned and return it in an array. Just return the array and nothing else."},
          {"role": "user", "content": f"User: {user_decided_industries} \n Companies:"}]
      for i in range(0,10):
        try:
            
            result = openai.ChatCompletion.create(
                model="gpt-4",
                max_tokens = 100,
                temperature =0.8,
                messages=m1)
            user_decided_industries = result["choices"][0]['message']['content']
            break
        except:
           continue
      user_decided_industries = ast.literal_eval(user_decided_industries)
      
      st.session_state.user_decided_industries = user_decided_industries
      user_decided_industries = st.session_state.user_decided_industries
      st.write(user_decided_industries)
  

# def profile_validator():
#     global no_of_profiles
#     global target_persona
#     profiles = pd.read_csv("profiles.csv")
#     os.remove("profiles.csv")
#     profiles['valid_job_title'] = "-"
#     st.write(target_persona)
#     for m in range(0,10):
#       for i in range(0, len(profiles)):
#           system_content = f"""You are given a job title and a list of valid job titles, If the given job title matches any of the valid job titles, 
#           return "y", which means "yes, it's a valid job title". If not, return "n", which means that "no, it's not a valid job title" 
#           Please donot return anything else.
#           Example1:
#           Valid job titles: Data Scientist OR Software developer OR machine learning engineer
#           job title: junior software developer at Apple
#           valid_job_title_or_not: y

#           Example2:
#           Valid job titles: Data Scientist OR Software developer OR machine learning engineer
#           Job title: head of sales at Armani Exchange
#           valid_job_title_or_not: n
#            """
#           user_content = f"""Valid job titles: {target_persona}\nJob title: {profiles['job'][i]}\n Valid_job_title_or_not:"""
#           try:
#             m1 = [{"role": "system", "content": f"{system_content}"},
#                 {"role": "user", "content": f"{user_content}"}]
#             result = openai.ChatCompletion.create(
#                 model="gpt-4",
#                 max_tokens = 1,
#                 temperature =0.8,
#                 messages=m1)
#             profiles['valid_job_title'][i] = result["choices"][0]['message']['content']
        
#             st.write(profiles['name'][i],"---",profiles['job'][i],"---",result["choices"][0]['message']['content'])
#             break
#           except:
#             st.write("failed", m)

#             continue
#     profiles = profiles[profiles['valid_job_title'] == "y"].reset_index(drop=True)
#     if len(profiles) > int(no_of_profiles):
#         profiles = profiles[0:int(no_of_profiles)].reset_index(drop=True)
#         profiles.to_csv("profiles.csv")
#     else:
#         profiles.to_csv("profiles.csv")
def profile_validator():
    profiles = pd.read_csv("profiles.csv")
    os.remove("profiles.csv")
    profiles['valid_job_title'] = "-"
    st.write(target_persona)

    for i in range(0, len(profiles)):
        system_content = f"""You are given a job title and a list of valid job titles, If the given job title matches any of the valid job titles, 
        return "y", which means "yes, it's a valid job title". If not, return "n", which means that "no, it's not a valid job title" 
        Please donot return anything else.
        Example1:
        Valid job titles: Data Scientist OR Software developer OR machine learning engineer
        job title: junior software developer at Apple
        valid_job_title_or_not: y

        Example2:
        Valid job titles: Data Scientist OR Software developer OR machine learning engineer
        Job title: head of sales at Armani Exchange
        valid_job_title_or_not: n
          """
        user_content = f"""Valid job titles: {target_persona}\nJob title: {profiles['job'][i]}\n Valid_job_title_or_not:"""
        for m in range(0, 10):
            try:
                m1 = [{"role": "system", "content": f"{system_content}"},
                      {"role": "user", "content": f"{user_content}"}]

                with st.spinner("Validating..."):
                    result = openai.ChatCompletion.create(
                        model="gpt-4",
                        max_tokens=1,
                        temperature=0.8,
                        messages=m1)
                    
                profiles['valid_job_title'][i] = result["choices"][0]['message']['content']
                break
            except Exception as e:
                print("failed", m, e)
                continue

    profiles = profiles[profiles['valid_job_title'] == "y"].reset_index(drop=True)
    if len(profiles) > int(no_of_profiles):
        profiles = profiles[0:int(no_of_profiles)].reset_index(drop=True)
        profiles.to_csv("profiles.csv")
    else:
        profiles.to_csv("profiles.csv")

def no_industries_path():
    global ps
    global pdn
    global updated_industry_options
    global industry_options
    global company_size_array
    st.write("\nNo worries! Let's find out which industry your product/service works best in!\n")

    # ps = st.text_input("\nWhat problem is your product/service aiming to solve? \n")
    # if ps == "":
    #     pass
    # else:
    #     pdn = st.text_input("\nExplain in detail about how your product works to solve the problem you have mentioned. \n")
    #     if pdn == "":
    #         pass
    #     else:
    if 'industry_options' in st.session_state and st.session_state.industry_options != "":
        industry_options = st.session_state.industry_options
        st.write(industry_options)
    else:
        system_content = """
        You are given a problem statement and a solution that our company has built.
        Your task is to generate 10 company categories or sectors that would be interested in the solution we're offering.
        Only return the company or sector names and nothing else. No explanation is required.
        Arrange the terms in descending order of relevance.
        Follow this format to present your answer.
        Format:
        1. <company category 1>
        2. <company category 2>
        3. <company category 3>
        4. <company category 4>
        5. <company category 5>
        6. <company category 6>
        7. <company category 7>
        8. <company category 8>
        9. <company category 9>
        10. <company category 10>
        """

        user_content = f"Problem statement: {ps} \n product description: {pdn} \n search term:"

        m1 = [{"role": "system", "content": f"{system_content}"},
              {"role": "user", "content": f"{user_content}"}]
        for i in range(0,10):
            try:
                
                result = openai.ChatCompletion.create(
                    model="gpt-4",
                    max_tokens = 100,
                    temperature =0.8,
                    messages=m1)
                response = result["choices"][0]['message']['content']
                break
            except:
                continue
        import re
        text = f"""{response}"""
        pattern = r'\d+\.\s+(.+)'
        industry_options = re.findall(pattern, text)
        st.write("\n\nSuggested Industries: \n")
        st.session_state.industry_options = industry_options
        industry_options = st.session_state.industry_options
        st.write(industry_options)

    changes = st.text_input("Which industries do you choose?\n")
    if changes == "":
        pass
    else:
        if 'updated_industry_options' in st.session_state and st.session_state.updated_industry_options != "":
            st.write(st.session_state.updated_industry_options)
            updated_industry_options = st.session_state.updated_industry_options
        else:
            m1 = [{"role": "system", "content": "You will be given a system-generated answer and changes requested by the user. You have to return the new answer after applying the said changes in the same format you received the data in. It may happen that the user might only choose a single industry or an entirely different industry not present in the given options. Then only return the single industry or the new industry mentioned by the user. No need to include everything."},
                  {"role": "user", "content": f"System-generated response: {industry_options} \n User request: only {changes} \n Changed Answer:"}]
            for i in range(0,10):
                try:
                    
                    result = openai.ChatCompletion.create(
                        model="gpt-4",
                        max_tokens = 100,
                        temperature =0.8,
                        messages=m1)
                    response_after_change = result["choices"][0]['message']['content']
                    break
                except:
                    continue

            updated_industry_options = ast.literal_eval(response_after_change)

            st.session_state.updated_industry_options = updated_industry_options
            updated_industry_options = st.session_state.updated_industry_options
            st.write(updated_industry_options)

        st.write("Select company size: ")
        small_selected = st.checkbox("Small")
        medium_selected = st.checkbox("Medium")
        large_selected = st.checkbox("Large")

        # Create a list to store the selected options
        company_size_array = []
        if small_selected:
            company_size_array.append("small")
        if medium_selected:
            company_size_array.append("medium")
        if large_selected:
            company_size_array.append("large")

        # Display information for each selected option using st.info()
        if "small" in company_size_array:
            st.info("Small: 1-10 employees, 11-50 employees, 51-200 employees")

        if "medium" in company_size_array:
            st.info("Medium: 201-500 employees, 501-1000 employees, 1001-5000 employees")

        if "large" in company_size_array:
            st.info("Large: 5001-10,000 employees, 10,000+ employees")

        location = st.text_input('Enter location: ')
        
        if location == "":
            pass

        if location != "":
            # st.write("updated industry options", updated_industry_options)
            if not os.path.exists("companies.csv"):
                for option in updated_industry_options:
                    linkedin_company_scraper(option, location)
                try:
                    companies = pd.read_csv("companies.csv").dropna(subset=['companyUrl']).reset_index(drop=True)
                    length_profiles = 0
                    for i in range(len(companies)):
                        if int(length_profiles) < (int(no_of_profiles) + 10):
                            company_name = companies['companyName'][i]
                            company_url = companies['companyUrl'][i]
                            linkedin_profile_scraper(company_url, company_name)

                            try:
                                profiles = pd.read_csv("profiles.csv")
                                profiles = profiles[pd.isna(profiles['error'])].reset_index(drop=True)
                            except:
                                profiles = pd.read_csv("profiles.csv")

                            length_profiles = len(profiles)
                        else:
                            break
                except:
                    st.write('No companies found. Try again with different industry/location.')
                    st.session_state.updated_industry_options = ""



# def send_connection_request(linkedin_url, message=""):
#   st.write("connection request sent.")
def send_connection_request(linkedin_url, message=""):
    global phantom_key
    global auto_connect_data_download_link
    global auto_connect_id
    import requests
    import re
    import time
    import pandas as pd
      # Define your variables
    auto_connect_id = f"{auto_connect_id}"
    phantom1_id = auto_connect_id
    phantom_key = f"{phantom_key}"

    # Define the API URL
    url = "https://api.phantombuster.com/api/v2/agents/launch"

    # Define the payload
    payload = {
        "id": phantom1_id,
        "argument": {
            "numberOfAddsPerLaunch": 10,
            "onlySecondCircle": False,
            "dwellTime": True,
            "spreadsheetUrl": linkedin_url,
            "spreadsheetUrlExclusionList": [],
            "sessionCookie": session_cookie,
            "message": message
        }
    }

    # Define headers
    headers = {
        "content-type": "application/json",
        "X-Phantombuster-Key": phantom_key
    }


    # Make a POST request
    response1 = requests.post(url, json=payload, headers=headers)
    text1 = response1.text

    # Extract the container ID
    container_id = re.search(r'"containerId":"(\d+)"', text1).group(1)

    # Define the URL to fetch output
    url = f"https://api.phantombuster.com/api/v2/containers/fetch-output?id={container_id}"

    # Define headers
    headers = {
        "accept": "application/json",
        "X-Phantombuster-Key": phantom_key
    }

    # Display a spinner while waiting
    with st.spinner("Sending Connection Request..."):
        m = 0
        while m < 1200:
            response2 = requests.get(url, headers=headers)
            text = response2.text
            if text != '{"output":null}':
                break
            time.sleep(20)
            m += 1
    
    wget.download(f"{auto_connect_data_download_link}", "connection_status.csv")
    
    connection_status = pd.read_csv("connection_status.csv")
    try:
        st.info(connection_status['error'][0] +f": {linkedin_url}")
    except:
        st.success("Connection request sent: " + linkedin_url)
    os.remove("connection_status.csv")



import os
import streamlit as st
import pandas as pd

def email_to_csv(profiles):
    if not os.path.exists("profiles_with_emails.csv"):
        profiles['email'] = "-"
        profiles['score'] = "-"
        with st.spinner("Finding email and phone numbers..."):
            for i in range(0, len(profiles)):
                first_name = profiles['firstName'][i]
                last_name = profiles['lastName'][i]
                company_name = profiles['company'][i]
                try:
                    email, score = email_finder(first_name, last_name, company_name)
                    profiles['email'][i] = email
                    profiles['score'][i] = score
                except:
                    profiles['email'][i] = "not found"
                    profiles['score'][i] = "not found"

        profiles.to_csv("profiles_with_emails.csv")

        st.success("profiles_with_emails.csv has been created!")
        st.download_button(label="Download", data=profiles.to_csv(), file_name="profile_with_emails.csv", mime="text/csv")
    else:
        profiles = pd.read_csv("profiles_with_emails.csv")
        st.success("profiles_with_emails.csv has been created!")
        st.download_button(label="Download", data=profiles.to_csv(), file_name="profile_with_emails.csv", mime="text/csv")






def phone_to_csv(profiles):
  if not os.path.exists("profiles_with_email_phone.csv"):
    profiles['phone_number_1'] = "-"
    profiles['phone_number_2'] = "-"
    profiles['phone_number_3'] = "-"

    for i in range(0, len(profiles)):
        email = profiles['email'][i]
        if email == "none" or email == "not found":
            phone_number = "not found"
        else:
            phone_number = phone_number_finder(f"{email}")
            # phone_number = str(phone_number).replace("[", "").replace("]", "").replace("'", "")
        try:
          if phone_number[0]!= "n":
            profiles['phone_number_1'][i] = phone_number[0]
          else:
            profiles['phone_number_1'][i] = "-"
        except:
          continue
        try:
          if phone_number[1]!= "o":
            profiles['phone_number_2'][i] = phone_number[1]
          else:
            profiles['phone_number_1'][i] = "-"
        except:
          continue
        try:
          if phone_number[2]!="t":
            profiles['phone_number_3'][i] = phone_number[2]
          else:
            profiles['phone_number_1'][i] = "-"
        except:
          continue

    profiles.to_csv("profiles_with_email_phone.csv")

    st.success("profiles_with_email_phone.csv has been created!")
    st.download_button(label="Download",data=profiles.to_csv(),file_name="profiles_with_email_phone.csv",mime="text/csv")
  else:
    profiles = pd.read_csv("profiles_with_email_phone.csv")
    st.success("profiles_with_email_phone.csv has been created!")
    st.download_button(label="Download",data=profiles.to_csv(),file_name="profile_with_email_phone.csv",mime="text/csv")

# combined_data_companies = pd.DataFrame()  # Initialize outside the function

import streamlit as st

def linkedin_company_scraper(industry_name, country):
    global search_export_data_download_link
    global phantom_key
    global combined_data_companies
    global search_export_id
    search_export_id = f"{search_export_id}"
    phantom_key = f"{phantom_key}" # Make sure to define this

    url = "https://api.phantombuster.com/api/v2/agents/launch"
    payload = {
        "id": f"{search_export_id}",
        "argument": f"""{{
            "connectionDegreesToScrape": [
                "2",
                "3+"
            ],
            "category": "Companies",
            "numberOfLinesPerLaunch": 10,
            "sessionCookie": "{session_cookie}",
            "search": "{search_link_generator(country, industry_name, company_size_array)}",
            "numberOfResultsPerSearch": 50
        }}"""
    }
    headers = {
        "content-type": "application/json",
        "X-Phantombuster-Key": f"{phantom_key}"
    }

    response1 = requests.post(url, json=payload, headers=headers)
    text1 = response1.text
    container_id = re.search(r'"containerId":"(\d+)"', text1).group(1)
    url = f"https://api.phantombuster.com/api/v2/containers/fetch-output?id={container_id}"
    headers = {
        "accept": "application/json",
        "X-Phantombuster-Key": f"{phantom_key}"
    }
    response2 = requests.get(url, headers=headers)
    text = response2.text

    with st.spinner("Scraping companies..."):
        m = 0
        for m in range(0, 1200):
            if text == '{"output":null}':
                time.sleep(20)
                url = f"https://api.phantombuster.com/api/v2/containers/fetch-output?id={container_id}"
                headers = {
                    "accept": "application/json",
                    "X-Phantombuster-Key": f"{phantom_key}"
                }
                response2 = requests.get(url, headers=headers)
                text = response2.text
                m = m + 1
            else:
                break

    st.success("Scraping companies complete!")
    wget.download(f"{search_export_data_download_link}", "company.csv")

# Assuming you have imported the necessary libraries and defined session_cookie, search_link_generator, and other variables elsewhere in your code.


    csv_data = pd.read_csv("company.csv")

    # Append the CSV data to the combined DataFrame
    # combined_data_companies = combined_data_companies.append(csv_data, ignore_index=True)
    combined_data_companies = pd.concat([combined_data_companies, csv_data], ignore_index=True)
    combined_data_companies.to_csv("companies.csv")

    os.remove("company.csv")
    return combined_data_companies
    


import streamlit as st

# Assuming you have imported the necessary libraries and defined session_cookie, target_persona, and other variables elsewhere in your code.

# combined_data_profiles = pd.DataFrame()

def linkedin_profile_scraper(company_url, company_name):
    global combined_data_profiles
    global phantom_key
    global employee_export_id
    global employee_export_data_download_link
    url = "https://api.phantombuster.com/api/v2/agents/launch"
    employee_export_id = f"{employee_export_id}"
    phantom_key = f"{phantom_key}"

    payload = {
        "id": f"{employee_export_id}",
        "argument": f"""{{
            "numberOfCompaniesPerLaunch": 10,
            "spreadsheetUrl": "{company_url}",
            "sessionCookie": "{session_cookie}",
            "numberOfResultsPerCompany": 6,
            "positionFilter": "{target_persona}"
        }}"""
    }

    headers = {
        "content-type": "application/json",
        "X-Phantombuster-Key": f"{phantom_key}"
    }

    response1 = requests.post(url, json=payload, headers=headers)
    text1 = response1.text
    container_id = re.search(r'"containerId":"(\d+)"', text1).group(1)
    url = f"https://api.phantombuster.com/api/v2/containers/fetch-output?id={container_id}"

    headers = {
        "accept": "application/json",
        "X-Phantombuster-Key": f"{phantom_key}"
    }

    response2 = requests.get(url, headers=headers)
    text = response2.text

    with st.spinner("Scraping profiles..."):
        m = 0
        for m in range(0, 1200):
            if text == '{"output":null}':
                time.sleep(20)
                url = f"https://api.phantombuster.com/api/v2/containers/fetch-output?id={container_id}"

                headers = {
                    "accept": "application/json",
                    "X-Phantombuster-Key": f"{phantom_key}"
                }

                response2 = requests.get(url, headers=headers)
                text = response2.text
                m = m + 1
            else:
                break

    # st.success("Scraping employees complete!")
    wget.download(f"{employee_export_data_download_link}", "profile.csv")


    csv_data = pd.read_csv("profile.csv")
    csv_data['company'] = company_name

    # Append the CSV data to the combined DataFrame
    # combined_data_profiles = combined_data_profiles.append(csv_data, ignore_index=True)
    combined_data_profiles = pd.concat([combined_data_profiles, csv_data], ignore_index=True)


    combined_data_profiles.to_csv("profiles.csv")
    os.remove("profile.csv")






##########################################




# reset()
import os
import pandas as pd
import streamlit as st


global combined_data_companies
global combined_data_profiles
combined_data_companies = pd.DataFrame()
combined_data_profiles = pd.DataFrame()

global target_persona
global no_of_profiles
global outreach_method
global session_cookie
global profiles
global location


# st.session_state.target_persona = ""
# target_persona = st.text_input("Who is your target persona? \n")
# if target_persona == "":
#   pass
# else:
#   target_persona = target_persona_decider(target_persona)
#   st.session_state.target_persona = target_persona
#   st.write(target_persona)
selected_option = st.sidebar.selectbox("Select an option", ["Home", "Send Connection Requests", "Send Emails"])
if selected_option == "Send Connection Requests":
  st.title("Send Connection Requests")
  # Create an upload button for CSV files
  uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
  # Display the uploaded file details
  if uploaded_file is not None:
      # Display the file details
      st.header("Uploaded File Details:")
      file_details = {"Filename": uploaded_file.name, "Filesize": uploaded_file.size}
      st.write(file_details)

      # Read the CSV file into a DataFrame
      df = pd.read_csv(uploaded_file)

      # Display the DataFrame
      st.header("CSV File Contents:")
      st.dataframe(df)
      

      session_cookie = st.text_input("Enter your session cookie: ", type = "password")
      if session_cookie == "":
        pass
      else:
        ps_or_pdn = st.text_input("Do you have a problem statement and product description? \n")
        if ps_or_pdn == "":
          pass
        else:
          ps_or_pdn = decide(ps_or_pdn)
        if ps_or_pdn == "n":
                if 'auto_generated_connection_message' in st.session_state and st.session_state.auto_generated_connection_message == "":
                  auto_generated_connection_message = generate_connection_message(ps,pdn, changes="no changes")
                  st.session_state.auto_generated_connection_message = auto_generated_connection_message
                  connection_message= st.text_area("Auto-generated connection message", st.session_state.auto_generated_connection_message, height = 300)
                else:
                    connection_message= st.text_area("Auto-generated connection message", st.session_state.auto_generated_connection_message, height = 300)
                if st.button("send"):
                  if len(connection_message) > 300:
                    st.warning("Please keep the connection message under 300 characters")
                    st.write("Current length: ", len(connection_message))
                  else:
                    if len(st.session_state.connection_message) <= 300:
                      st.success("Connection message under 300 characters ✅")
                      st.session_state.connection_message = connection_message
                      for linkedin_url in df['profileUrl']:
                          send_connection_request(linkedin_url, message= st.session_state.connection_message)  
        if ps_or_pdn == "y":
          ps = st.text_input("Enter your problem statement: ")
          pdn = st.text_input("Enter your product description: ")
          if pdn == "":
            pass
          else:
            if 'auto_generated_connection_message' in st.session_state and st.session_state.auto_generated_connection_message == "":
              auto_generated_connection_message = generate_connection_message(ps,pdn, changes="include problem statement and product description")
              st.session_state.auto_generated_connection_message = auto_generated_connection_message
              connection_message= st.text_area("Auto-generated connection message", st.session_state.auto_generated_connection_message, height = 300)
              
            else:
                connection_message= st.text_area("Auto-generated connection message", st.session_state.auto_generated_connection_message, height = 300)
            if st.button("send"):
                st.session_state.connection_message = connection_message
                if len(st.session_state.connection_message) > 300:
                  st.warning("Please keep the connection message under 300 characters")
                  st.write("Current length: ", len(connection_message))
                else:
                  if len(st.session_state.connection_message) <= 300:
                    st.write("Connection message under 300 charadcters ✅")

                    st.write("Connection message: ", st.session_state.connection_message)
                    for linkedin_url in df['profileUrl']:
                        send_connection_request(linkedin_url, message= st.session_state.connection_message)  
      
if selected_option == "Send Emails":
    st.title("Send Emails")
  
  # Create an upload button for CSV files
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    # Display the uploaded file details
    if uploaded_file is not None:
        # Display the file details
        st.header("Uploaded File Details:")
        file_details = {"Filename": uploaded_file.name, "Filesize": uploaded_file.size}
        st.write(file_details)

        # Read the CSV file into a DataFrame
        df = pd.read_csv(uploaded_file)

        # Display the DataFrame
        st.header("CSV File Contents:")
        st.dataframe(df)
        valid_email_list = df[df['email'] != "not found"].reset_index(drop=True)
    
        for email in valid_email_list['email']:
          st.write(email)
        ps_or_pdn = st.text_input("Do you have a problem statement and product description? \n")
        if ps_or_pdn == "":
          pass
        else:
          ps_or_pdn = decide(ps_or_pdn)
        if ps_or_pdn == "n":
            subject = st.text_input("Subject: ")
            email_content = st.text_area("Enter your email content: \n", height = 400)
          
            if st.button("send"):

              for i in range(len(valid_email_list)):
                  sender_email = "ece21735017@matrusri.edu.in"
                  sender_name = "Phani Hrushik"
                  receiver_email = str(valid_email_list['email'][i])
                  receiver_name = str(valid_email_list['name'][i])
                  email_content.replace("[name]", receiver_name)
                  from mailjet_rest import Client
                  import os
                  mailjet_api_key = f"{mailjet_api_key}"
                  mailjet_api_secret = f"{mailjet_api_secret}"
                  mailjet = Client(auth=(mailjet_api_key, mailjet_api_secret), version='v3.1')
                  data = {
                  'Messages': [
                          {
                              "From": {
                                  "Email": f"{sender_email}",
                                  "Name": f"{sender_name}"
                              },
                              "To": [
                                  {
                                      "Email": f"{receiver_email}",
                                      "Name": f"{receiver_name}"
                                  }
                              ],
                              "Subject": f"{subject}",
                              # "TextPart": f"{email_content}",  because some email clients or recipients may not support HTML content, so the plain text version serves as a fallback that ensures the message can still be read and understood.
                              "HTMLPart": f"<h3>{email_content}</h3>"
                          }
                      ]
                  }

                  result = mailjet.send.create(data=data)
                  st.write("Mail: ", receiver_email)
                  if str(result.status_code) == "200":
                      st.write("Success")
        if ps_or_pdn == "y":
          ps = st.text_input("Enter your problem statement: ")
          pdn = st.text_input("Enter your product description: ")
          if pdn=="":
            pass
          else:
              if 'auto_generated_email_subject' in st.session_state and st.session_state.auto_generated_email_subject == "":
                  auto_generated_email_subject = generate_email_subject(ps,pdn)
                  st.session_state.auto_generated_email_subject = auto_generated_email_subject
                  email_subject= st.text_area("Auto-generated email subject", st.session_state.auto_generated_email_subject)
                  subject = email_subject
              else:
                email_subject= st.text_area("Auto-generated email subject", st.session_state.auto_generated_email_subject)
                subject = email_subject
                
              
              if 'auto_generated_email_content' in st.session_state and st.session_state.auto_generated_email_content == "":
                  auto_generated_email_content = generate_email_content(ps,pdn)
                  st.session_state.auto_generated_email_content = auto_generated_email_content
                  email_content= st.text_area("Auto-generated email content", st.session_state.auto_generated_email_content, height=400)
              else:
                email_content= st.text_area("Auto-generated email content", st.session_state.auto_generated_email_content, height = 400)
              if st.button("send", key = "send_emails"):
                  st.session_state.email_content = email_content
                  for i in range(len(valid_email_list)):
                    receiver_email = str(valid_email_list['email'][i])
                    receiver_name = str(valid_email_list['name'][i])

                    from mailjet_rest import Client
                    import os
                    mailjet_api_key = f"{mailjet_api_key}"
                    mailjet_api_secret = f"{mailjet_api_secret}"
                    mailjet = Client(auth=(mailjet_api_key, mailjet_api_secret), version='v3.1')
                    data = {
                    'Messages': [
                            {
                                "From": {
                                    "Email": f"{sender_email}",
                                    "Name": f"{sender_name}"
                                },
                                "To": [
                                    {
                                        "Email": f"{receiver_email}",
                                        "Name": f"{receiver_name}"
                                    }
                                ],
                                "Subject": f"{subject}",
                                # "TextPart": f"{email_content}",  because some email clients or recipients may not support HTML content, so the plain text version serves as a fallback that ensures the message can still be read and understood.
                                "HTMLPart": f"<h3>{email_content}</h3>"
                            }
                        ]
                    }

                    result = mailjet.send.create(data=data)
                    st.write("Mail: ", receiver_email)
                    if str(result.status_code) == "200":
                        st.write("Success")
        else:
          pass
        
  
        
  

if selected_option == "Home":
    if st.button("reset"):
        reset()
    st.title("Linkedin Agent")
    
    if 'ran_profile_validator' not in st.session_state:
        st.session_state.ran_profile_validator = ""

    # Initialize session state
    if 'target_persona' not in st.session_state:
        st.session_state.target_persona = ""
    if 'no_of_profiles' not in st.session_state:
        st.session_state.no_of_profiles = ""

    if 'industries' not in st.session_state:
        st.session_state.industries = ""
        
    if 'connect_or_not' not in st.session_state:
        st.session_state.connect_or_not = ""

    if 'user_decided_industries' not in st.session_state:
        st.session_state.user_decided_industries = ""
        
    if 'send_emails_decision' not in st.session_state:
        st.session_state.send_emails_decision = ""

    if 'industry_options' not in st.session_state:
        st.session_state.industry_options = ""

    if 'updated_industry_options' not in st.session_state:
        st.session_state.updated_industry_options = ""
        
    if 'auto_generated_connection_message' not in st.session_state:
        st.session_state.auto_generated_connection_message = ""

    if 'connection_message' not in st.session_state:
        st.session_state.connection_message = ""

    if 'auto_generated_email_content' not in st.session_state:
        st.session_state.auto_generated_email_content = ""

    if 'email_content' not in st.session_state:
        st.session_state.email_content = ""

    if 'auto_generated_email_subject' not in st.session_state:
        st.session_state.auto_generated_email_subject = ""
    if  'email_subject' not in st.session_state:
        st.session_state.email_subject = ""
    
    ps = st.text_input("Enter your problem statement")
    if ps == "":
        pass
    if ps!="":
        pdn = st.text_input("Enter your product description")
        if pdn == "":
            pass
        else:
            if 'generated_target_profiles' in st.session_state and st.session_state.generated_target_profiles != "":
                st.write("\n\nAutogenerated target profiles: \n")
                generated_target_profiles = st.session_state.generated_target_profiles
                
                st.write(generated_target_profiles)
            else:
                   system_content = f"""You will be given a problem statement and a description of the product or service our company has built to solve the problem.
                    Your task is to figure out which profiles you should target to sell the product or service to.
                    The profiles should be specific enough to be actionable. For example, "data scientist" is not a specific enough profile, because they have no say in the purchasing process.
                    On the other hand, "VP of Engineering" is too specific, because there are not enough of them to target.
                    Return the target profiles in an list. Make sure to limit it to only six profiles.
                    follow this format only:
                    format: 
                    1. profile1 job title
                    2. profile2 job title
                    3. profile3 job title
                    4. profile4 job title
                    5. pofile5 job title
                    6. profile6 job title
                    Only name their job title
                    """
                   user_content = f"""Problem Statement: {ps} \nProduct description: {pdn} \ntarget profiles: """
                   m1 = [{"role": "system", "content": f"{system_content}"},
                        {"role": "user", "content": f"{user_content}"}]
                   for i in range(0,10):
                       try:
                           
                        result = openai.ChatCompletion.create(
                                model="gpt-4",
                                max_tokens = 100,
                                temperature =0.8,
                                messages=m1)
                        generated_target_profiles = result["choices"][0]['message']['content']
                        break
                       except:
                           continue
                   text = f"""{generated_target_profiles}"""
                   pattern = r'\d+\.\s+(.+)'
                   generated_target_profiles = re.findall(pattern, text)
                   st.write("\n\nAutogenerated target profiles: \n")
                   st.session_state.generated_target_profiles = generated_target_profiles
                   generated_target_profiles =  st.session_state.generated_target_profiles
                   st.write(generated_target_profiles)
            
            

            target_persona = st.text_input("Choose target persona or enter your own: \n")
            if target_persona == "":
                pass
            else:
                if 'target_persona' in st.session_state and st.session_state.target_persona != "":
                    target_persona = st.session_state.target_persona
                    st.write(st.session_state.target_persona)
                else:
                    # Make the API call and store the response in session state
                    st.session_state.target_persona = target_persona_decider(target_persona)
                    target_persona = st.session_state.target_persona
                    st.write(target_persona)

                no_of_profiles = st.text_input("\nGot it! How many people would you like to target? \n")
                if no_of_profiles == "":
                    pass
                else:
                    if 'no_of_profiles' in st.session_state and st.session_state.no_of_profiles != "":
                        st.write(st.session_state.no_of_profiles)
                    else:
                        # Make the API call and store the response in session state
                        st.session_state.no_of_profiles = decide_number_of_profiles(no_of_profiles)
                        no_of_profiles = st.session_state.no_of_profiles
                        st.write(no_of_profiles)

                    outreach_method = st.write("""
                    Please choose one of the following outreach methods:
                    1. LinkedIn Only: Send a connection request on LinkedIn, either with or without a personalized message.
                    2. LinkedIn + Email: Send LinkedIn connection requests and obtain/send emails, which will be provided in a .csv file.
                    3. LinkedIn + Email + Phone: Send LinkedIn connection requests, obtain/send emails, and collect phone numbers, all of which will be provided in a .csv file.
                    4. Email Only: Send emails exclusively or request a .csv file containing email addresses.
                    5. Email + Phone: Collect and send emails, as well as gather phone numbers, with all data being provided in a .csv file.
                    """)

                    outreach_method = st.radio("Pick one", ["None", "Linkedin Only", "Linkedin + Email", "Linkedin + Email + Phone", "Email Only", "Email + Phone"])

                    outreach_methods_dict = {
                        "None": "None",
                        "Linkedin Only": "1",
                        "Linkedin + Email": "2",
                        "Linkedin + Email + Phone": "3",
                        "Email Only": "4",
                        "Email + Phone": "5"
                    }

                    if outreach_method != "None":
                        # st.write("You selected:", outreach_methods_dict[outreach_method])
                        outreach_method = outreach_methods_dict[outreach_method]
                        session_cookie = st.text_input("\nEnter your session cookie \n", type = "password")
                        if session_cookie == "":
                            pass
                        elif len(session_cookie) < 152 and session_cookie != "":
                            st.write("Please enter a valid session cookie")
                        else:
                            industries = st.text_input("\nDo you have any industries or companies in mind? \n")
                            if industries == "":
                                pass
                            else:
                                if 'industries' in st.session_state and st.session_state.industries != "":
                                    # st.write(st.session_state.industries)
                                    industries = st.session_state.industries
                                else:
                                    # Make the API call and store the response in session state
                                    st.session_state.industries = decide(industries)
                                    industries = st.session_state.industries
                                    st.write(industries)

                                if industries == "y" or industries == "Y" or industries == "Yes" or industries == "YES":
                                    industries_path()
                                    st.write("Select company size: ")
                                    small_selected = st.checkbox("Small")
                                    medium_selected = st.checkbox("Medium")
                                    large_selected = st.checkbox("Large")

                                    # Create a list to store the selected options
                                    company_size_array = []
                                    if small_selected:
                                        company_size_array.append("small")
                                    if medium_selected:
                                        company_size_array.append("medium")
                                    if large_selected:
                                        company_size_array.append("large")

                                    # Display information for each selected option using st.info()
                                    if "small" in company_size_array:
                                        st.info("Small: 1-10 employees, 11-50 employees, 51-200 employees")

                                    if "medium" in company_size_array:
                                        st.info("Medium: 201-500 employees, 501-1000 employees, 1001-5000 employees")

                                    if "large" in company_size_array:
                                        st.info("Large: 5001-10,000 employees, 10,000+ employees")

                                    location = st.text_input('Enter location: ')

                                    if location == "":
                                        pass

                                    if location != "":
                                        if not os.path.exists("companies.csv"):
                                            for option in user_decided_industries:
                                                linkedin_company_scraper(option, location)
                                                try:
                                                    companies = pd.read_csv("companies.csv").dropna(subset=['companyUrl']).reset_index(drop=True)
                                                    length_profiles = 0
                                                    for i in range(len(companies)):
                                                        if int(length_profiles) < (int(no_of_profiles) + 10):
                                                            company_name = companies['companyName'][i]
                                                            company_url = companies['companyUrl'][i]
                                                            linkedin_profile_scraper(company_url, company_name)
                                                            try:
                                                                profiles = pd.read_csv("profiles.csv")
                                                                profiles = profiles[pd.isna(profiles['error'])].reset_index(drop=True)
                                                            except:
                                                                profiles = pd.read_csv("profiles.csv")
                                                            length_profiles = len(profiles)
                                                        else:
                                                            break
                                                except:
                                                    pass
                                        if os.path.exists("companies.csv") and (len(pd.read_csv("companies.csv")) == 1):
                                            st.write("No results found. Please try a different industry/location")
                                            os.remove("comapnies.csv")
                                        if os.path.exists("companies.csv") and (len(pd.read_csv("companies.csv")) > 1):
                                            st.write("outreach method", outreach_method)
                                            profiles = pd.read_csv("profiles.csv")
                                            if len(profiles) != 0:
                                                if outreach_method == "1":
                                                    outreach_1()
                                                elif outreach_method == "2":
                                                    outreach_2()
                                                    send_emails()
                                                elif outreach_method == "3":
                                                    outreach_3()
                                                    send_emails()
                                                elif outreach_method == "4":
                                                    outreach_4()
                                                    send_emails()
                                                elif outreach_method == '5':
                                                    outreach_5()
                                                    send_emails()
                                            else:
                                                st.write("\nNo profiles found. They might be out of your network. \nPlease choose a different industry/company and try again.\n")
                                                # try_again(industry = True)
                                                os.remove("companies.csv")
                                                os.remove("profiles.csv")
                                                st.session_state.updated_industry_options = ""

                                if industries == "n" or industries == "N" or industries == "No" or industries == "NO":
                                    no_industries_path()
                                    if os.path.exists("companies.csv"):
                                        st.write("outreach method", outreach_method)
                                        if os.path.exists("profiles.csv"):
                                            profiles = pd.read_csv("profiles.csv")
                                            try:
                                                profiles = profiles[pd.isna(profiles['error'])].reset_index(drop=True)
                                            except:
                                                profiles = pd.read_csv("profiles.csv")

                                            if len(profiles) != 0:
                                                if outreach_method == "1":
                                                    outreach_1()
                                                elif outreach_method == "2":
                                                    outreach_2()
                                                    send_emails()
                                                elif outreach_method == "3":
                                                    outreach_3()
                                                    send_emails()
                                                elif outreach_method == "4":
                                                    outreach_4()
                                                    send_emails()
                                                elif outreach_method == '5':
                                                    outreach_5()
                                                    send_emails()
                                            else:
                                                st.write("\nNo profiles found. They might be out of your network. \nPlease choose a different industry/company and try again.\n")
                                                os.remove("profiles.csv")
                                                os.remove("companies.csv")
                                                st.session_state.updated_industry_options = ""
                                                # try_again(industry = False)
                                        else:
                                            st.write("No companies found. Please try a different industry/location")
                                            os.remove("companies.csv")
                                            st.session_state.updated_industry_options = ""

                # Create a sidebar with navigation options



