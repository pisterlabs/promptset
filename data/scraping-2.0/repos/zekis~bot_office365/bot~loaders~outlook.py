import sys
import traceback
import bot_config
import os
from datetime import datetime
from typing import Any, Dict, Optional, Type

sys.path.append("/root/projects")
import common.bot_logging
from common.bot_comms import publish_email_card, publish_list, publish_draft_card, publish_draft_forward_card, send_to_me, publish_event_card, send_to_user, send_to_another_bot, publish_error
from common.bot_utils import tool_description, tool_error, sanitize_subject
#from common.card_factories import create_list_card

from O365 import Account

from langchain.callbacks.manager import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
from langchain.tools import BaseTool
from langchain.agents import initialize_agent, load_tools, Tool
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI

from bs4 import BeautifulSoup

from langchain.chat_models import ChatOpenAI

from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

#common.bot_logging.bot_logger = common.bot_logging.logging.getLogger('ToolLogger')
#common.bot_logging.bot_logger.addHandler(common.bot_logging.file_handler)
example_email_format = """

<body>
    <div>
        <h2>Automated Response</h2>
        <p>Hi [Recipient's Name],</p>
        <p>Email Body</p>
        <p></p>
        <p>Signature</p>
    </div>
</body>

"""

def authenticate():
    
    credentials = (bot_config.APP_ID, bot_config.APP_PASSWORD)
    account = Account(credentials,auth_flow_type='credentials',tenant_id=bot_config.TENANT_ID, main_resource=bot_config.OFFICE_USER)
    account.authenticate()
    return account

# This function returns a summary of the given email using OpenAI's GPT-3 API.
def get_email_summary(email, body_soup):

    str_to_address = ""
    if email.to:
        for to_address in email.to:
            str_to_address = str_to_address + to_address.address + ", "


    try:
        os.environ["OPENAI_API_KEY"] = bot_config.OPENAI_API_KEY
        chat = ChatOpenAI(temperature=0, model_name=bot_config.TOOL_AI)
        query = f"""Provide a summary of the latest response in this email chain (chain is ordered newest to oldest, newest at the top), and conversation history (below), into two sections, ignoring capability statements and confidentiality disclaimers or anything after the signature for the following email
        To: {str_to_address}, From: {email.sender.address}, Subject: {email.subject}, Date: {email.received.strftime('%Y-%m-%d %H:%M:%S')}, Body: {body_soup}"""

        common.bot_logging.bot_logger.debug(f"Query: {query}")
        email_response = chat([HumanMessage(content=query)]).content
        return email_response
    except Exception as e:
        #traceback.print_exc()
        tb = traceback.format_exc()
        return tool_error(e, tb)


def reply_to_email_summary(summary, sender, example=None, comments=None, previous_draft=None):
    os.environ["OPENAI_API_KEY"] = bot_config.OPENAI_API_KEY
    llm = ChatOpenAI(temperature=0, model_name=bot_config.TOOL_AI, verbose=bot_config.VERBOSE)
    query = f"""Given this email summary:  {summary}, 
    
    please create a reasonable reply from 'Chad the AI Assistant' on behalf of {bot_config.OFFICE_USER} to {sender}.
    Response is include an informal 'To' salutation and opening line at the start and add a signature from 'Chad the AI Assistant'
    Response is to be HTMl formatted with simlary tags and styling {example_email_format}

    """
    if example:
        query += f": {example}"
    if comments:
        query += f"Consider the following comments: {comments}"
    if previous_draft:
        query += f"Based on the previous draft: {previous_draft}"
    print(query)
    common.bot_logging.bot_logger.debug(f"Query: {query}")
    email_response = llm.predict(query)
    return email_response

def forward_email_summary(summary, example=None, comments=None, previous_draft=None):
    os.environ["OPENAI_API_KEY"] = bot_config.OPENAI_API_KEY
    llm = ChatOpenAI(temperature=0, model_name=bot_config.TOOL_AI, verbose=bot_config.VERBOSE)
    query = f"""Given this email summary: {summary}, 
    
    please create a reasonable email from 'Chad the AI Assistant' on behalf of {bot_config.OFFICE_USER} on behalf of {bot_config.OFFICE_USER}.
    Response is include an informal 'To' salutation and opening line at the start and add a signature from 'Chad the AI Assistant'
    Response is to be HTMl formatted with simlary tags and styling {example_email_format}

    """
    if example:
        query += f"example: {example}"
    if comments:
        query += f"Consider the following comments: {comments}"
    if previous_draft:
        query += f"Based on the previous draft: {previous_draft}"

    common.bot_logging.bot_logger.debug(f"Query: {query}")
    email_response = llm.predict(query)
    return email_response

def modify_draft(body, comments, previous_draft=None):
    os.environ["OPENAI_API_KEY"] = bot_config.OPENAI_API_KEY
    llm = ChatOpenAI(temperature=0, model_name=bot_config.TOOL_AI, verbose=bot_config.VERBOSE)
    query = f"""Given this request: {body}, 
    
    
    please create a reasonable email from 'Chad the AI Assistant' on behalf of {bot_config.OFFICE_USER} on behalf of {bot_config.OFFICE_USER}.
    Response is include an informal 'To' salutation and opening line at the start and add a signature from 'Chad the AI Assistant'
    Response is to be HTMl formatted with simlary tags and styling {example_email_format}
    """
    if comments:
        query += f"Consider the following comments: {comments}"
    if previous_draft:
        query += f"Based on the previous draft: {previous_draft}"

    common.bot_logging.bot_logger.debug(f"Query: {query}")
    email_response = llm.predict(query)
    return email_response

def get_conversation_sorted(ConversationID):
    account = authenticate()
    mailbox = account.mailbox()
    inbox = mailbox.inbox_folder()

    query = inbox.new_query().on_attribute("receivedDateTime").greater(datetime(2023, 1, 1)).chain("and").on_attribute('conversationid').equals(ConversationID)
    common.bot_logging.bot_logger.debug(f"Query: {query}")
    returned_emails = inbox.get_messages(limit=5,query=query, order_by="receivedDateTime desc")
    
    count = 0
    if returned_emails:
        emails = list(returned_emails)
        return emails[0]
    return None

def get_conversation(ConversationID):
    account = authenticate()
    mailbox = account.mailbox()
    inbox = mailbox.inbox_folder()

    query = inbox.new_query().on_attribute('conversationid').equals(ConversationID)
    common.bot_logging.bot_logger.debug(f"Query: {query}")
    returned_emails = inbox.get_messages(limit=1,query=query)
    
    count = 0
    if returned_emails:
        emails = list(returned_emails)
        return emails[0]
    return None

# This function takes an `ObjectID` as input and returns the email associated with that ID.
def get_message(ObjectID):
    common.bot_logging.bot_logger.debug(f"ObjectID: {ObjectID}")
    account = authenticate()
    mailbox = account.mailbox()
    inbox = mailbox.inbox_folder()
    # Fetches a single email matching the given `ObjectID` from the inbox.
    returned_email = inbox.get_message(ObjectID)
    return returned_email

def search_emails_return_unique_conv(search_query):
    clean_search_query = search_query.replace('"', "")
    common.bot_logging.bot_logger.debug(f"Query: {search_query}")
    account = authenticate()
    mailbox = account.mailbox()
    inbox = mailbox.inbox_folder()

    query = inbox.new_query().search(clean_search_query)
    emails = inbox.get_messages(limit=15, query=query)

    count = 0
    if emails:
        final_response = []
        conversation_ids = set()  # Using a set to keep track of unique conversation_ids
        for email in emails:
            if email.conversation_id not in conversation_ids:  # Check if conversation_id is unique
                conversation_ids.add(email.conversation_id)  # Add the unique conversation_id to the set
                final_response.append(format_email_header(email))  # Only append if conversation_id is unique
        return final_response
    return None
    

def search_emails(search_query):
    account = authenticate()
    mailbox = account.mailbox()
    inbox = mailbox.inbox_folder()

    query = inbox.new_query().search(search_query)
    emails = inbox.get_messages(limit=5, query=query)

    count = 0
    if emails:
        return emails
    return None

def create_email_reply(ConversationID, body, save=False):
    common.bot_logging.bot_logger.debug(f"Conversation ID: {ConversationID} | Body: {body}")
    account = authenticate()
    mailbox = account.mailbox()
    inbox = mailbox.inbox_folder()

    query = inbox.new_query().on_attribute('conversationid').equals(ConversationID)
    emails = list(inbox.get_messages(limit=1, query=query))

    #Get first email
    email = emails[0]
    email.mark_as_read()
    reply_msg = email.reply()
    reply_msg.body = body

    if save:
        reply_msg.save_draft()
    return reply_msg

def create_email_forward(ConversationID, recipient, body, save=False):
    common.bot_logging.bot_logger.debug(f"Conversation ID: {ConversationID} | Recipient: {recipient} | Body: {body}")
    account = authenticate()
    mailbox = account.mailbox()
    inbox = mailbox.inbox_folder()

    query = inbox.new_query().on_attribute('conversationid').equals(ConversationID)
    emails = list(inbox.get_messages(limit=1, query=query))
    email = emails[0]
    

    email.mark_as_read()
    reply_msg = email.reply()
    reply_msg.to.clear()
    reply_msg.cc.clear()
    reply_msg.to.add(recipient)
    reply_msg.body = body
    
    if save:
        reply_msg.save_draft()

    return reply_msg


def draft_email(recipient, subject, body, user_improvements=None, previous_draft=None, save=True):
    common.bot_logging.bot_logger.debug(f"Recipient: {recipient} | Subject: {subject}")
    account = authenticate()
    mailbox = account.mailbox()
    inbox = mailbox.inbox_folder()

    if body or body != "":

        if user_improvements:
            body = modify_draft(body, user_improvements)
            if previous_draft:
                body = modify_draft(body, user_improvements, previous_draft)

        message = mailbox.new_message()
        message.to.add(recipient)
        message.subject = subject
        
        message.body = body

        if save:
            message.save_draft()
        
        return message
    

def clean_html(html):
    remove_strings = [
        "SG Controls - Capability Statement",
        "SG Controls - Case Studies",
        "SG Controls - Technical Services",
        "SG Controls Pty Ltd is ISO 9001 Quality certified, safety aware and environmentally conscious.",
        "This email contains material, which may be confidential, legally privileged, and/or the subject of copyright.",
        "If you are not an intended recipient, please advise the sender and delete it.",
        "Confidentiality and privilege are not waived.",
        "The views or opinions expressed in this email may be the sender",
        "own and not necessarily shared / authorised by SG Controls Pty Ltd.",
        "No liability for loss or damage resulting from your receipt of / dealing with this email is accepted.",
        "INTERNAL EMAIL: This email originated from inside the SG Controls network.",
        "CAUTION: This email originated from outside of the organisation. Do not click links or open attachments unless you recognise the sender and know the content is safe."]
    soup = BeautifulSoup(html, 'html.parser')
    # Remove unnecessary tags
    for tag in soup(['style', 'script', 'img']):
        tag.decompose()
    clean_text = soup.get_text()
    for s in remove_strings:
        clean_text = clean_text.replace(s, '')
    #finally truncate the message to avoid token errors
    clean_text = clean_text[:8000]
    return clean_text

def format_email(email):

    str_to_address, str_cc_address, str_bcc_address = email_contacts_to_string(email, 10)
    
    if email.received is None:
        received_date = "Not sent"
    else:
        received_date = email.received.strftime('%Y-%m-%d %H:%M:%S')


    email_s = f"""```
To: {str_to_address}
CC: {str_cc_address}
BCC: {str_bcc_address}
From: {email.sender.address}
Subject: {email.subject}
Date: {received_date}
Body: {email.body}
```
"""
    return email_s

def format_email_summary_only(email, summary):

    str_to_address, str_cc_address, str_bcc_address = email_contacts_to_string(email, 10)

    if email.received is None:
        received_date = "Not sent"
    else:
        received_date = email.received.strftime('%Y-%m-%d %H:%M:%S')

    email_s = f"""```
To: {str_to_address}
CC: {str_cc_address}
BCC: {str_bcc_address}
From: {email.sender.address}
Subject: {email.subject}
Date: {received_date}
Body(Summarised): {summary}
```
"""
    return email_s


def email_contacts_to_string(email, max_contacts):
    def get_string_from_list(address_list, max_contacts):
        if len(address_list) > max_contacts:
            return ", ".join([addr.address for addr in address_list[:max_contacts]]) + ", ..."
        else:
            return ", ".join([addr.address for addr in address_list])

    str_to_address = get_string_from_list(email.to, max_contacts)
    str_cc_address = get_string_from_list(email.cc, max_contacts)
    str_bcc_address = get_string_from_list(email.bcc, max_contacts)

    return str_to_address, str_cc_address, str_bcc_address


def format_email_header(email):
    header = { 'object_id': email.object_id, 'conversationid': email.conversation_id, 'subject': email.subject, 'from': email.sender.address }
    return header

def scheduler_check_emails():
    current_date_time = datetime.utcnow().strftime('%Y-%m-%d')
    query = f"isread:no received:{current_date_time}"
    
    #print(query)
    emails = search_emails(query)
    
    if emails:
        for email in emails:
            domain = email.sender.address.split('@')[1]
            if not email.is_event_message:
                if domain not in bot_config.IGNORE_DOMAINS:
                    summary = get_email_summary(email, clean_html(email.body))
                    
                    
                    #Determine Type and Intent of Email
                    review_email(email, summary)
                else:
                    send_to_user(f"Ignoring email from {email.sender.address} as domain is in the ignore list")
                #Always mark email read, or we end up reviewing the same email forever.
                email.mark_as_read()
                
            else:
                publish_event_card("New Event", email.get_event())
                send_to_another_bot('journal',f"Please add to my journal that I recieved an invite from {email.sender.address} about {email.subject} ")
                email.mark_as_read()
    return None

def review_email(email, summary):
    #summary = get_email_summary(email, clean_html(email.body))
    #publish_email_card("Email", email, summary)
    #publish a task question back to itself
    
    #email.mark_as_read()
    type_prompt = ""
    email_type = get_email_type(email, summary)
    if email_type:
        
        if  "DIRECT" in email_type:
            type_prompt = f"I have determined that this is email is direct to you"
            intent = get_email_intent(email, summary)
            #next_action = task_reply_or_ignore(email, summary)
        elif "OTHER" in email_type:
            type_prompt = f"I was not able to determine the type of email."
            intent = get_email_intent(email, summary)
            #next_action = task_reply_or_ignore(email, summary)
        
        else:
            send_to_user(f"I have determined that this email about {email.subject} from  {email.sender.address} is a {email_type} email and will mark it as read and ignore it.")
            return
    else:
        return
    publish_email_card("You Have Received A New Email!", email, summary)

    ai_summary = format_email_summary_only(email, summary)
    

    str_to_address = email.sender.address
    intent_prompt = f"I have also determined {intent}."
    link = f"https://outlook.office.com/mail/inbox/id/{email.object_id}"

    send_to_another_bot('journal',f"Please add to my journal a brief note that I recieved an {intent} email from {str_to_address} about {summary} ")

    task_action_prompt = f"Please use the CREATE_TASK tool to create a task in the 'Tasks' folder with the subject {email.subject}, body to include steps to complete and url {link}, for {bot_config.FRIENDLY_NAME} to action."
    email_action_prompt = f"Given this email I just received from {str_to_address}, Please use the DRAFT_REPLY_TO_EMAIL tool using ConverstationID: {email.conversation_id} to draft a reply in HTML format to {str_to_address} from 'Chad the AI Assistant' on behalf of {bot_config.FRIENDLY_NAME} with helpfull tips and add a signature from 'Chad the AI Assistant'. Email Received: {ai_summary}"
    #common.bot_logging.bot_logger.debug("task_action_prompt: " + task_action_prompt)
    #common.bot_logging.bot_logger.debug("email_action_prompt: " + email_action_prompt)
    common.bot_logging.bot_logger.debug(intent)
    
    if "INQUIRY/QUESTION" in intent:
        """ Emails where the sender is seeking information or clarification on a specific topic. """
        auto_reply(f"{type_prompt} {intent_prompt} I think I could respond to this inquery/question for you", email_action_prompt)
        

    elif "PERMISSION/ACCESS" in intent:
        """ The sender is requesting access to a server or service. """
        send_to_user(f"{type_prompt} {intent_prompt} I think the next thing to do would be to create a reminder about this request for you")
        send_to_me(task_action_prompt)
        # do something

    elif "FEEDBACK/OPINION" in intent:
        """ Comments or feedback about a product, service, or event. """
        auto_reply(f"{type_prompt} {intent_prompt} I think I could respond to this feedback/opinion for you", email_action_prompt)
    
        # do something

    elif "COMPLAINT/PROBLEM" in intent:
        """ When something has gone wrong or the sender is dissatisfied. """
        auto_reply(f"{type_prompt} {intent_prompt} I think I could respond to this complaint/problem for you", email_action_prompt)
        # do something

    elif "REQUEST FOR ASSISTANCE" in intent:
        """ Support requests or help with a product or service. """
        auto_reply(f"{type_prompt} {intent_prompt} I think I could respond to this request for assistance for you", email_action_prompt)
        # do something

    elif "ORDER/TRANSACTION" in intent:
        """ Emails related to the purchase, delivery, or refund of a product or service. """
        auto_reply(f"{type_prompt} {intent_prompt} I think I could respond to this order/transaction email for you", email_action_prompt)
        # do something

    elif "APPOINTMENT/MEETING" in intent:
        """ Setting, changing, or canceling a meeting or appointment. """
        # do something

    elif "THANK YOU/APPRECIATION" in intent:
        """ Expressions of gratitude or appreciation. """
        # do something

    elif "INTRODUCTION/NETWORKING" in intent:
        """ Where the sender is introducing themselves or looking to establish a connection. """
        auto_reply(f"{type_prompt} {intent_prompt} I think I could respond to this introduction/networking email for you", email_action_prompt)
        # do something

    elif "NEWS/UPDATE" in intent:
        """ Information about new products, services, events, or updates. """
        # do something

    elif "SUBSCRIPTION" in intent:
        """ Requests or confirmations about subscribing or unsubscribing from a service or mailing list. """
        send_to_user(f"{type_prompt} {intent_prompt} I think the next thing to do would be to create a reminder about this subscription for you")
        send_to_me(task_action_prompt)

        # do something

    elif "RECOMMENDATION/REFERRAL" in intent:
        """ Where someone is recommending a person, product, or service. """
        auto_reply(f"{type_prompt} {intent_prompt} I think I could respond to this recommendation/referral for you", email_action_prompt)
        # do something

    elif "PROMOTION/ADVERTISEMENT" in intent:
        """ Unsolicited promotional material or offers. """
        # do something

    elif "CONFIRMATION" in intent:
        """ Confirming actions, purchases, sign-ups, or other activities. """
        send_to_user(f"{type_prompt} {intent_prompt} I think the next thing to do would be to create a reminder about this confirmation for you")
        send_to_me(task_action_prompt)
        # do something

    elif "REMINDER" in intent:
        """ Reminder for events, payments, or other actions. """
        send_to_user(f"{type_prompt} {intent_prompt} I think the next thing to do would be to create a reminder for you")
        send_to_me(task_action_prompt)
        # do something

    elif "LEGAL/OFFICIAL" in intent:
        """ Communications regarding contracts, official notices, or other legal matters. """
        send_to_user(f"{type_prompt} {intent_prompt} I think the next thing to do would be to create a reminder to respond to this official/legal issue")
        send_to_me(task_action_prompt)
        # do something

    elif "SOCIAL/PERSONAL" in intent:
        """ Informal communications, greetings, or personal updates. """
        auto_reply(f"{type_prompt} {intent_prompt} I think I could respond to this social/personal email for you", email_action_prompt)
        # do something

    elif "SPAM" in intent:
        """ Unwanted or suspicious emails. """
        # do something

def auto_reply(message, email_action_prompt):
    if bot_config.AUTO_DRAFT_REPLY.lower() in 'yes,true,enable':
        send_to_user(message)
        send_to_me(email_action_prompt)



# def task_reply_or_ignore(email, summary):
#     #this function uses ai to determine the next course of action for the user
#     ai_summary = format_email_summary_only(email, summary)

#     str_to_address = ""
#     for to_address in email.to:
#         str_to_address = str_to_address + to_address.address + ", "

#     try:
#         os.environ["OPENAI_API_KEY"] = bot_config.OPENAI_API_KEY
#         llm = ChatOpenAI(temperature=0, model_name=bot_config.TOOL_AI, verbose=bot_config.VERBOSE)
#         tools = load_tools(["human"], llm=llm)

#         available_actions = f"""name: CREATE_TASK description: if a email reply wont solve the issue, use this tool that creates a reminder for {bot_config.FRIENDLY_NAME} to action at a later date. 
#         name: DRAFT_REPLY_TO_EMAIL description: if more information is needed or we can suggest basic troubleshooting tips and advise, use this too that generates a helpfull response with instructions and helpfull tips to resolve the issue. 
#         name: ARCHIVE description: email doesnt require any action from {bot_config.FRIENDLY_NAME} and can be ignored."""
        
#         prompt = f"""Given the following email to {bot_config.FRIENDLY_NAME} and reviewing the latest response, identify which action to perform next. return only the action name
#         actions: {available_actions}
        
#         email: {ai_summary}"""
#         common.bot_logging.bot_logger.debug(prompt)

#         agent_executor = initialize_agent(tools, llm, agent=AgentType.OPENAI_FUNCTIONS, verbose=True, agent_kwargs = {
#                 "input_variables": ["input", "agent_scratchpad"]
#             })

#         return agent_executor.run(input=prompt).upper()
#     except Exception as e:
#         #traceback.print_exc()
#         tb = traceback.format_exc()
#         publish_error(e, tb)
#         return tool_error(e, tb)
        
def get_email_type(email, summary):
    #this function uses ai to determine the next course of action for the user
    ai_summary = format_email_summary_only(email, summary)

    str_to_address = ""
    for to_address in email.to:
        str_to_address = str_to_address + to_address.address + ", "

    try:
        os.environ["OPENAI_API_KEY"] = bot_config.OPENAI_API_KEY
        llm = ChatOpenAI(temperature=0, model_name=bot_config.TOOL_AI, verbose=bot_config.VERBOSE)
        tools = load_tools(["human"], llm=llm)

        available_types = f"""name: DONOTREPLY description: the email has been sent from an address indicating that its from a do not reply address, 
        name: COPIED description: {bot_config.OFFICE_USER} was CCd or BCCd and not sent directly to {bot_config.OFFICE_USER}, 
        name: MICROSOFT description: the email has been sent from microsoft.
        name: DIRECT description: email sent directly to {bot_config.OFFICE_USER} and not from microsoft, a do not reply address.
        name: OTHER description: Unable to determine"""
        
        prompt = f"""Helping {bot_config.OFFICE_USER} you are given the following email and its your task to check the from and to address to determine if this email is one of the following types,  return only the type name
        Types: {available_types}
        
        email: {ai_summary}"""
        common.bot_logging.bot_logger.debug(prompt)

        agent_executor = initialize_agent(tools, llm, agent=AgentType.OPENAI_FUNCTIONS, verbose=True, agent_kwargs = {
                "input_variables": ["input", "agent_scratchpad"]
            })

        return agent_executor.run(input=prompt).upper()
    except Exception as e:
        #traceback.print_exc()
        tb = traceback.format_exc()
        publish_error(e, tb)
        return tool_error(e, tb)

def get_email_intent(email, summary):
    #this function uses ai to determine the next course of action for the user
    ai_summary = format_email_summary_only(email, summary)

    str_to_address = ""
    for to_address in email.to:
        str_to_address = str_to_address + to_address.address + ", "

    try:
        os.environ["OPENAI_API_KEY"] = bot_config.OPENAI_API_KEY
        llm = ChatOpenAI(temperature=0, model_name=bot_config.TOOL_AI, verbose=bot_config.VERBOSE)
        tools = load_tools(["human"], llm=llm)
        available_intents = """
name: INQUIRY/QUESTION
description: Emails where the sender is seeking information or clarification on a specific topic.

name: PERMISSION/ACCESS
description: Email where the sender is requesting access to a server or service.

name: FEEDBACK/OPINION
description: Comments or feedback about a product, service, or event.

name: COMPLAINT/PROBLEM
description: When something has gone wrong or the sender is dissatisfied.

name: REQUEST FOR ASSISTANCE
description: Support requests or help with a product or service.

name: ORDER/TRANSACTION
description: Emails related to the purchase, delivery, or refund of a product or service.

name: APPOINTMENT/MEETING
description: Setting, changing, or canceling a meeting or appointment.

name: THANK YOU/APPRECIATION
description: Expressions of gratitude or appreciation.

name: INTRODUCTION/NETWORKING
description: Where the sender is introducing themselves or looking to establish a connection.

name: NEWS/UPDATE
description: Information about new products, services, events, or updates.

name: SUBSCRIPTION
description: Requests or confirmations about subscribing or unsubscribing from a service or mailing list.

name: RECOMMENDATION/REFERRAL
description: Where someone is recommending a person, product, or service.

name: PROMOTION/ADVERTISEMENT
description: Unsolicited promotional material or offers.

name: CONFIRMATION
description: Confirming actions, purchases, sign-ups, or other activities.

name: REMINDER
description: Reminder for events, payments, or other actions.

name: LEGAL/OFFICIAL
description: Communications regarding contracts, official notices, or other legal matters.

name: SOCIAL/PERSONAL
description: Informal communications, greetings, or personal updates.

name: SPAM
description: Unwanted or suspicious emails.
"""

        
        
        prompt = f"""Given the following email to {bot_config.FRIENDLY_NAME}, its your task to determine the intent from the list available,  return only the intent name
        Intents: {available_intents}
        Ensure to only return only the intent name

        email: {ai_summary}"""
        common.bot_logging.bot_logger.debug(prompt)

        agent_executor = initialize_agent(tools, llm, agent=AgentType.OPENAI_FUNCTIONS, verbose=True, agent_kwargs = {
                "input_variables": ["input", "agent_scratchpad"]
            })

        return agent_executor.run(input=prompt).upper()
    except Exception as e:
        #traceback.print_exc()
        tb = traceback.format_exc()
        publish_error(e, tb)
        return tool_error(e, tb)

class MSSearchEmailsId(BaseTool):
    parameters = []
    optional_parameters = []
    name = "SEARCH_EMAILS"
    summary = """useful for when you need to search through emails and get their content. This tool only returns 15 emails maximum."""
    parameters.append({"name": "query", "description": "query must use the Keyword Query Language (KQL) syntax. Example query: subject:Meeting or from:Dan AND received:2023-05-19..2023-05-20. Do not enclose strings in quotes"})
    description = tool_description(name, summary, parameters, optional_parameters)
    return_direct = False

    def _run(self, query: str, publish: str = "True", run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        try:
            ai_summary = ""
            human_summary = []

            emails = search_emails_return_unique_conv(query)
            if emails:
                for email in emails:
                    ai_summary = ai_summary + " - Email - From: " + email['from'] + ", Subject: " + email['subject'] + ", EmailID: " + email['object_id'] + ", ConversatonID: " + email['conversationid'] + "\n"
                    title = email['from'] + ": " + email['subject']
                    value = "Please use the GET_EMAIL_CHAIN using EmailID: " + email['object_id'] + " and create_task: False"
                    human_summary.append((title, value))

                if publish.lower() == "true":
                    publish_list(f"Choose an option:", human_summary)
                    self.return_direct = True
                    #return bot_config.PROMPT_PUBLISH_TRUE
                else:
                    self.return_direct = False
                    return ai_summary
            else:
                return "No emails found"

        except Exception as e:
            #traceback.print_exc()
            tb = traceback.format_exc()
            publish_error(e, tb)
            return tool_error(e, tb, self.description)
    
    async def _arun(self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("SEARCH_EMAILS_RETURN_IDS does not support async")

class MSGetEmailDetail(BaseTool):
    parameters = []
    optional_parameters = []
    name = "GET_EMAIL_CHAIN"
    summary = """useful for when you need to get the email content for a single email or email chain."""
    parameters.append({"name": "EmailID", "description": "Retrieve the Email ID by using the SEARCH_EMAILS tool" })
    parameters.append({"name": "ConversationID", "description": "Retrieve the conversation IDs by using the SEARCH_EMAILS tool" })
    #parameters.append({"name": "follow_up", "description": "The tool will consider if any action is required" })
    description = tool_description(name, summary, parameters, optional_parameters)
    return_direct = False

    def _run(self, EmailID: str = None, ConversationID: str = None, follow_up: bool = True, publish: str = "True", run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        try:
            if EmailID is not None:
                #response = get_email(EmailID)
                email = get_message(EmailID)

            if ConversationID is not None:
                #response = get_email_chain(ConversationID)
                email = get_conversation_sorted(ConversationID)

            if email:
                summary = get_email_summary(email, clean_html(email.body))
                ai_summary = format_email_summary_only(email, summary)
                #review_email(email)
                

                #send_to_bot(bot_config.USER_ID,"Only If the following email requires " + bot_config.FRIENDLY_NAME + " to perform an action such as reply, send a file, fix a problem, complete work etc, then use CREATE_TASK to create a new task in the Tasks folder. Email: " + ai_summary)
                if publish.lower() == "true":
                    publish_email_card("Email Review", email, summary)
                    self.return_direct = True
                    return None
                else:
                    self.return_direct = False
                    return ai_summary
        
            return "No emails"

        except Exception as e:
            #traceback.print_exc()
            tb = traceback.format_exc()
            publish_error(e, tb)
            return tool_error(e, tb, self.description)
    
    async def _arun(self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("GET_EMAIL_CHAIN does not support async")


class MSDraftEmail(BaseTool):
    parameters = []
    optional_parameters = []
    name = "DRAFT_EMAIL"
    summary = """useful for when you need to create a draft new email."""
    parameters.append({"name": "recipient", "description": "A valid email address" })
    parameters.append({"name": "subject", "description": "Email subject" })
    parameters.append({"name": "body", "description": "HTML formated email body" })
    optional_parameters.append({"name": "user_improvements", "description": "Direct the draft email and can be used in combination with the previous_draft" })
    optional_parameters.append({"name": "previous_draft", "description": "To be used with the user_improvements" })
    description = tool_description(name, summary, parameters, optional_parameters)
    return_direct = True

    def _run(self, recipient: str, subject: str, body: str, user_improvements: str = None, previous_draft: str = None, publish: str = "True", run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        try:

            publish = "true"
            response = draft_email(recipient, subject, body, user_improvements, previous_draft)
            ai_summary = format_email(response)

            if publish.lower() == "true":
                publish_draft_card("New Draft Email", response, body, reply=False)
                response.delete()
                self.return_direct = True
                return None
            else:
                response.delete()
                self.return_direct = False
                return ai_summary

        except Exception as e:
            #traceback.print_exc()
            self.return_direct = False
            tb = traceback.format_exc()
            publish_error(e, tb)
            return tool_error(e, tb, self.description)
    
    async def _arun(self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("DRAFT_EMAIL does not support async")

class MSSendEmail(BaseTool):
    parameters = []
    optional_parameters = []
    name = "SEND_EMAIL"
    summary = """useful for when you need to send an email."""
    parameters.append({"name": "recipient", "description": "A valid email address" })
    parameters.append({"name": "subject", "description": "Email subject" })
    parameters.append({"name": "body", "description": "HTML formated email body" })
    description = tool_description(name, summary, parameters, optional_parameters)
    return_direct = False

    def _run(self, recipient: str, subject: str, body: str, publish: str = "True", run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        try:
            response = draft_email(recipient, subject, body, save=True)
            if publish.lower() == "true":
                send_to_user("A draft email is saved - Please manually send from outlook")
                send_to_another_bot('journal',f'Please add to my journal that I created a draft email to {recipient} about {subject}')
                self.return_direct = True
                return None
            else:
                self.return_direct = False
                return "AI is not allowed to email directly, A draft email is saved - Please manually send from outlook"

        except Exception as e:
            #traceback.print_exc()
            tb = traceback.format_exc()
            publish_error(e, tb)
            return tool_error(e, tb, self.description)
    
    async def _arun(self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("SEND_EMAIL does not support async")

class MSReplyToEmail(BaseTool):
    parameters = []
    optional_parameters = []
    name = "REPLY_TO_EMAIL"
    summary = """useful for when you need to send a reply to an existing email chain."""
    parameters.append({"name": "ConversationID", "description": "A valid email conversation ID" })
    parameters.append({"name": "body", "description": "HTML formatted email body" })
    description = tool_description(name, summary, parameters, optional_parameters)
    return_direct = False

    def _run(self, ConversationID: str, body: str, publish: str = "True", run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        try:
            response = create_email_reply(ConversationID, body, True)
            if publish.lower() == "true":
                send_to_user("A draft email is saved - Please manually send from outlook")
                #publish_draft_card("New Draft Email", response, body, reply=False)
                self.return_direct = True
                return None
            else:
                self.return_direct = False
                return "AI is not allowed to email directly, A draft email is saved - Please manually send from outlook"

        except Exception as e:
            #traceback.print_exc()
            tb = traceback.format_exc()
            publish_error(e, tb)
            return tool_error(e, tb, self.description)
    
    async def _arun(self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("REPLY_TO_EMAIL does not support async")

class MSForwardEmail(BaseTool):
    parameters = []
    optional_parameters = []
    name = "FORWARD_EMAIL"
    summary = """useful for when you need to create a forward email to an existing email chain."""
    parameters.append({"name": "ConversationID", "description": "A valid email conversation ID" })
    parameters.append({"name": "body", "description": "HTML formated email body" })
    parameters.append({"name": "recipient", "description": "email addresses" })
    description = tool_description(name, summary, parameters, optional_parameters)
    return_direct = False

    def _run(self, ConversationID: str, body: str, recipient: str, publish: str = "True", run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        try:
            forward_email = create_email_forward(ConversationID, recipient, body, True)
            if publish.lower() == "true":
                send_to_user("A draft email is saved - Please manually send from outlook")
                self.return_direct = True
                return None
            else:
                self.return_direct = False
                return "AI is not allowed to email directly, A draft email is saved - Please manually send from outlook"

        except Exception as e:
            #traceback.print_exc()
            tb = traceback.format_exc()
            publish_error(e, tb)
            return tool_error(e, tb, self.description)
    
    async def _arun(self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("FORWARD_EMAIL does not support async")

class MSDraftForwardEmail(BaseTool):
    parameters = []
    optional_parameters = []
    name = "DRAFT_FORWARD_TO_EMAIL"
    summary = "useful for when you need to generate a forward email to an existing email chain."
    parameters.append({"name": "ConversationID", "description": "A valid email conversation ID" })
    parameters.append({"name": "recipient", "description": "A valid email address" })
    optional_parameters.append({"name": "example", "description": "example or template to give to the AI to help generate the new email draft" })
    optional_parameters.append({"name": "previous_draft", "description": "If available, the previous generated draft attempt may be provided, To be used with the user_improvements" })
    optional_parameters.append({"name": "previous_draft_improvements", "description": "If a previous draft is provided, this parameter can be used to direct the AI to address issues with the previous draft email" })
    description = tool_description(name, summary, parameters, optional_parameters)
    return_direct = False

    def _run(self, ConversationID: str, recipients: str, example: str = None, previous_draft: str = None, previous_draft_improvements: str = None, publish: str = "True", run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        try:
            email_chain = get_conversation_sorted(ConversationID)
            summary = get_email_summary(clean_html(email_chain.body))

            email_response = forward_email_summary(summary, example, previous_draft_improvements, previous_draft)
            forward_email = create_email_forward(ConversationID, recipients, email_response, False)

            ai_summary = format_email_summary_only(forward_email, email_response)
            #force publish true
            publish = "True"
            if publish.lower() == "true":
                publish_draft_forward_card("New Forward Draft Email", forward_email, email_response)
                forward_email.delete()
                self.return_direct = True
                return None
            else:
                forward_email.delete()
                self.return_direct = False
                return ai_summary
        except Exception as e:
            #traceback.print_exc()
            tb = traceback.format_exc()
            publish_error(e, tb)
            return tool_error(e, tb, self.description)
    
    async def _arun(self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("DRAFT_FORWARD_TO_EMAIL does not support async")

class MSDraftReplyToEmail(BaseTool):
    parameters = []
    optional_parameters = []
    name = "DRAFT_REPLY_TO_EMAIL"
    summary = """useful for when you need to generate a reply to an existing email chain."""
    parameters.append({"name": "ConversationID", "description": "A valid email conversation ID" })
    optional_parameters.append({"name": "example", "description": "example or template to give to the AI to help generate the new email draft" })
    optional_parameters.append({"name": "previous_draft", "description": "If available, the previous generated draft attempt may be provided, To be used with the user_improvements" })
    optional_parameters.append({"name": "previous_draft_improvements", "description": "If a previous draft is provided, this parameter can be used to direct the AI to address issues with the previous draft email" })
    description = tool_description(name, summary, parameters, optional_parameters)
    return_direct = False

    def _run(self, ConversationID: str, example:str = None, previous_draft: str = None, previous_draft_improvements: str = None, publish: str = "True", run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        try:
            email_chain = get_conversation_sorted(ConversationID)
            summary = get_email_summary(email_chain, clean_html(email_chain.body))
            #summary = format_email_summary_only(email_chain, clean_html(email_chain.body))

            email_response = reply_to_email_summary(summary, email_chain.sender.address, example, previous_draft_improvements, previous_draft)

            reply_email = create_email_reply(ConversationID, email_response)

            ai_summary = format_email_summary_only(reply_email, email_response)
            #force publish true
            publish = "True"
            if publish.lower() == "true":
                publish_draft_card("New Draft Email", reply_email, email_response, True)
                reply_email.delete()
                self.return_direct = True
                return None
            else:
                reply_email.delete()
                self.return_direct = False
                return ai_summary

        except Exception as e:
            #traceback.print_exc()
            tb = traceback.format_exc()
            publish_error(e, tb)
            return tool_error(e, tb, self.description)
    
    async def _arun(self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("DRAFT_REPLY_TO_EMAIL does not support async")
