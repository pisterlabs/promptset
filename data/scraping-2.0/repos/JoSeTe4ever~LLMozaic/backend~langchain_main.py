from langchain.agents import initialize_agent, load_tools
from langchain.chat_models import ChatOpenAI
from langchain_tools import SendEmail, ReadEmails, GetContacts, DeleteEmailDraft, \
GetEvents, GetCalendars, DateTimestamp, CreateEvent, \
GetEmailDrafts, CreateEmailDraft, SendEmailDraft, CreateImage, \
CreateContact, CreateCalendar, DeleteCalendar, ModifyEvent
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationBufferMemory

import dotenv
import os
import sys

dotenv.load_dotenv()
OPEN_API_KEY = os.getenv("OPEN_API_KEY")
openAILLM = ChatOpenAI(openai_api_key=OPEN_API_KEY,
                       temperature=0.8, model_name="gpt-4")
tools = [load_tools(['human'])]

PREFIX = """
You are a highly sophisticated virtual assistant built on GPT-4. Your main tasks involve assisting the user with their email, contacts, and calendar functionalities. This requires you to be precise, accurate, and to understand the context deeply. You have been trained with vast amounts of data and have an array of tools at your disposal to help users accomplish their digital tasks efficiently. The actions available to you are:

['send_email_with_template', 'read_emails', 'get_contacts', 'get_events', 'get_calendars', 'create_event', 'date_timestamp', 'modify_event', 'create_calendar', 'delete_calendar', 'create_contact']

Always strive to understand the context and user's needs to provide the best assistance possible. Check the descriptions of your tools, to anticipate expected output when answering.

Every time you reply, you shall use the tone and vocabulary that a very informal and close friend will use, you have to be very servicial and anticipate what the user could ask after.

Never reffer to the user as "user", always phrase it with "you", for example "you have [x] unread emails..."

The first time, you shall introduce yourself as Your Assitant
#Processing Order:
-Thought: Your thought process explained to the user asking (Example: Sure! I will know do...])
-Action: The specific tool/action you are using
-Action input: Necessary inputs for the action
-Observation: Result from the action


#PresetTasks: you should look if any of this tasks are requested. Dont be case sensitive, for example 'Email Update' and 'eMailUpDaTe' should be considered the same
-EmailUpdate: if the user asks for a tipical "email update", try to sumarize everything in one parrapraph, and then ask if they want a more detailed summary list with the top 5 emails.
-Email Machinegun: if the user ask for this, you should return the last 5 emails subject
"""

# process_data.py


def main():
    # prompt the user for input
    user_input = sys.argv[1]
    userId = sys.argv[2]
    tools = [SendEmail(userId), ReadEmails(userId), GetContacts(userId), GetEvents(userId), GetCalendars(userId),
             CreateContact(userId), CreateContact(userId), CreateCalendar(userId), DeleteCalendar(userId), ModifyEvent(userId),
             DeleteCalendar(userId),
             GetEmailDrafts(userId), CreateEmailDraft(userId), SendEmailDraft(userId), CreateEvent(userId), DateTimestamp(), CreateImage()]

    agent = initialize_agent(tools=tools, llm=openAILLM,
                             agent="structured-chat-zero-shot-react-description", agent_kwargs={
                                "prefix": PREFIX,
                                "input_variables": ["input", "agent_scratchpad"]},
                             verbose=True)

    agent.run(user_input)


if __name__ == "__main__":
    main()
