from __future__ import print_function
from __future__ import annotations

import os
import re

import json

from dotenv import load_dotenv

from mautrix.client import Client
from mautrix.types import (EventType, Membership, MessageEvent, MessageType,
                           StrippedStateEvent, TextMessageEventContent, UserID, Format)


from yarl import URL
import asyncio

from datetime import datetime

import logging

#own imports
import OpenAIConnector
import MailConnector

BOT_HELP_MESSAGE = f"""
            Help Message:
                prefix: !
                commands:
                    help:
                        command: help, ?, h
                        description: display help command
                    mail:
                        command: mail
                        description: get mails
                        command has to be followed by:
                            count: number of mails to get
                            positive: list of positive labels eg. ['INBOX', 'UNREAD']
                            negative: list of negative labels eg. ['READ']
                            !mail count: 5 positive: ['INBOX', 'UNREAD'] negative: ['READ']
                    model:
                        command: model
                        description: change model
            """

load_dotenv()

logging.basicConfig(level=logging.INFO)

class MailBot:

    # Matrix credentials
    user_id: UserID = os.getenv('MATRIX_USER_ID')
    base_url: URL = URL(os.getenv('MATRIX_BASE_URL'))
    token: str = os.getenv('MATRIX_TOKEN')
    api_token: str = os.getenv('OPENAI_API_KEY')

    time_to_message: str = os.getenv('TIME_TO_MESSAGE')
    check_every_minutes: int = int(os.getenv('CHECK_EVERY_MINUTES'))

    importance_to_answer: int = int(os.getenv('IMPORTANCE_THRESHOLD'))
    how_many_mails: int = int(os.getenv('HOW_MANY_EMAILS_TO_SUMMARIZE'))

    labels_to_summarize: str = json.loads(os.getenv('LABELS_TO_SUMMARIZE'))
    labels_to_ignore: str = json.loads(os.getenv('LABELS_TO_IGNORE'))

    labels_to_summarize_everyx_minutes = json.loads(os.getenv('LABELS_TO_SUMMARIZE_EVERYX_MINUTES'))
    labels_to_ignore_everyx_minutes = json.loads(os.getenv('LABELS_TO_IGNORE_EVERYX_MINUTES'))

    label_gpt: str = os.getenv('LABEL_GPT_SUMMARIZED')

    allow_list = os.getenv('MATRIX_ALLOWED_USER')
    
    mailconnector = MailConnector.MailConnector()
    openaiconnector = OpenAIConnector.OpenAiConnector()
       
    #dict for all summarized mails
    summarized_mails = {}
    
    def __init__(self):
        self.client = Client(
            mxid=self.user_id,
            base_url=self.base_url,
            token=self.token,
        )
        self.client.ignore_initial_sync = True
        self.client.ignore_first_sync = True
        self.client.add_event_handler(EventType.ROOM_MEMBER, self.handle_invite)
        self.client.add_event_handler(EventType.ROOM_MESSAGE, self.handle_message)

        logging.info("MailBot initialized")
    
    async def start(self):
        logging.info("MailBot started")
        whoami = await self.client.whoami()

        logging.info(f"\tConnected, I'm {whoami.user_id} using {whoami.device_id}")
        self.scheduled_mail_task = asyncio.create_task(self.get_scheduled_mail())
        
        await self.client.start(None)

    async def handle_invite(self, event: StrippedStateEvent) -> None:
        # Ignore the message if it's not an invitation for us.
        if (event.state_key == self.user_id
            and event.content.membership == Membership.INVITE):
            # If it is, join the room.
            if event.sender in self.allow_list:
                await self.client.join_room(event.room_id)
                with open ("room_id.json", "w") as f:
                    json.dump(event.room_id, f)
                logging.info(f"\tJoined {event.room_id}")
        
    async def send_text_message(self, room_id: str, message: str) -> None:
        content = TextMessageEventContent(msgtype=MessageType.TEXT, body=message)
        await self.client.send_message(room_id=room_id, content=content)

    def split_message_command(self, message: str):   
        
        try:
            # Remove the '!mail' command from the message
            message = message.replace('!mail', '').strip()

            # Split the message into parts based on the keywords
            parts = re.split(r'(?=\bcount:|\bpositive:|\bnegative:)', message)

            # Filter out any empty strings
            parts = [part for part in parts if part]

            # Further split each part on the colon character and store in a dictionary
            data = {part.split(": ")[0]: part.split(": ")[1] for part in parts}

            # Retrieve the values from the dictionary
            count = int(data["count"])
            positiveList = eval(data["positive"])
            negativeList = eval(data["negative"])
            return count, positiveList, negativeList
        except Exception as e:
            logging.error(e)
            return None, None, None

    async def handle_message(self, event: MessageEvent) -> None:
        if event.sender == self.user_id:
            return
        if event.content.msgtype != MessageType.TEXT:
            return

        #save event.room_id to json
        with open ("room_id.json", "w") as f:
            json.dump(event.room_id, f)
        
        message_text = event.content.body
        if "!help" in message_text:

            await self.send_text_message(event.room_id, BOT_HELP_MESSAGE)
            return

        if "!mail" in message_text:
            counts, positiveList, negativeList = self.split_message_command(str(message_text))
            if counts is None:
                answer = "No valid command"
                await self.send_text_message(event.room_id, answer)
                return
            else:
                answer = self.mailconnector.get_mail(maxResults=counts, positiveLabels=positiveList, negativeLabels=negativeList)
        elif "!model" in message_text:
            open_ai_model = message.body.split("model: ")[1]
            self.openaiconnector.set_model(open_ai_model)
            answer = "Model changed to {open_ai_model}"
            await self.send_text_message(event.room_id, answer)
        elif "!help" not in message_text:
            answer = "No valid command"
            await self.send_text_message(event.room_id, answer)
            return            

        if answer is not None:
            for mail in answer:
                answer_string = f"{mail['subject']}\nSummary: {mail['summary']}\nImportance: {mail['importance']}"
                #extract number from importance if importance is a string and has also strings in it
                #importance : int = int(''.join(filter(str.isdigit, mail['importance'])))
                
                if int(mail['importance']) > self.importance_to_answer:
                    answer_string += f"\nResponse: {mail['response']}"
                    
                await self.send_text_message(
                    event.room_id,
                    answer_string)
        else:
            await self.send_text_message(
                event.room_id,
                "No new mails")

    async def get_scheduled_mail(self):
        #get pickled room_id
        logging.info("get_scheduled_mail started")
        room_id = None
 

          
        
        while True:
            await asyncio.sleep(30)
            while True:      
                if os.path.exists("./room_id.json"):
                    with open("./room_id.json", "r") as f:
                        room_id = json.load(f)
                        break
                else:
                    logging.error("room_id.json not found")
                    await asyncio.sleep(10)
                    continue
            
            current_time = datetime.now().time()

            if current_time.minute % self.check_every_minutes == 0: # every 15 minutes
            #if current_time.minute % 1 == 0: # every 15 minutes
                logging.info(f"checked: every {self.check_every_minutes} minutes")
                answer = self.mailconnector.get_mail(self.how_many_mails, ['UNREAD', 'INBOX'], ['READ', self.label_gpt, 'TurnApp'])
                if answer is not None:

                    for mail in answer:
                        answer_string = f"{mail['subject']}\nSummary: {mail['summary']}\nImportance: {mail['importance']}"
                        if int(mail['importance'] > 2):
                            answer_string += f"\nResponse: {mail['response']}"
            
                            await self.send_text_message(
                                room_id,
                                answer_string)            
                else:
                    logging.info("scheduled checked: no new mails")
            #every day at the time defined in time_to_message
            if current_time.strftime("%H:%M") == self.time_to_message:
                logging.info(f"checked: at {self.time_to_message}")
                
                answer = self.mailconnector.get_mail(self.how_many_mails, self.labels_to_summarize, self.labels_to_ignore)
                if answer is not None:
                        
                    for mail in answer:
                        answer_string = f"{mail['subject']}\nSummary: {mail['summary']}\nImportance: {mail['importance']}"
                        if int(mail['importance'] > 2):
                            answer_string += f"Response: {mail['response']}"
                            
                        await self.send_text_message(
                            room_id,
                            answer_string)
                else:
                        # await self.send_text_message(
                        # room_id,
                        # "No new scheduled mails")
                        logging.info("scheduled checked: no new mails")
                

async def main():
    bot = MailBot()
    
    #await bot.get_labels()
    await bot.start()


#main python routine
if __name__ == "__main__":
    logging.info("Starting MailBot")
    asyncio.run(main())