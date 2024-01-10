import sqlite3
import openai
import logging
import json
from typing import Optional

logger = logging.getLogger(__name__)

class GPT4SMSAutoReply:
    def __init__(self, api_key: str, default_model: str = "gpt-3.5-turbo"):
        openai.api_key = api_key
        self.default_model = default_model
        self.db = sqlite3.connect("conversations.db")
        self.create_table()
        self.auto_reply_enabled = False

    def create_table(self):
        cursor = self.db.cursor()
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS conversations (
            contact_number TEXT PRIMARY KEY,
            history TEXT
        )
        ''')
        self.db.commit()

    def handle_incoming_sms(self, contact_number: str, message: str) -> Optional[str]:
        if not self.auto_reply_enabled:
            logger.info("Auto-reply is disabled. No response generated.")
            return None

        # Retrieve or initialize the conversation history for the contact
        conversation_history = self.get_conversation_history(contact_number)

        # Add the incoming SMS to the conversation history
        conversation_history.append({"role": "user", "content": message})

        # Generate a response using GPT-4
        response_message = self.generate_response(contact_number, conversation_history)

        if response_message is None:
            logger.error("Failed to generate response message.")
            return None

        # Add the outgoing SMS to the conversation history
        conversation_history.append({"role": "assistant", "content": response_message})

        # Update the conversation history in the database
        self.update_conversation_history(contact_number, json.dumps(conversation_history))

        return response_message

    def get_conversation_history(self, contact_number: str) -> list:
        cursor = self.db.cursor()
        cursor.execute('''
        SELECT history FROM conversations
        WHERE contact_number = ?
        ''', (contact_number,))
        row = cursor.fetchone()
        if row is None:
            return []
        return json.loads(row[0])

    def update_conversation_history(self, contact_number: str, history: str):
        cursor = self.db.cursor()
        cursor.execute('''
        INSERT OR REPLACE INTO conversations (contact_number, history)
        VALUES (?, ?)
        ''', (contact_number, history))
        self.db.commit()

    def generate_response(self, contact_number: str, conversation_history: list) -> Optional[str]:
        model = self.get_model_for_contact(contact_number)
        messages = [{"role": msg["role"], "content": msg["content"]} for msg in conversation_history]

        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Failed to generate response: {str(e)}")
            return None

    def get_model_for_contact(self, contact_number: str) -> str:
        # Here you can implement any logic to dynamically select the model based on the contact
        return self.default_model

    def enable_auto_reply(self):
        self.auto_reply_enabled = True
        logger.info("Auto-reply is enabled.")

    def disable_auto_reply(self):
        self.auto_reply_enabled = False
        logger.info("Auto-reply is disabled.")
