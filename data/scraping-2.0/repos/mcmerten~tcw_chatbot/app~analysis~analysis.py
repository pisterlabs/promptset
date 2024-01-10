import json
import openai
import datetime

from app.database import DatabaseManager, Conversation, Summary
from app.chatbot.prompts import DefaultPrompts
from app.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)

openai.api_key = settings.OPENAI_API_KEY

db_manager = DatabaseManager()
db_session = db_manager.create_session()

def list_conversations(session, all_conversations=True):
    query = session.query(Conversation.conversation_id,
                          Conversation.user_id).distinct()
    # TODO: implement filter
    if not all_conversations:
        query = query.filter(...)

    return query.all()


def fetch_conversation(session, conversation_id):
    conversations = [c[0] for c in list_conversations(session)]

    if conversation_id not in conversations:
        logger.error("Conversation not found")
        return

    conversation = (session.query(Conversation)
                    .filter(Conversation.conversation_id == conversation_id)
                    .order_by(Conversation.created_at)
                    .all())

    conversation_str = ""

    for msg in conversation:
        if msg.user_msg:
            conversation_str += f"user: {msg.user_msg}\n"

        if msg.bot_msg:
            conversation_str += f"assistant: {msg.bot_msg}\n"

    return {
        "user_id": conversation[0].user_id,
        "conversation_id": conversation_id,
        "conversation_str": conversation_str
    }


def create_response(conversation):
    schema = DefaultPrompts.summary_schema()
    schema["required"] = ["name", "email", "phone", "company", "company_size", "industry", "role", "interest", "pain", "budget", "additional_info"]
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=[
            {"role": "system", "content": f"CONVERSATION HISTORY: '{conversation['conversation_str']}"},
            {"role": "user", "content": "Extract only information stated by the user information from the conversation history."}
        ],
        functions=[{"name": "extract_lead_data", "parameters": DefaultPrompts.summary_schema()}],
        function_call={"name": "extract_lead_data"},
        temperature=0,
    )
    result = json.loads(completion.choices[0].message.function_call.arguments)

    logger.info(f"Data extracted for {conversation['conversation_id']}")

    # add missing keys with empty values
    for key in schema["required"]:
        if key not in result.keys():
            result[key] = None

    # reorder keys to match schema
    result = {k: result[k] for k in schema["required"]}
    result['conversation_id'] = conversation['conversation_id']
    result['user_id'] = conversation['user_id']
    result['created_at'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return Summary(**result)


if __name__ == "__main__":

    for convo in list_conversations(db_session, all_conversations=True):
        conversation = fetch_conversation(db_session, convo[0])
        response = create_response(conversation)
        logger.info(f"Writing to database: {response}")
        db_manager.write_to_db(response)