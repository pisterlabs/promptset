# /SBU/app/api/conv_gen_route.py
from flask import Blueprint, request, jsonify
import os
import openai
import json
from app.models.user import db, Bot, Message, Debate, ConversationSetting
from datetime import datetime

conv_gen_route = Blueprint('conv_gen_route', __name__)

# Be sure to use your own OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

@conv_gen_route.route('/create_conversation', methods=['POST'])
def create_conversation():
    # Get input parameters
    bot_id_1 = request.json.get('bot_id_1')
    bot_id_2 = request.json.get('bot_id_2')
    conv_settings_id = request.json.get('conv_settings_id')
    max_messages = request.json.get('max_messages')
    topic = request.json.get('topic')
    length_limit = request.json.get('length_limit')
    owner_id = request.json.get('owner_id')
    if not owner_id:
        return jsonify({"error": "Owner not found"}), 400

    # Fetch the bots
    bot_1 = Bot.query.get(bot_id_1)
    bot_2 = Bot.query.get(bot_id_2)

    # Check if bot_1 and bot_2 are not None
    # if bot_1 is none or bot_2 is none
    if not bot_1 or not bot_2:
        return jsonify({"error": "Bot not found."}), 400

    bot_names = [bot_1.name, bot_2.name]
    conv_settings = ConversationSetting.query.get(conv_settings_id)

    # Validate the parameters
    if not bot_1 or not bot_2:
        return jsonify({"error": "Bot not found."}), 400

    if not max_messages:
        return jsonify({"error": "max_messages not found"}), 400

    if not conv_settings:
        return jsonify({"error": "conv_settings not found"}), 400

    if not topic:  # check if topic is not null
        return jsonify({"error": "Topic not found"}), 400

    # Create a new debate
    new_debate = Debate(conversation_setting_id=conv_settings.id, initiator_bot_id=bot_1.id, owner_id=owner_id, opponent_bot_id=bot_2.id, start_time=datetime.utcnow(), topic=topic)
    db.session.add(new_debate)
    db.session.commit()

    # Initialize the conversation
    system_def = {
        "role": "system",
        "content": "You are a silent assistant who specializes in emulating speech between two characters based on provided settings."
        f"There should be exactly {max_messages} total messages exchanged."
    }
    # print('\n','System Def',system_def,'\n')

    user_request = {
        "role": "user",
        "content": f"Create a conversation happening under these circumstances: {conv_settings.setting_details} on the topic of: '{new_debate.topic}' between '{bot_1.name}' who is described as {bot_1.settings} and '{bot_2.name}' who is described as {bot_2.settings}. "
        "Limit each person's response to 150 words or less. Provide the conversation in a JSON object titled 'messages' where each new response is a different entry formatted exactly like so: {name: 'name', message:'response', index:'index'}. Format your response precisely so that it is parsable by this code: chat_content = chat['choices'][0]['message']['content'] chat_json = json.loads(chat_content) messages = chat_json['messages']"
    }

    # print('\n','User Request',user_request,'\n')

    chat = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=[system_def, user_request]
    )

    try:
        chat_content = chat['choices'][0]['message']['content']
        chat_json = json.loads(chat_content)
        messages = chat_json['messages']
    except (json.JSONDecodeError, KeyError):
        messages = []
        print("Unexpected response format from chat API")

    # Save the messages to the database
    if messages:
        for message_data in messages:
            print('\n','Message_Data:',message_data,'\n')
            message = Message(
                debate_id=new_debate.id,
                bot_id=bot_id_1 if message_data['name'] == bot_1.name else bot_id_2,
                name=message_data['name'],
                content=message_data['message'],
                role='assistant',
                index=int(message_data['index'])
            )
            db.session.add(message)
        db.session.commit()


    return jsonify({
        "message": "Conversation created successfully",
        "debate": {
            "id": new_debate.id,
            "topic": new_debate.topic
        },
        "cleaned_res": messages,
        "openAI_res": chat
    }), 200
