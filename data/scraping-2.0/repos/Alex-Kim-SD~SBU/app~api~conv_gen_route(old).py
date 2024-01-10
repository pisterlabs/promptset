# /SBU/app/api/conv_gen_route.py
from flask import Blueprint, request, jsonify
import os
import openai
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

    # Fetch the bots
    bot_1 = Bot.query.get(bot_id_1)
    bot_2 = Bot.query.get(bot_id_2)
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
    new_debate = Debate(conversation_setting_id=conv_settings.id, initiator_bot_id=bot_1.id,
                        opponent_bot_id=bot_2.id, start_time=datetime.utcnow(), topic=topic)
    db.session.add(new_debate)
    db.session.commit()

    # Initialize the conversation
    system_def = {
        "role": "system",
        "content": "You are a silent assistant who specializes in emulating speech between two people"
    }
    user_request = {
        "role": "user",
        "content": f"Emulate an arguement on this topic: {new_debate.topic} between {bot_1.name} who has these settings {bot_1.settings} and {bot_2.name} who has these settings {bot_2.settings}. "
                   f"Limit each person's response to 150 characters or less."
                   f"Seperate each response with a new line"
                   f"Start each response with ###"
                   f"Utilize all {max_messages} messages"
    }

    chat = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[system_def, user_request]
    )

    return jsonify({
        "message": "Conversation created successfully",
        "debate": {
            "id": new_debate.id,
            "topic": new_debate.topic
        },
        "openAI_res": chat
    }), 200
