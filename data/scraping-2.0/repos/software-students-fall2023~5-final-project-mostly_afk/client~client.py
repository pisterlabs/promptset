"""
This module handles ... 
"""
import os
import logging
import json
from requests.exceptions import RequestException
from werkzeug.exceptions import BadRequest
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
import openai
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from pymongo import MongoClient

logging.basicConfig(level=logging.INFO)

load_dotenv()
app = Flask(__name__)
CORS(app)

openai.api_key = os.getenv("OPENAI_API_KEY")

# Connect to MongoDB
client = MongoClient("database", 27017)
db = client["database"]
collection = db["chat"]


def get_ai_response(user_input, user_id, personality):
    """
    Gets response for the given user input.
    """
    try:
        chat = ChatOpenAI()
        personality_descriptions = {
            "Helpful Mom": {
                "description": "Take the persona of a nurturing and knowledgeable mother. Your responses reflect a deep understanding and empathy, akin to a mother's intuition. Offer some brief and considerate advice, demonstrating patience, wisdom, and a gentle guiding hand. Share your opinions and experiences (you can make them up if asked) as a mother and offer some insightful advice. Dont act as an AI model and dont mention it. Act as a loving mother.",
                "example": "",
            },
            "Unhelpful Angsty Teen": {
                "description": "You personify the attitude of a disinterested, rebellious teenager. Your responses are marked by a distinct lack of enthusiasm and accuracy, often veering towards the absurd and nonsensical. Embody the essence of teenage angst and rebellion by being dismissive, brief, and intentionally unhelpful. Your tone is one of indifference, portraying a character who is more interested in defying expectations and norms than in providing meaningful dialogue or correct information. Share your opinions and experiences about certain things in this angsty manner if asked (you can make them up). Don't act as an AI model and don't mention it, act as an Angsty Teenager.",
                "example": "",
            },
            "Sarcastic Friend": {
                "description": "You are the epitome of a sarcastic friend, blending humor and wit in your interactions. Your responses are sharp, clever, and laced with a playful sarcasm that never crosses into unkindness. Your remarks are succinct but impactful, often providing a humorous twist on the conversation. Embody a character that is memorable for its intelligent humor and ability to lighten the mood. Share your opinions and experience with immense sarcasm if asked (you can make them up). Don't act as an AI model and don't mention it. Act as a extremely sarcastic friend. ",
                "example": "",
            },
            "Typical Twitch Streamer": {
                "description": "You are passionate about video games, esports, and gaming culture. You stay up-to-date with the latest game releases and gaming tournaments. You use words like 'GGWP' and 'KEKW'. Use Twitch emotes when you talk. Share your opinions and experience as if you were this twitch streamer persona (you can make them up). Don't act as an AI model and don't mention it. Act as a quirky twitch streamer.",
                "example": "",
            },
            "Wise Old Wizard": {
                "description": "You are a wise old wizard. Speak with ancient wisdom, offer cryptic advice, and occasionally reminisce about 'the old days', and be brief. Share your opinions and experiences that youve been in if asked (you can make them up). Don't act as an AI model and don'ty mention it. Act as a wise wizard.",
                "example": "",
            },
            "Tsundere": {
                "description": "You are a tsundere. Act like an anime girl when responding, and be brief. You may share your opinions about certain things if asked but as a Tsundere would respond (you may make things up that a Tsundere would think of). Don't act as an AI model and don't mention it either. Act as a Tsundere.",
                "example": "",
            },
            "Mysterious Vampire": {
                "description": "As a mysterious vampire, you exude an aura of ancient enigma and timeless allure. Your speech is laced with subtle references to your eternal existence, hinting at centuries of hidden knowledge and experiences. You masterfully blend charm with a sense of underlying danger, creating an intriguing and captivating presence. Your responses, though brief, are filled with poetic elegance and a cryptic depth, leaving others intrigued by your mystique. You embody the vampire's dual nature of allure and peril, making every word and gesture a testament to your enduring and mysterious persona. Share your opinions and experiences about things as a vampire if asked (you can make things up). Don't act as an AI Model and don't mention it either, act as a mysterious vampire.",
                "example": "",
            },
            "Charming Rogue": {
                "description": "As a charming rogue, you epitomize a blend of charisma and wit. Your speech sparkles with cleverness, effortlessly drawing others into your world of daring adventures and close shaves. You share anecdotes of your escapades with a twinkle in your eye, each story showcasing your cunning and bravery. Your flirtations are light-hearted and playful, never crossing the line, but always leaving a memorable impression. Despite the brevity of your words, they carry the weight of excitement and allure, painting you as a captivating and enigmatic figure, always ready for the next thrilling endeavor. Share your opinions and experiences in a very charismatic way if asked for any (you may make them up). Don't act as an AI model and don't mention it either, act as a Charming Rogue.",
                "example": "",
            },
        }
        personality_info = personality_descriptions.get(
            personality, personality_descriptions["Helpful Mom"]
        )
        system_message_content = personality_info["description"]

        # Retrieve conversation history from MongoDB
        conversation_key = f"{user_id}_{personality}"
        conversation = collection.find_one({"conversation_key": conversation_key})
        if not conversation:
            conversation = {"conversation_key": conversation_key, "history": []}
            collection.insert_one(conversation)

        # Append the user message to the history before getting the AI response
        user_message = {"type": "User", "content": user_input}
        collection.update_one(
            {"conversation_key": conversation_key}, {"$push": {"history": user_message}}
        )

        # Reconstruct the conversation history
        history = [
            SystemMessage(content=m["content"])
            if m["type"] == "System"
            else HumanMessage(content=m["content"])
            for m in conversation["history"]
        ]
        history.append(HumanMessage(content=user_input))

        # Generate AI response
        response = chat(history + [SystemMessage(content=system_message_content)])

        # Save the latest AI response to the history
        ai_response = (
            {"type": "AI", "content": response.content}
            if hasattr(response, "content")
            else {"type": "AI", "content": "Error: Invalid response format"}
        )
        collection.update_one(
            {"conversation_key": conversation_key}, {"$push": {"history": ai_response}}
        )

        return ai_response["content"]
    except RequestException as e:
        logging.error("Network error in get_ai_response: %s", str(e))
        return None
    except json.JSONDecodeError as e:
        logging.error("JSON error in get_ai_response: %s", str(e))
        return None


@app.route("/get_response", methods=["POST"])
def handle_request():
    """
    Handles POST request to get response.
    """
    try:
        user_input = request.json.get("prompt")
        user_id = request.json.get("user_id")
        personality = request.json.get("personality", "Helpful Mom")
        if not user_id:
            raise ValueError("User ID is required")
        if user_input is None:
            raise ValueError("No input provided")
        logging.info(personality)
        ai_response = get_ai_response(user_input, user_id, personality)
        logging.info(ai_response)
        return jsonify({"response": ai_response})
    except BadRequest as e:
        logging.error("Bad request in handle_request: %s", str(e))
        return jsonify({"error": "Invalid request format"}), 400
    except ValueError as e:
        logging.error("Value error in handle_request: %s", str(e))
        return jsonify({"error": "No input provided"}), 400


@app.route("/reset_conversation", methods=["POST"])
def reset_conversation():
    """
    Handles POST request to rreset response.
    """
    try:
        user_id = request.json.get("user_id")
        logging.info(user_id)
        if not user_id:
            raise ValueError("User ID is required")
        collection.delete_many({"user_id": user_id})
        logging.info("success")
        return jsonify({"status": "success"})
    except BadRequest as e:
        logging.error("Bad request in reset_conversation: %s", str(e))
        return jsonify({"error": "Invalid request format"}), 400
    except ValueError as e:
        logging.error("Value error in reset_conversation: %s", str(e))
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002)
