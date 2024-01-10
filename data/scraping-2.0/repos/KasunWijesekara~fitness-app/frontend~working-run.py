from flask import Blueprint, jsonify, make_response, session, request
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from models import Chat
import os
import time
import openai

from database import (
    db,
    app,
)  # Import the Flask app and the SQLAlchemy instance from database.py
from models import Chat
import uuid

from dotenv import load_dotenv

load_dotenv()

# Remove the Flask app initialization
# app = Flask(__name__)
# app.secret_key = os.getenv("FLASK_SECRET_KEY")
openai.api_key = os.getenv("OPENAI_API_KEY")

# Enable CORS for all routes and all origins
CORS(app, supports_credentials=True)

# Add a to_dict method to the Chat model
Chat.to_dict = lambda self: {
    "id": self.id,
    "user_message": self.user_message,
    "ai_response": self.ai_response,
    "session_id": self.session_id,
    "ip_address": self.ip_address,
    "timestamp": self.timestamp.isoformat(),
}


@app.route("/api/chats", methods=["GET"])
def get_chats():
    chats = Chat.query.all()
    return jsonify([chat.to_dict() for chat in chats])


migrate = Migrate(app, db)

# Create a blueprint for chatbot routes
chatbot_blueprint = Blueprint("chatbot", __name__)


# Define a route for chatting with the bot
@chatbot_blueprint.route("/message", methods=["POST"])
def chat_with_bot():
    start_time = time.time()
    user_message = request.json.get("message", "")
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that helps the user.",
            },
            {"role": "user", "content": user_message},
        ],
        max_tokens=100,
        temperature=0.1,
    )
    end_time = time.time()
    app.logger.info(f"Time taken: {end_time - start_time} seconds")
    response_message = response.choices[0].message["content"].strip()

    # Retrieve or set the session ID
    session_id = request.json.get("session_id", "")
    if session_id is None:
        session_id = str(uuid.uuid4())
        app.logger.debug(f"New session ID generated: {session_id}")

    ip_address = request.remote_addr

    # Save the chat to the database
    new_chat = Chat(
        user_message=user_message,
        ai_response=response_message,
        session_id=session_id,
        ip_address=ip_address,
    )
    db.session.add(new_chat)
    db.session.commit()

    # Create the response and set the session ID cookie
    response = make_response(jsonify({"response": response_message}))
    response.set_cookie("session_id", session_id, domain="localhost", samesite="Lax")

    return response


# Register the blueprint
app.register_blueprint(chatbot_blueprint, url_prefix="/frontend/chatbot")

# Run the Flask app
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True, port=5100)
