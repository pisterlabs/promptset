import os

from flask import Blueprint, jsonify, send_file, send_from_directory
from openai import OpenAI
from sqlalchemy import func
from werkzeug.utils import secure_filename
from datetime import datetime
from models import *
from auth.utilities import get_user

magic_blueprint = Blueprint("magic_blueprint", __name__)


@magic_blueprint.route("/api/magic/post", methods=["POST"])
def user_update():
    current_app.logger.info("Reading from user")

    current_user = get_user(request)

    if current_user is None:
        return jsonify({"success": False, "message": "Not logged in"}), 401

    data = request.get_json()
    if data is None:
        return jsonify({"success": False, "message": "No data received"}), 400

    if "community" not in data or "title" not in data or "text" not in data:
        return jsonify({"success": False, "message": "Missing community, title, or text"}), 400

    try:
        community = int(data["community"])
    except ValueError:
        return jsonify({"success": False, "message": "Invalid community"}), 400

    topic = db.session.query(Community).filter(Community.id == community).first()

    if topic is None:
        return jsonify({"success": False, "message": "Invalid community"}), 400

    title = data["title"].replace("\n", " ")
    text = data["text"].replace("\n", " ")

    client = OpenAI(
        api_key=current_app.config["OPENAI_API_KEY"]
    )

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a user on a social media platform creating a post. You will generate a title and text. Ensure that your title is prefixed by \"Title:\", as well as your text with \"Text:\". Provided a query containing hints for both the title and text, generate a new title and text post based on the given information. If you are not provided a title and/or text, you can be creative and come up with your own title and text, but it MUST conform to the topic. You will also be provided the post topic, which MUST be followed over the title and text. DO NOT use emojis unless it is expected from the prompt. DO NOT use hashtags. Keep your post short, under about a thousand characters."},
            {"role": "user", "content": f"Topic: {topic.name}\n\nTitle: {title}\n\nText: {text}\n\n"}
        ]
    )

    data = response.choices[0].message.content

    # Split the Title and Text out

    title = data.split("Title:")[1].split("Text:")[0].strip()
    text = data.split("Text:")[1].strip()

    return jsonify({"success": True, "message": "Post generated", "title": title, "text": text}), 200
