import os
import uuid
from flask import Blueprint, render_template, redirect, url_for, request, jsonify
from openai.error import InvalidRequestError
from .utils import (
    generate_image,
    generate_essay,
    generate_ai_chatbot_response,
    generate_corrected_transcript_with_cloudinary_audio_file,
    generate_corrected_transcript_with_local_audio_file,
)
import cloudinary.uploader
import cloudinary.api
from dotenv import load_dotenv

# Load environment variables from .flaskenv
load_dotenv()

cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET"),
)

main_bp = Blueprint("main", __name__)
image_gen_bp = Blueprint("image_gen", __name__)
essay_gen_bp = Blueprint("essay_gen", __name__)
ai_chatbot_bp = Blueprint("ai_chatbot", __name__)
speech_to_text_bp = Blueprint("speech_to_text", __name__)


@main_bp.route("/", methods=("GET", "POST"))
def index():
    return redirect(url_for("ai_chatbot.ai_chatbot"))


@image_gen_bp.route("/", methods=("GET", "POST"))
def image_generator():
    if request.method == "POST":
        prompt = request.form["description"]
        status, result = generate_image(prompt)
        return redirect(
            url_for(
                "image_gen.image_generator",
                status=status,
                result=result,
                prompt=prompt,
            )
        )
    args = request.args
    status, result, prompt = args.get("status"), args.get("result"), args.get("prompt")
    return render_template(
        "image_generator.html", status=status, result=result, prompt=prompt
    )


@essay_gen_bp.route("/", methods=("GET", "POST"))
def essay_generator():
    if request.method == "POST":
        essay_idea = request.form.get("essayIdea", "")
        # Get the value of "essayWordCount" from the form or provide a default value (e.g., 500) if not present
        essay_word_count = request.form.get("essayWordCount", 500)
        prompt = f"Write me an essay about {essay_idea} with {essay_word_count} words."
        messages = [{"role": "user", "content": prompt}]
        status, result = generate_essay(messages)
        return jsonify({"status": status, "result": result, "prompt": prompt}), 200
    return render_template("essay_generator.html")


@ai_chatbot_bp.route("/", methods=("GET", "POST"))
def ai_chatbot():
    messages = (
        request.args.get("messages")
        if request.args.get("messages")
        else [{"role": "system", "content": "What can I help you today?"}]
    )
    if request.method == "POST":
        prompt = request.form["input"]
        messages.append({"role": "user", "content": prompt})
        status, messages = generate_ai_chatbot_response(messages)
        return jsonify({"status": status, "messages": messages}), 200
    return render_template("ai_chatbot.html", messages=messages)


# save audio file to Cloudinary
@speech_to_text_bp.route("/", methods=("GET", "POST"))
def speech_to_text():
    if request.method == "POST":
        if "audio" in request.files:
            audio_file = request.files["audio"]
            if audio_file:
                # Generate a unique filename for the audio file
                folder = "open-ai-audio"
                filename = f"{str(uuid.uuid4())}.wav"
                # Save the audio file to the "open-ai-audio" folder in Cloudinary
                result = cloudinary.uploader.upload(
                    audio_file,
                    folder=folder,
                    resource_type="raw",
                    public_id=filename,
                    overwrite=True,
                )
                # Get the public URL of the uploaded audio file
                audio_url = result["secure_url"]
                (
                    status,
                    result,
                ) = generate_corrected_transcript_with_cloudinary_audio_file(audio_url)
                # Delete the file
                public_id = f"{folder}/{filename}"
                cloudinary.uploader.destroy(public_id, resource_type="raw")
                return jsonify({"status": status, "result": result}), 200
        return jsonify({"status": "error", "message": "No audio file received."})
    return render_template("speech_to_text.html")


# save audio file to Local File Sytem
# @speech_to_text_bp.route("/", methods=("GET", "POST"))
# def speech_to_text():
#     if request.method == "POST":
#         if "audio" in request.files:
#             audio_file = request.files["audio"]
#             if audio_file:
#                 # Generate a unique filename for the audio file
#                 filename = f"{str(uuid.uuid4())}.wav"
#                 # Save the audio file to the "static/audio" directory
#                 audio_file_path = os.path.join("app", "static", "audio", filename)
#                 audio_file.save(audio_file_path)
#                 # Check if the file exists before generating the transcript
#                 if os.path.isfile(audio_file_path):
#                     status, result = generate_corrected_transcript_with_local_audio_file(audio_file_path)
#                     # Close the file
#                     audio_file.close()
#                     # Delete the file
#                     os.remove(audio_file_path)
#                     return jsonify(status=status, result=result)
#         return jsonify({"status": "error", "message": "No audio file received."})
#     return render_template("speech_to_text.html")
