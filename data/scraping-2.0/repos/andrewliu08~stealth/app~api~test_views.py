import json
import os
import time
from app.constants.language import Language

from flask import Response, request, jsonify, stream_with_context
from openai.types.chat.chat_completion_chunk import (
    ChatCompletionChunk,
    Choice as ChunkChoice,
    ChoiceDelta,
)

from app import caching
from app.api import api_utils
from app.app import app, s3_client
from app.constants.fixed_response_options import FIXED_RESPONSE_OPTIONS
from app.generative.openai_gpt import (
    OpenAIReponseOptionStreamDFA,
    parse_options_with_translations,
)
from app.models import ConversationParticipant, Message

from app.utils.aws_utils import generate_presigned_url

test_tts_object = "polly.a35a3a62-ff8f-4e1f-b84a-f92670c44d6f.mp3"


@app.route("/")
def hello_world():
    resp = {"data": ["Hello World", "x", "y"]}
    return jsonify(resp)


@app.route("/test_stream_tts")
def test_stream_tts():
    conversation_id = request.args.get("conversationId")
    message_id = request.args.get("messageId")
    if conversation_id is None or message_id is None:
        return jsonify(error="Missing parameter"), 400

    conversation_id = conversation_id.lower()
    message_id = message_id.lower()
    print("-" * 10)
    print("call test_stream_tts")
    print("conversation_id:", conversation_id)
    print("message_id:", message_id)

    def generate():
        s3_response = s3_client.get_object(
            Bucket=os.environ.get("SHUO_TTS_BUCKET_NAME"), Key=test_tts_object
        )
        for chunk in s3_response["Body"].iter_chunks(chunk_size=4096):
            yield chunk

    return Response(generate(), mimetype="audio/mp3")


@app.route("/test_new_conversation", methods=["POST"])
def test_new_conversation():
    params = request.get_json()
    content = params.get("content")
    user_lang = params.get("userLang")
    resp_lang = params.get("respLang")
    print("-" * 10)
    print("call test_new_conversation")
    print("content:", content)
    print("user_lang:", user_lang)
    print("resp_lang:", resp_lang)

    presigned_tts_url = generate_presigned_url(
        s3_client,
        object_name=test_tts_object,
        bucket_name=os.environ.get("SHUO_TTS_BUCKET_NAME"),
        expires_in=3600,
    )

    conversation_dict = {
        "id": "a66e89f7-a440-4122-af1b-5cb67fd86c5c",
        "intro_message": "I am using a translation app. Please speak into the phone when you respond.",
        "history": [
            {
                "id": "5fbec385-368b-4762-bcc4-eb03d50db9fc",
                "sender": "user",
                "content": "Excuse me, could you help me find the closest hospital?\n\nI am using a translation app. Please speak into the phone when you respond.",
                "translation": "Excusez-moi, pourriez-vous m'aider à trouver l'hôpital le plus proche ?\n\nJ'utilise une application de traduction. Veuillez parler dans le téléphone lorsque vous répondez.",
                "tts_uri": presigned_tts_url,
                "tts_task_id": "e0a6a1b2-e739-4c00-853d-67febd415da2",
            }
        ],
        "user_lang": "english",
        "resp_lang": "french",
    }
    return jsonify(conversation_dict), 200


@app.route("/test_new_user_message", methods=["POST"])
def test_new_user_message():
    params = request.get_json()
    conversation_id = params.get("conversationId")
    content = params.get("content")
    print("-" * 10)
    print("call test_new_user_message")
    print("conversation_id:", conversation_id)
    print("content:", content)
    if conversation_id is None or content is None:
        return jsonify(error="Missing parameter"), 400
    conversation_id = conversation_id.lower()

    presigned_tts_url = generate_presigned_url(
        s3_client,
        object_name=test_tts_object,
        bucket_name=os.environ.get("SHUO_TTS_BUCKET_NAME"),
        expires_in=3600,
    )

    new_message = Message(
        id=caching.create_id(),
        sender=ConversationParticipant.USER,
        content=content,
        translation="asdf",
        tts_uri=presigned_tts_url,
        tts_task_id="d7582c37-cc79-4c82-a931-aa6108d03177",
    )

    return jsonify(new_message.to_dict()), 200


@app.route("/test_new_resp_message", methods=["POST"])
def test_new_resp_message():
    conversation_id = request.form.get("conversationId")
    file = request.files.get("file")
    print("-" * 10)
    print("call test_new_resp_message")
    print("conversation_id:", conversation_id)
    print("filename:", file.filename)
    if conversation_id is None or file is None:
        return jsonify(error="Missing parameter"), 400
    if not api_utils.allowed_audio_file(file.filename):
        return jsonify(error="Invalid file name"), 400

    conversation_id = conversation_id.lower()
    # from app.app import redis_client

    # conversation = caching.get_conversation(redis_client, conversation_id)
    # if conversation is None:
    #     return jsonify(error="Conversation not found"), 404

    api_utils.save_resp_audio(file, file.filename)

    time.sleep(3)

    presigned_tts_url = generate_presigned_url(
        s3_client,
        object_name=test_tts_object,
        bucket_name=os.environ.get("SHUO_TTS_BUCKET_NAME"),
        expires_in=3600,
    )

    new_message = Message(
        id=caching.create_id(),
        sender=ConversationParticipant.RESPONDENT,
        content="你受伤了吗？",
        translation="Are you hurt?",
        tts_uri=presigned_tts_url,
        tts_task_id="d7582c37-cc79-4c82-a931-aa6108d03177",
    )
    # conversation.new_message(new_message)
    # caching.save_conversation(redis_client, conversation)

    return jsonify(new_message.to_dict()), 200


@app.route("/test_delete_message", methods=["POST"])
def test_delete_message():
    params = request.get_json()
    conversation_id = params.get("conversationId")
    message_id = params.get("messageId")
    print("-" * 10)
    print("call test_delete_message")
    print("conversation_id:", conversation_id)
    print("message_id:", message_id)

    if conversation_id is None or message_id is None:
        return jsonify(error="Missing parameter"), 400

    return jsonify({}), 200


@app.route("/test_response_options", methods=["POST"])
def test_response_options():
    params = request.get_json()
    conversation_id = params.get("conversationId")
    print("-" * 10)
    print("call test_response_options")
    print("conversation_id:", conversation_id)
    if conversation_id is None:
        return jsonify(error="Missing parameter"), 400

    conversation_id = conversation_id.lower()

    # time.sleep(3)
    response_content = """Option 1:
"Je vais à Amsterdam."
"I'm going to Amsterdam."

Option 2:
"Je n'ai pas encore décidé de la ville exacte."
"I haven't decided on the exact city yet."

Option 3:
"Ma destination est Rotterdam."
"My destination is Rotterdam."
"""
    options = parse_options_with_translations(response_content)

    return jsonify(api_utils.format_response_options(options)), 200


@app.route("/test_response_options_stream", methods=["GET"])
def test_response_options_stream():
    conversation_id = request.args.get("conversationId")
    if conversation_id is None:
        return jsonify(error="Missing parameter"), 400
    print("-" * 10)
    print("call test_response_options_stream")
    print("conversation_id:", conversation_id)

    def generate():
        for response in FIXED_RESPONSE_OPTIONS[Language.ENGLISH]:
            response_event = {"event": "message", "data": response}
            end_event = {"event": "end"}
            print(json.dumps(response_event))
            print(json.dumps(end_event))
            yield f"data: {json.dumps(response_event)}\n\n"
            yield f"data: {json.dumps(end_event)}\n\n"

        response_content = """
<Start>
"I'm going to Amsterdam."
<End>
<Start>
"I haven't decided on the exact city yet."
<End>
<Start>
"My destination is Rotterdam."
<End>
"""

        parser = OpenAIReponseOptionStreamDFA()
        for char in response_content:
            chunk = ChatCompletionChunk(
                id="chatcmpl-8J7mnNeFwHVJxH9Pw0MEdSg1hgdOF",
                choices=[
                    ChunkChoice(
                        delta=ChoiceDelta(
                            content=char,
                            function_call=None,
                            role=None,
                            tool_calls=None,
                        ),
                        finish_reason=None,
                        index=0,
                    )
                ],
                created=1699568853,
                model="gpt-3.5-turbo-0613",
                object="chat.completion.chunk",
                system_fingerprint=None,
            )

            events = parser.process_chunk(chunk)
            for event in events:
                print(json.dumps(event))
                yield f"data: {json.dumps(event)}\n\n"
            time.sleep(0.01)

        print("".join(parser.response_chars))
        for start_idx, end_idx in parser.message_idx:
            print("".join(parser.response_chars[start_idx:end_idx]))

    return Response(stream_with_context(generate()), content_type="text/event-stream")
