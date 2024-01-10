from flask import Blueprint, jsonify
from flask_jwt_extended import get_jwt_identity
from .models import User, Conversation, Message, Dictionary
from . import db
import os
import openai
import deepl

service = Blueprint("service", __name__)
OPENAI_TOKEN = os.environ.get('OPENAI_TOKEN')
DEEPL_TOKEN = os.environ.get('DEEPL_TOKEN')


class ChatAPIError(Exception):
    """Custom exception for handling chat API errors."""
    pass


def get_user_id_by_token_identify():
    username = get_jwt_identity()
    user_object = User.query.filter_by(username=username).first()
    user_id = user_object.id
    return user_id


def find_all_conversations_names_ids():
    user_id = get_user_id_by_token_identify()
    user_conversations = Conversation.query.filter_by(user_id=user_id).all()
    conversations_names_and_ids = [{"id": conversation.id, "name": conversation.conversation_name} for conversation in
                                   user_conversations]
    return conversations_names_and_ids


def find_conversation_by_conversation_id(conversation_id):
    conversation_object = Conversation.query.filter_by(id=conversation_id).first()
    if not conversation_object:
        return jsonify({"error": "Conversation not found"}), 404
    return conversation_object


def save_message_to_database(message_text, conversation_id, is_user, summary):
    new_message = Message(message_text=message_text, conversation_id=conversation_id, is_user=is_user, summary=summary)
    db.session.add(new_message)
    db.session.commit()


def prepare_api_payload(conversation_id):
    conversation_object = find_conversation_by_conversation_id(conversation_id)
    user_message = conversation_object.messages[-1].message_text  # hi
    if len(conversation_object.messages) > 1:
        sum_up_sentence = conversation_object.messages[-2].summary  # greeting.
    else:
        sum_up_sentence = None
    language = conversation_object.language
    return language, user_message, sum_up_sentence


def message_for_api(language, user_message, sum_up_sentence):
    # User response and sum up sentence if there is one
    sum_up_or_new_conv = ""
    formatted_messages = [{"role": "user", "content": user_message}]
    if sum_up_sentence:
        sum_up_or_new_conv += sum_up_sentence
    else:
        sum_up_or_new_conv += "sentence"

    # instruction for chat
    instruction = {"role": "system",
                   "content": f"You must return your whole response in JSON. You're a chat assistant fluent in {language}. Always begin with a summary sentence (max 15 words). Then, provide a concise answer or question (max 10 words) related to {sum_up_or_new_conv}. \n \n It's crucial to respond ONLY in format with key such as summary and answer. Only JSON allowed"}
    formatted_messages.insert(0, instruction)

    return formatted_messages


def prepare_messages(conversation_id):
    # last message (chat_message) and conversation summary
    conversation_object = find_conversation_by_conversation_id(conversation_id)
    if len(conversation_object.messages) >= 2:
        last_message = conversation_object.messages[-1].message_text
        # summary = conversation_object.messages[-1].summary
        summary = getattr(conversation_object.messages[-1].summary, 'summary', None)

    else:
        raise ValueError("Please, start conversation before using hint or sentence advanced correction")
    return last_message, summary


def call_chat_response(guidance_message):
    try:
        # call chat to response
        openai.api_key = OPENAI_TOKEN
        response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=guidance_message)

        # return chat response
        guidance_response = response["choices"][0]["message"]["content"]
        return guidance_response

    except (KeyError, ValueError) as e:
        raise ChatAPIError("Failed to get a response from the chat") from e


def save_to_db_dictionary(word_to_dictionary, translated_word, contex_sentence, source_lang, target_lang,
                          translated_contex_sentence):
    user_id = get_user_id_by_token_identify()
    new_word = Dictionary(user_id=user_id, word_to_dictionary=word_to_dictionary, translated_word=translated_word,
                          contex_sentence=contex_sentence, source_lang=source_lang, target_lang=target_lang,
                          translated_contex_sentence=translated_contex_sentence)
    db.session.add(new_word)
    db.session.commit()


def get_translate_deepl(word_to_translate, sentence_to_translate, source_lang, target_lang):
    auth_key = DEEPL_TOKEN
    translator = deepl.Translator(auth_key)

    translated_word = translator.translate_text(word_to_translate, source_lang=source_lang, target_lang=target_lang)
    translated_sentence = translator.translate_text(sentence_to_translate, source_lang=source_lang,
                                                    target_lang=target_lang)

    return translated_word.text, translated_sentence.text


