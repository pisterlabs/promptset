import bleach
import openai
from django.conf import settings
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response

from livechat.constants import API_RESULT_KEY, NO_CACHE_HEADERS, API_GENERIC_FAILURE, API_GENERIC_SUCCESS
from livechat.forms import ChatMessageForm
from livechat.helpers import get_chat_session, get_chat_bot
from utilities.debugging import log_message
from utilities.utility_functions import is_empty, get_client_ip

openai.api_key = getattr(settings, "OPENAI_API_KEY")


@api_view(["GET"])
def get_session_messages(request, **kwargs):
    messages = []

    continue_chat = request.GET.get("continue_chat") == "true"

    chat_session = get_chat_session(request)
    new_messages = chat_session.get_messages(include_sent=continue_chat)

    if new_messages.count() > 0:
        chat_bot = get_chat_bot(request)

        for message in new_messages:
            message.shown = True
            message.message_type = 'sent' if message.sender == request.user else "received"

            if not is_empty(message.sender):
                if message.sender == request.user:
                    message.name = chat_session.name

                elif message.sender == chat_bot.bot_user:
                    message.name = chat_bot.bot_name

            message.save()

            messages.append(message.response_dict)

    return Response({API_RESULT_KEY: messages}, status=status.HTTP_200_OK, headers=NO_CACHE_HEADERS)


@api_view(["POST"])
def add_message(request):
    chat_session = get_chat_session(request)
    chat_bot = get_chat_bot(request)

    message = bleach.clean(request.data.get("message"))

    form = ChatMessageForm(request.POST)

    result = API_GENERIC_FAILURE
    response_status = status.HTTP_401_UNAUTHORIZED

    if not form.is_valid():
        log_message(form)

    else:
        if request.user.is_authenticated:
            try:
                chat_session.add_message(request.user, message)

            except Exception as e:
                log_message("Failed to create object: %s" % e)

            else:
                result = API_GENERIC_SUCCESS
                response_status = status.HTTP_200_OK

                chat_user = chat_bot.bot_user

                bot = chat_session.bot(client_ip=get_client_ip(request), debug=False)
                chat_response, response_source = bot.respond(message)

                if not is_empty(chat_response):
                    chat_session.add_message(chat_user, chat_response, response_source)

    return Response({API_RESULT_KEY: result}, status=response_status, headers=NO_CACHE_HEADERS)


@api_view(["POST"])
def set_location(request):
    chat_session = get_chat_session(request)

    chat_session.latitude = request.data.get("latitude")
    chat_session.longitude = request.data.get("longitude")
    chat_session.save()

    result = API_GENERIC_SUCCESS
    response_status = status.HTTP_200_OK

    return Response({API_RESULT_KEY: result}, status=response_status, headers=NO_CACHE_HEADERS)
