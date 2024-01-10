import base64
import io
import logging
import re
import string
from typing import List, Optional

import openai
from aiohttp import ClientResponseError
from openai.error import ServiceUnavailableError, RateLimitError

from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import CallbackContext
from telethon.tl.types import Message as TelethonMessage, Chat as TelethonChat, User as TelethonUser

import bobweb
from bobweb.bob import database, openai_api_utils, telethon_service, async_http
from bobweb.bob.command import ChatCommand, regex_simple_command_with_parameters, get_content_after_regex_match
from bobweb.bob.openai_api_utils import notify_message_author_has_no_permission_to_use_api, \
    ResponseGenerationException, GptModel, \
    determine_suitable_model_for_version_based_on_message_history, GptChatMessage, ContextRole
from bobweb.bob.resources.bob_constants import PREFIXES_MATCHER
from bobweb.bob.utils_common import object_search, send_bot_is_typing_status_update
from bobweb.web.bobapp.models import Chat as ChatEntity

logger = logging.getLogger(__name__)

# Regexes for matching sub commands
system_prompt_pattern = regex_simple_command_with_parameters('system')
use_quick_system_pattern = rf'{PREFIXES_MATCHER}([123])'
use_quick_system_message_without_prompt_pattern = rf'(?i)^{use_quick_system_pattern}\s*$'
set_quick_system_pattern = rf'{PREFIXES_MATCHER}[123]\s*=\s*'


class GptCommand(ChatCommand):
    invoke_on_edit = True
    invoke_on_reply = True

    def __init__(self):
        super().__init__(
            name='gpt',
            # 'gpt' with optional 3, 3.5 or 4 in the end
            regex=regex_simple_command_with_parameters(r'gpt(3)?(\.5)?4?'),
            help_text_short=('!gpt[3|4]', '[|1|2|3] [prompt] -> (gpt3.5|4) vastaus')
        )

    async def handle_update(self, update: Update, context: CallbackContext = None):
        """
        1. Check permission. If not, notify user
        2. Check that has content after command or is reply to another message. If not, notify user
        3. Check if message has any subcommand. If so, handle that
        4. Default: Handle as normal prompt
        """
        has_permission = openai_api_utils.user_has_permission_to_use_openai_api(update.effective_user.id)
        command_parameters = self.get_parameters(update.effective_message.text)

        has_content_after_command = len(command_parameters) > 0
        is_reply_to_message = update.effective_message.reply_to_message is not None
        has_image_media = update.effective_message.photo is not None and len(update.effective_message.photo) > 0

        if not has_permission:
            return await notify_message_author_has_no_permission_to_use_api(update)

        # If has no parameters and is not reply to another message -> give info message.
        # If is reply to another message or if contains any image media, process normally
        elif not has_content_after_command and not is_reply_to_message and not has_image_media:
            quick_system_prompts = database.get_quick_system_prompts(update.effective_message.chat_id)
            no_parameters_given_notification_msg = generate_no_parameters_given_notification_msg(quick_system_prompts)
            return await update.effective_chat.send_message(no_parameters_given_notification_msg)

        # if contains quick system message command without prompt
        elif re.search(use_quick_system_message_without_prompt_pattern, command_parameters) is not None:
            no_prompt_after_quick_system_message_selection = generate_no_parameters_given_notification_msg()
            return await update.effective_chat.send_message(no_prompt_after_quick_system_message_selection)

        # If contains update system prompt sub command
        elif re.search(system_prompt_pattern, command_parameters) is not None:
            await handle_system_prompt_sub_command(update, command_parameters)

        # If contains quick system set sub command
        elif re.search(set_quick_system_pattern, command_parameters) is not None:
            await handle_quick_system_set_sub_command(update, command_parameters)

        else:
            await gpt_command(update, context)

    def is_enabled_in(self, chat: ChatEntity):
        """ Is always enabled for chat. Users specific permission is specified when the update is handled """
        return True


async def gpt_command(update: Update, context: CallbackContext) -> None:
    """ Internal controller method of inputs and outputs for gpt-generation """
    started_reply_text = 'Vastauksen generointi aloitettu. Tämä vie 30-60 sekuntia.'
    started_reply = await update.effective_chat.send_message(started_reply_text)
    await send_bot_is_typing_status_update(update.effective_chat)

    use_quote = True
    try:
        reply = await generate_and_format_result_text(update)
    except (ServiceUnavailableError, RateLimitError):
        # Same error both for when service not available or when too many requests
        # have been sent in a short period of time from any chat by users.
        # In case of error, given message is not sent as quote to the original request
        # message. This is done so that they do not affect message reply history.
        use_quote = False
        reply = ('OpenAi:n palvelu ei ole käytettävissä tai se on juuri nyt ruuhkautunut. '
                 'Ole hyvä ja yritä hetken päästä uudelleen.')
    except ResponseGenerationException as e:  # If exception was raised, reply its response_text
        use_quote = False
        reply = e.response_text
    except ClientResponseError as e:
        use_quote = False
        reply = 'Jokin virhe Gpt-komennon käsittelyssä, mille ei ole jaksettu tehdä varsinaista virheiden käsittelyä.'

    # All replies are as 'reply' to the prompt message to keep the message thread
    await update.effective_message.reply_text(reply, quote=use_quote, parse_mode=ParseMode.MARKDOWN)

    # Delete notification message from the chat
    if context is not None:
        await context.bot.deleteMessage(chat_id=update.effective_message.chat_id,
                                        message_id=started_reply.message_id)


async def generate_and_format_result_text(update: Update) -> string:
    """ Determines system message, current message history and call api to generate response """
    openai_api_utils.ensure_openai_api_key_set()

    message_history: List[GptChatMessage] = await form_message_history(update)
    context_msg_count = len(message_history)
    model: GptModel = determine_used_model(update.effective_message.text, message_history)

    system_message_obj: GptChatMessage = determine_system_message(update)
    if system_message_obj is not None:
        message_history.insert(0, system_message_obj)

    # Uses OpenAi Http api as Openai python library version is too old and its upgrade is left for later development.
    payload = {
        "model": model.name,
        "messages": model.serialize_message_history(message_history),
        "max_tokens": 4096  # 4096 is maximum number of response tokens available with gpt 4 visions model
    }
    url = 'https://api.openai.com/v1/chat/completions'
    headers = {'Authorization': 'Bearer ' + openai.api_key}

    response = await async_http.post_expect_json(url=url, headers=headers, json=payload)
    content = object_search(response, 'choices', 0, 'message', 'content')

    cost_message = openai_api_utils.state.add_chat_gpt_cost_get_cost_str(
        model,
        object_search(response, 'usage', 'prompt_tokens'),
        object_search(response, 'usage', 'completion_tokens'),
        context_msg_count
    )
    return f'{content}\n\n{cost_message}'


def determine_used_model(message_text: str, message_history: List[GptChatMessage]) -> GptModel:
    command_name_parameter = re.search(rf'(?i)^{PREFIXES_MATCHER}gpt(\d?\.?\d?)?', message_text)[1]
    return determine_suitable_model_for_version_based_on_message_history(command_name_parameter, message_history)


def determine_system_message(update: Update) -> Optional[GptChatMessage]:
    """ Returns either given quick system prompt or chats main system prompt """
    command_parameter = instance.get_parameters(update.effective_message.text)
    regex_match = re.match(rf'{PREFIXES_MATCHER}([123])', command_parameter)
    quick_system_parameter = regex_match[1] if regex_match is not None else None

    if quick_system_parameter is not None and quick_system_parameter != '':
        quick_system_prompts = database.get_quick_system_prompts(update.effective_chat.id)
        content = quick_system_prompts.get(quick_system_parameter, None)
    else:
        content = database.get_gpt_system_prompt(update.effective_chat.id)

    if content is None:
        return None
    return GptChatMessage(ContextRole.SYSTEM, content)


async def form_message_history(update: Update) -> List[GptChatMessage]:
    """ Forms message history for reply chain. Latest message is last in the result list.
        This method uses both PTB (Telegram bot api) and Telethon (Telegram client api).
        Adds all images contained in any messages in the reply chain to the message history """
    messages: list[GptChatMessage] = []

    # First create object of current message
    cleaned_message = bobweb.bob.openai_api_utils.remove_openai_related_command_text_and_extra_info(
        update.effective_message.text)

    # If message has image, download all possible images related to the message by media_group_id
    # (Each image is its own message even though they appear to be grouped in the chat client)
    base_64_images = []
    if update.effective_message.photo:
        base_64_images = await download_all_images_as_base_64_strings_for_update(update)

    if cleaned_message != '' or len(base_64_images) > 0:
        # If the message contained only gpt-command, it is not added to the history
        messages.append(GptChatMessage(ContextRole.USER, cleaned_message, base_64_images))

    # If current message is not a reply to any other, early return with it
    reply_to_msg = update.effective_message.reply_to_message
    if reply_to_msg is None:
        return messages

    # Now, current message is reply to another message that might be replied to another.
    # Iterate through the reply chain and find all messages in it
    next_id = reply_to_msg.message_id

    # Iterate over all messages in the reply chain. Telethon Telegram Client is used from here on
    while next_id is not None:
        message, next_id = await find_and_add_previous_message_in_reply_chain(update, next_id)
        if message is not None:
            messages.append(message)

    messages.reverse()
    return messages


async def find_and_add_previous_message_in_reply_chain(update: Update, next_id: int) -> \
        tuple[Optional[GptChatMessage], Optional[int]]:
    # Telethon api from here on. Find message with given id. If it was a reply to another message,
    # fetch that and repeat until no more messages are found in the reply thread

    current_message: TelethonMessage = await telethon_service.client.find_message(chat_id=update.effective_chat.id,
                                                                                  msg_id=next_id)
    # Message authors id might be in attribute 'peer_id' or in 'from_id'
    author_id = None
    if current_message.from_id and current_message.from_id.user_id:
        author_id = current_message.from_id.user_id

    if author_id is None:
        # If author is not found, set message to be from user
        is_bot = False
    else:
        author: TelethonUser = await telethon_service.client.find_user(author_id)  # Telethon User
        is_bot = author.bot

    next_id = object_search(current_message, 'reply_to', 'reply_to_msg_id', default=None)

    base_64_images = []
    if current_message.media and current_message.media.photo:
        chat = await telethon_service.client.find_chat(update.effective_chat.id)
        base_64_images = await download_all_images_as_base_64_strings(chat, current_message)

    cleaned_message = bobweb.bob.openai_api_utils.remove_openai_related_command_text_and_extra_info(current_message.message)
    if cleaned_message != '' or len(base_64_images) > 0:
        # If author of message is bot, it's message is added with role assistant and
        # cost so far notification is removed from its messages
        context_role = ContextRole.ASSISTANT if is_bot else ContextRole.USER
        message = GptChatMessage(context_role, cleaned_message, base_64_images)
        return message, next_id

    return None, next_id


async def download_all_images_as_base_64_strings_for_update(update: Update) -> List[str]:
    # Handle any possible media. Message might contain a single photo or might be a part of media group that contains
    # multiple photos. All images in media group can't be requested in any straightforward way. Here we try to find
    # All associated photos and add them to the message history. This search uses Telethon Client API.
    chat = await telethon_service.client.find_chat(update.effective_chat.id)
    original_message = await telethon_service.client.find_message(chat.id, update.effective_message.message_id)
    return await download_all_images_as_base_64_strings(chat, original_message)


async def download_all_images_as_base_64_strings(chat: TelethonChat, message: TelethonMessage) -> List[str]:
    messages = await telethon_service.client.get_all_messages_in_same_media_group(chat, message)
    image_bytes_list = await telethon_service.client.download_all_messages_image_bytes(messages)
    return convert_all_image_bytes_base_64_data(image_bytes_list)


def convert_all_image_bytes_base_64_data(image_bytes_list: List[io.BytesIO]) -> List[str]:
    """ Converts all io.BytesIO objects to base64 data strings """
    base_64_images = []
    for image_bytes in image_bytes_list:
        base64_photo = base64.b64encode(image_bytes.getvalue()).decode('utf-8')
        image_url = f'data:image/jpeg;base64,{base64_photo}'
        base_64_images.append(image_url)
    return base_64_images


def remove_gpt_command_related_text(text: str) -> str:
    # remove gpt-command and any sub commands
    pattern = rf'^({instance.regex})(\s*{PREFIXES_MATCHER}\S*)*\s*'
    return re.sub(pattern, '', text).strip()


def msg_obj(role: ContextRole, content: str) -> dict[str, str]:
    return {'role': role.value, 'content': content}


async def handle_quick_system_set_sub_command(update: Update, command_parameter):
    sub_command = command_parameter[1]
    sub_command_parameter = get_content_after_regex_match(command_parameter, set_quick_system_pattern)

    quick_system_prompts = database.get_quick_system_prompts(update.effective_message.chat_id)
    current_prompt = quick_system_prompts.get(sub_command, None)

    # If actual prompt after quick system set option is empty
    if sub_command_parameter.strip() == '':
        empty_message_last_part = f" tyhjä. Voit asettaa pikaohjausviestin sisällön komennolla '/gpt {sub_command} = (uusi viesti)'."
        current_message_msg = empty_message_last_part if current_prompt is None else f':\n\n{current_prompt}'
        await update.effective_message.reply_text(
            f"Nykyinen pikaohjausviesti {sub_command} on nyt{current_message_msg}")
    else:
        database.set_quick_system_prompt(update.effective_chat.id, sub_command, sub_command_parameter)
        await update.effective_message.reply_text(f"Uusi pikaohjausviesti {sub_command} asetettu.")


def generate_no_parameters_given_notification_msg(quick_system_prompts: dict = None):
    if quick_system_prompts:
        quick_system_prompts_str = ''.join([f'\n{key}: {value}' for key, value in quick_system_prompts.items()])
    else:
        quick_system_prompts_str = ''
    no_parameters_given_notification_msg = \
        f'Anna jokin syöte komennon jälkeen. [.!/]gpt (syöte). Voit valita jonkin kolmesta valmiista ' \
        f'ohjeistusviestistä laittamalla numeron 1-3 ennen syötettä. {quick_system_prompts_str}'
    return no_parameters_given_notification_msg


async def handle_system_prompt_sub_command(update: Update, command_parameter):
    sub_command_parameter = get_content_after_regex_match(command_parameter, system_prompt_pattern)
    # If sub command parameter is empty, print current system prompt. Otherwise, update system prompt for chat
    if sub_command_parameter is None or sub_command_parameter.strip() == '':
        current_prompt = database.get_gpt_system_prompt(update.effective_chat.id)
        empty_message_last_part = " tyhjä. Voit asettaa system-viestin sisällön komennolla '/gpt /system {uusi viesti}'."
        current_message_msg = empty_message_last_part if current_prompt is None else f':\n\n{current_prompt}'
        await update.effective_message.reply_text(f"Nykyinen system-viesti on nyt{current_message_msg}")
    else:
        database.set_gpt_system_prompt(update.effective_chat.id, sub_command_parameter)
        await update.effective_message.reply_text("System-viesti asetettu annetuksi.", quote=True)


# Single instance of these classes
instance = GptCommand()
