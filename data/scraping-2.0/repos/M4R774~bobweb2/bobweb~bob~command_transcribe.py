from telegram import Update
from telegram.ext import CallbackContext

from bobweb.bob import openai_api_utils
from bobweb.bob.command import ChatCommand, regex_simple_command
from bobweb.bob.message_handler_voice import transcribe_and_send_response
from bobweb.bob.openai_api_utils import notify_message_author_has_no_permission_to_use_api
from bobweb.bob.utils_common import send_bot_is_typing_status_update


class TranscribeCommand(ChatCommand):
    invoke_on_edit = True
    invoke_on_reply = True

    def __init__(self):
        super().__init__(
            name='tekstitä',
            regex=regex_simple_command('tekstitä'),
            help_text_short=('!tekstitä', 'tekstittää kohteen ääniviestin')
        )

    async def handle_update(self, update: Update, context: CallbackContext = None):
        """ Checks requirements, if any fail, user is notified. If all are ok, transcribe-function is called """
        has_permission = openai_api_utils.user_has_permission_to_use_openai_api(update.effective_user.id)
        target_message = update.effective_message.reply_to_message

        if not has_permission:
            return await notify_message_author_has_no_permission_to_use_api(update)
        elif not target_message:
            reply_text = 'Tekstitä mediaa sisältävä viesti vastaamalla siihen komennolla \'\\tekstitä\''
            return await update.effective_message.reply_text(reply_text)

        # Use this update as the one which the bot replies with.
        # Use voice of the target message as the transcribed voice message
        media = target_message.voice or target_message.audio or target_message.video or target_message.video_note
        if media:
            await send_bot_is_typing_status_update(update.effective_chat)
            await transcribe_and_send_response(update, media)
        else:
            reply_text = 'Kohteena oleva viesti ei ole ääniviesti, äänitiedosto tai videotiedosto jota voisi tekstittää'
            await update.effective_message.reply_text(reply_text)
