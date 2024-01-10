from os import path
from loguru import logger
from telegram import Update
from src.bot.handlers import mom
from src.domain import dc, util, config
from telegram import ParseMode
import traceback
import openai


app_config = config.get_config()

BOT_USERNAME = app_config.get("TELEGRAM", "BOT_USERNAME")

mom_response_blacklist = [BOT_USERNAME]

COMMAND_COST = 200

openai.api_key = app_config.get("OPENAI", "API_KEY")


def handle(update: Update, context):

    try:

        dc.log_command_usage("mom3", update)

        initiator_id = update.message.from_user.id
        if initiator_id is None:
            logger.error("[mom3] initiator_id was None!")
            return

        if not util.paywall_user(initiator_id, COMMAND_COST):
            update.message.reply_text(
                "Sorry! You don't have enough ₹okda! Each `/mom3` costs {} ₹okda.".format(
                    COMMAND_COST
                ),
                parse_mode=ParseMode.MARKDOWN,
            )
            return

        protagonist = util.extract_pretty_name_from_tg_user(update.message.from_user)

        message = mom.extract_target_message(update)
        if message is None:
            logger.info("[mom3] message was None!")
            return

        prompt = open(
            path.join(util.RESOURCES_DIR, "openai", "mom3-prompt.txt"), "r"
        ).read()
        prompt += "\n\nUser: " + message[:200]
        prompt += "\nChaddi: "

        logger.debug("[mom3] prompt={}", prompt)

        response = openai.Completion.create(
            engine="text-davinci-001",
            prompt=prompt,
            temperature=0.5,
            max_tokens=60,
            top_p=0.3,
            frequency_penalty=0.5,
            presence_penalty=0,
        )

        logger.info("[mom3] openai response='{}'", response)

        response = "{}".format(response["choices"][0]["text"])

        if update.message.reply_to_message:
            update.message.reply_to_message.reply_text(response)
            return
        else:
            update.message.reply_text(response)
            return

    except Exception as e:
        logger.error(
            "Caught Error in mom.handle - {} \n {}",
            e,
            traceback.format_exc(),
        )
        return
