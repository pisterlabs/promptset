# AidBot - Telegram bot project for finding volunteer help using semantic search
# Copyright (C) 2023
# Anastasia Mayorova aka EternityRei  <anastasiamayorova2003@gmail.com>
#    Andrey Vlasenko aka    chelokot   <andrey.vlasenko.work@gmail.com>

# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, either version 3 of the License, or any later version. This
# program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
# details. You should have received a copy of the GNU General Public License along with this program. If not,
# see <https://www.gnu.org/licenses/>.

import telebot
from src.database.ProposalsTable import ProposalsTable
from src.database.BotRequestsTable import BotRequestsTable
from src.bot.TelegramBotUtils import TelegramBotUtils
from src.database.data_types.ColumnNames import ColumnNames

from src.embeddings.OpenAITextEmbedder import OpenAITextEmbedder
from src.embeddings.EmbeddingAda1536 import EmbeddingAda1536
from src.config.OpenAIConfig import openai_api_key
from src.config.DatabaseConfig import site_table_name, user_table_name

from src.database.data_types.BotRequest import BotRequest


class AidBot:
    __proposals_table = ProposalsTable()
    __proposals_table.create_table_or_add_columns(site_table_name)
    __bot_requests_table = BotRequestsTable()
    __bot_requests_table.create_table_or_add_columns(user_table_name)

    __openai_text_embedder = OpenAITextEmbedder(openai_api_key, EmbeddingAda1536)
    __telebot = None

    def __init__(self, token):
        if AidBot.__telebot is not None:
            raise Exception("TelegramBot class is a singleton! Use get_instance() method to get the instance.")
        else:
            self.bot = telebot.TeleBot(token)  # type: telebot.TeleBot
            AidBot.__telebot = self

            @self.bot.message_handler(commands=['start'])
            def start(message):
                localization = message.from_user.language_code
                welcome_message = f'{TelegramBotUtils.get_welcome_message_text(localization)}, <b>{message.from_user.first_name} {message.from_user.last_name}</b>'
                welcome_message_line2 = f'{TelegramBotUtils.get_explanation_message_text(localization)}'
                self.bot.send_message(message.chat.id, welcome_message + '\n' + welcome_message_line2, parse_mode='html')

            @self.bot.message_handler(content_types=['text'])
            def get_user_text(message):
                user_text = message.text  # type: str
                localization = message.from_user.language_code  # type: str
                self.bot.send_message(message.chat.id, TelegramBotUtils.get_received_message_text(localization))

                request = BotRequest(
                    characteristics={
                        ColumnNames.description: user_text,
                        ColumnNames.bot_request_answer_message_id: None,
                        ColumnNames.bot_request_start: 0,
                        ColumnNames.bot_request_amount: 5,
                    },
                    embedder=self.__openai_text_embedder,
                )
                result = TelegramBotUtils.format_proposal_results(
                    proposals = self.__proposals_table.get_similar(request),
                    localization = localization,
                )

                # We want to have buttons "Next" and "Previous" to show more results
                buttons = TelegramBotUtils.get_next_and_previous_buttons(localization, start = 0)

                api_reply = self.bot.send_message(
                    message.chat.id, result[:4000],
                    reply_markup=buttons, parse_mode='html'
                )

                request = BotRequest(
                    characteristics = {
                        ColumnNames.description: user_text,
                        ColumnNames.bot_request_user_id: message.from_user.id,
                        ColumnNames.bot_request_answer_message_id: api_reply.message_id,
                        ColumnNames.bot_request_start: 0,
                        ColumnNames.bot_request_amount: 5,
                    },
                    embedder = self.__openai_text_embedder,
                    embedding = request.embedding,
                )
                self.__bot_requests_table.add(request)

            @self.bot.callback_query_handler(func=lambda call: True)
            def callback_inline(call):
                if call.message:
                    localization = call.from_user.language_code
                    if call.data == 'next' or call.data == 'previous':
                        request: BotRequest = self.__bot_requests_table.get_request(
                            answer_message_id = call.message.message_id,
                            user_id = call.from_user.id,
                        )
                        start = int(request.get_characteristic(ColumnNames.bot_request_start))
                        amount = int(request.get_characteristic(ColumnNames.bot_request_amount))
                        if call.data == 'next':
                            start += amount
                        else:
                            start = max(0, start - amount)
                        buttons = TelegramBotUtils.get_next_and_previous_buttons(localization, start)
                        new_result = TelegramBotUtils.format_proposal_results(
                            proposals = self.__proposals_table.get_similar(request, start = start, amount = amount),
                            localization = localization,
                        )
                        self.bot.edit_message_text(
                            chat_id=call.message.chat.id,
                            message_id=call.message.message_id,
                            text=new_result[:4000],
                        )
                        self.__bot_requests_table.update_start(
                            answer_message_id = call.message.message_id,
                            user_id = call.from_user.id,
                            start = start,
                        )
                        self.bot.edit_message_reply_markup(
                            chat_id = call.message.chat.id,
                            message_id = call.message.message_id,
                            reply_markup = buttons
                        )

    def start(self):
        self.bot.polling()

    @staticmethod
    def get_instance(token):
        if AidBot.__telebot is None:
            AidBot(token)
        return AidBot.__telebot
