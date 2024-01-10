from datetime import datetime, timedelta

from langchain.chains import LLMChain

from mtg.data_handler import CardDB
from mtg.utils.logging import get_logger
from .chat_history import ChatHistory
from .lang_chain import create_chains

logger = get_logger(__name__)


class MagicGPT:
    def __init__(
        self,
        card_db: CardDB,
        chat_history: ChatHistory,
        model: str = "gpt-3.5-turbo",
        temperature_deck_building: int = 0.7,
        max_token_limit: int = 3000,
        max_responses: int = 1,
    ):
        (
            classifier_chain,
            deckbuilding_chat,
            rules_question_chat,
            memory,
        ) = create_chains(
            model=model,
            temperature_deck_building=temperature_deck_building,
            max_token_limit=max_token_limit,
            max_responses=max_responses,
        )

        self._last_updated: datetime = datetime.now()
        self.chat_history: ChatHistory = chat_history
        self.memory = memory
        self.card_db: CardDB = card_db
        self.classifier_chain: LLMChain = classifier_chain
        self.deckbuilding_chat: LLMChain = deckbuilding_chat
        self.rules_question_chat: LLMChain = rules_question_chat
        self.conversation_topic: str = None  # TODO should be ENUM

    def clear_memory(self):
        self.memory.clear()
        self.chat_history.clear()
        logger.info("memory cleared")

    def process_user_query(self, query):
        # branch
        topic = self.classifier_chain.invoke(query)
        logger.info(f"topic of the question is: {topic}")

        self.conversation_topic = topic

        if topic == "deck building":
            chat = self._process_deckbuilding_question(query)
        else:
            chat = self._process_rules_question(query)

        return chat

    def ask(self):
        try:
            if self.conversation_topic == "deck building":
                response_iterator = self._ask_deckbuilding_question()
            else:
                response_iterator = self._ask_rules_question()

            for response in response_iterator:
                message = self.card_db.create_minimal_message(
                    text=response, role="assistant"
                )
                if self.chat_history.chat[-1].role == "user":
                    self.chat_history.chat.append(message)
                else:
                    self.chat_history.chat[-1] = message
                chat = self.chat_history.get_human_readable_chat(number_of_messages=6)
                yield chat

            # create final message
            message = self.card_db.create_message(
                response, role="assistant", max_number_of_cards=10, threshold=0.4
            )
            self.chat_history.chat[-1] = message
            final_chat = self.chat_history.get_human_readable_chat(number_of_messages=6)

            # save memory
            if self.conversation_topic == "deck building":
                self.memory.save_context(
                    inputs={"human_input": self.chat_history.chat[-2].text},
                    outputs={"assistant": self.chat_history.chat[-1].text},
                )
            else:
                self.memory.save_context(
                    inputs={"human_input": self.chat_history.chat[-2].text},
                    outputs={"assistant": self.chat_history.chat[-1].text},
                )

            yield final_chat

        except Exception as e:
            logger.error(e)
            self.clear_memory()
            yield [
                [
                    self.chat_history.chat[-1].text,
                    f"Something went wrong, I am restarting. Please ask the question again.",
                ]
            ]

    def _process_deckbuilding_question(self, query) -> list[list[str, str]]:
        logger.info("processing deck building query")

        message = self.card_db.create_message(
            query,
            role="user",
            max_number_of_cards=2,
            threshold=0.5,
            lasso_threshold=0.03,
        )

        message = self.card_db.add_additional_cards(
            message=message, max_number_of_cards=10, threshold=0.5, lasso_threshold=0.03
        )
        self.chat_history.add_message(message=message)

        return self.chat_history.get_human_readable_chat(number_of_messages=6)

    def _ask_deckbuilding_question(self) -> str:
        logger.info("invoking deck building chain")

        card_data = self.chat_history.get_card_data(
            number_of_messages=1,
            max_number_of_cards=12,
            include_price=True,
            include_rulings=False,
        )

        partial_message = ""
        for response in self.rules_question_chat.stream(
            {"human_input": self.chat_history.chat[-1].text, "card_data": card_data}
        ):
            partial_message += response.content
            yield partial_message

    def _process_rules_question(self, query) -> list[list[str, str]]:
        logger.info("processing rules question")

        # process query
        message = self.card_db.create_message(
            query,
            role="user",
            max_number_of_cards=2,
            threshold=0.5,
            lasso_threshold=0.03,
        )
        self.chat_history.add_message(message=message)

        return self.chat_history.get_human_readable_chat(number_of_messages=6)

    def _ask_rules_question(self):
        logger.info("invoking rules chain")

        card_data = self.chat_history.get_card_data(
            number_of_messages=2,
            max_number_of_cards=6,
            include_price=False,
            include_rulings=True,
        )

        # TODO add rules data
        partial_message = ""
        for response in self.rules_question_chat.stream(
            {"human_input": self.chat_history.chat[-1].text, "card_data": card_data}
        ):
            partial_message += response.content
            yield partial_message
