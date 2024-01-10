import openai
import logging

from azure.search.documents import SearchClient
from azure.search.documents.models import QueryType

from athena.libs.prompt import GPTPrompt, ChatGPTPrompt, FollowUpQuestionsPrompt
from athena.core.models import Overrides, ChatHistory, MessageResponse
from athena.core.config import AzureSettings

logger = logging.getLogger()
settings = AzureSettings()


class ReadRetrieveReadApproach:
    def __init__(
        self,
        sourcepage_field: str,
        content_field: str,
        gpt_model: str | None = "text-davinci-003",
        chatgpt_model: str | None = "gpt-3.5-turbo",
    ):
        self.gpt_model = gpt_model
        self.chatgpt_model = chatgpt_model
        self.sourcepage_field = sourcepage_field
        self.content_field = content_field

    def run(
        self,
        search_client: SearchClient,
        history: list[ChatHistory],
        overrides: Overrides | None = None,
    ) -> None:
        logger.info("Building search query...")
        query = self.build_search_query(history=history)

        logger.info("Executing semantic search...")
        search_result = self.cognitive_search(
            query=query, client=search_client, overrides=overrides
        )
        content = "\n".join(search_result)
        follow_up_questions_prompt = (
            FollowUpQuestionsPrompt() if overrides.suggest_followup_questions else ""
        )

        chat_prompt = ChatGPTPrompt(
            sources=content,
            history=history,
            followup_questions=follow_up_questions_prompt,
        )
        logger.info("Running chatgpt on generated prompt...")
        chat_completion = openai.ChatCompletion.create(
            model=self.chatgpt_model,
            messages=chat_prompt,
            temperature=overrides.temperature,
            n=1,
        )
        return MessageResponse(
            data_points=search_result,
            answer=chat_completion.choices[0].message["content"],
            thoughts=f"Searched for:<br>{query}<br>",
        )

    def cognitive_search(
        self, query: str, client: SearchClient, overrides: Overrides
    ) -> list[str]:
        filter = (
            "category ne '{}'".format(overrides.exclude_category.replace("'", "''"))
            if overrides.exclude_category
            else None
        )
        if overrides.semantic_ranker:
            result = client.search(
                search_text=query,
                filter=filter,
                query_type=QueryType.SEMANTIC,
                query_language="en-us",
                query_speller="lexicon",
                semantic_configuration_name=settings.semantic_configuration,
                top=overrides.top,
                query_caption="extractive|highlight-false"
                if overrides.semantic_captions
                else None,
            )
        else:
            result = client.search(search_text=query, filter=filter, top=overrides.top)
        if overrides.semantic_captions:
            results = [
                doc[self.sourcepage_field]
                + ": "
                + self.nonewlines(
                    " . ".join([content.text for content in doc["@search.captions"]])
                )
                for doc in result
            ]
        else:
            results = [
                doc[self.sourcepage_field]
                + ": "
                + self.nonewlines(doc[self.content_field])
                for doc in result
            ]
        return results

    def build_search_query(self, history: list[ChatHistory]) -> str:
        prompt = GPTPrompt(
            history=self.get_chat_history_as_text(
                history=history, include_last_turn=False
            ),
            question=history[-1].user,
        )
        completion = openai.Completion.create(
            model=self.gpt_model,
            prompt=prompt,
            temperature=0.0,
            max_tokens=32,
            n=1,
            stop=["\n"],
        )
        return completion.choices[0].text

    def get_chat_history_as_text(
        self,
        history: list[ChatHistory],
        include_last_turn: bool = True,
        approx_max_tokens: int = 1000,
    ) -> str:
        history_text = ""
        for hist in reversed(history if include_last_turn else history[:-1]):
            history_text = (
                """<|im_start|>user"""
                + "\n"
                + hist.user
                + "\n"
                + """<|im_end|>"""
                + "\n"
                + """<|im_start|>assistant"""
                + "\n"
                + (hist.bot + """<|im_end|>""" if hist.bot else "")
                + "\n"
                + history_text
            )
            if len(history_text) > approx_max_tokens * 4:
                break
        return history_text

    @staticmethod
    def nonewlines(string: str):
        return string.replace("\n", " ").replace("\r", " ")
