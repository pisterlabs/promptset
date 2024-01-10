"""AgentSummarize"""

from typing import Any

from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionFunctionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionUserMessageParam,
)

import scraping_utils
import slack_link_utils
import slack_mrkdwn_utils
from agent_gpt import AgentGPT


class AgentSummarize(AgentGPT):
    """URLから本文を抜き出して要約する"""

    def __init__(
        self, context: dict[str, Any], chat_history: list[dict[str, str]]
    ) -> None:
        """初期化"""
        super().__init__(context, chat_history)
        self._openai_model = "gpt-3.5-turbo-1106"
        self._output_max_token: int = 4096
        self._context_max_token: int = 16385 - self._output_max_token

        self._openai_temperature: float = 0.0
        self._openai_stream = True

    def learn_context_memory(self) -> None:
        """コンテキストメモリの初期化"""
        super().learn_context_memory()
        url: str = slack_link_utils.extract_and_remove_tracking_url(
            self._chat_history[-1]["content"]
        )
        self._logger.debug("scraping url=%s", url)
        if not scraping_utils.is_allow_scraping(url):
            raise ValueError("scraping is not allowed")
        site: scraping_utils.Site = scraping_utils.scraping(url)
        if site is None:
            raise ValueError("scraping failed")
        self._context["site"] = site

        with open("./conf/summarize_prompt.toml", "r", encoding="utf-8") as file:
            self._context["summarize_prompt"] = file.read()

    def build_prompt(
        self, chat_history: list[dict[str, Any]]
    ) -> list[
        ChatCompletionSystemMessageParam
        | ChatCompletionUserMessageParam
        | ChatCompletionAssistantMessageParam
        | ChatCompletionToolMessageParam
        | ChatCompletionFunctionMessageParam
    ]:
        """OpenAI APIを使って要約するためのpromptを生成する"""
        site: scraping_utils.Site = self._context.get("site")  # type: ignore
        prompt: str = self._context["summarize_prompt"]
        prompt += f'[記事情報]\ntitle="{site.title}"\n'
        if site.heading is not None:
            prompt += "heading=["
            for head in site.heading:
                prompt += f'"{head}",\n'
            prompt += "]\n"
        prompt += "\n"
        prompt += f"[本文]\n{site.content}\n"
        return super().build_prompt([{"role": "user", "content": prompt}])

    def build_message_blocks(self, content: str) -> list:
        """レスポンスからブロックを作成する"""
        site: scraping_utils.Site = self._context.get("site")  # type: ignore
        if site is None:
            raise ValueError("site is empty")

        title_link: str = slack_link_utils.build_link(site.url, site.title)
        mrkdwn: str = slack_mrkdwn_utils.convert_mrkdwn(content)

        blocks: list[dict] = [
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": title_link},
            },
            {"type": "divider"},
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": mrkdwn},
            },
        ]
        return blocks
