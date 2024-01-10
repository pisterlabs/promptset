import logging
from langchain.chains import LLMChain
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.output_parsers import RetryWithErrorOutputParser
from langchain.output_parsers.pydantic import OutputParserException
from typing import Any, Dict, List, Optional

from ...hnclient import HackerNewsStory
from .prompt import PROMPT, output_parser

logger = logging.getLogger(__name__)


class ReviewerChain(LLMChain):

    prompt = PROMPT

    @property
    def input_keys(self) -> List[str]:
        return ["story"]

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        story = get_story_string(inputs["story"])
        retry_parser = RetryWithErrorOutputParser.from_llm(parser=output_parser, llm=self.llm)
        output = super()._call({"story": story}, run_manager=run_manager)

        try:
            output = output_parser.parse(output[self.output_key])
        except OutputParserException:
            try:
                output = retry_parser.parse_with_prompt(
                    output[self.output_key],
                    self.prompt.format_prompt(story=story)
                )
            except OutputParserException as e:
                logger.warning(f"Abort {story.title}, {e}.")
                output = ""

        return {self.output_key: output}

    def _acall(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        raise NotImplementedError()


def get_story_string(story: HackerNewsStory) -> str:
    text = story.documents[0].page_content if story.documents else ""

    return f"""Title: {story.title}

Submitted at: {story.posted_at}

Hacker News score: {story.score}

Text: {text}"""
