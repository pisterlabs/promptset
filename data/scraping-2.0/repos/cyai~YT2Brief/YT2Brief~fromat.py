from langchain import LLMChain, PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema import StrOutputParser

import os
from dotenv import load_dotenv
import json

load_dotenv()


class Reformat:
    def __init__(self, summary) -> None:
        self.summary = summary
        self.llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k")
        self.prompt = self.template()

    def template(self) -> str:
        prompt_tempate = """
        The below is a summary of the video. Please reformat it to make it more readable according to the following configuration:
        {summary}


        The configuration should as follows:
        tone: {tone}
        use of bullet points: {use_of_bullet_points}
        average sentence length: {average_sentence_length}
        use of paragraphs: {use_of_paragraphs}
        average paragraph length: {average_paragraph_length}
        use of emojis: {use_of_emojis}
        markdown language use: {markdown_language_use}

        """

        prompt = PromptTemplate(
            template=prompt_tempate,
            input_variables=[
                "summary",
                "tone",
                "use_of_bullet_points",
                "average_sentence_length",
                "use_of_paragraphs",
                "average_paragraph_length",
                "use_of_emojis",
                "markdown_language_use",
            ],
        )

        return prompt

    async def reformat(self):
        llm_chain = self.prompt | self.llm | StrOutputParser()

        # llm_chain.input_schema.schema()

        with open("config.json", "r") as f:
            config = json.load(f)

        return llm_chain.invoke(
            {
                "summary": self.summary,
                "tone": config["summary"]["tone"],
                "use_of_bullet_points": config["summary"]["bullet-points"]["use"],
                "average_sentence_length": config["summary"]["bullet-points"][
                    "average-sentence-length"
                ],
                "use_of_paragraphs": config["summary"]["paragraphs"]["use"],
                "average_paragraph_length": config["summary"]["paragraphs"][
                    "average-paragraph-length"
                ],
                "use_of_emojis": config["summary"]["emojis"],
                "markdown_language_use": config["summary"]["markdown"],
            }
        )


        
