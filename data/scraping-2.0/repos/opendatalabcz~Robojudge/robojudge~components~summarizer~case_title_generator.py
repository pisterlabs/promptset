from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain

from robojudge.utils.logger import logging
from robojudge.components.reasoning.llm_definitions import standard_llm

logger = logging.getLogger(__name__)

SYSTEM_MESSAGE_TEMPLATE = """\
Your task is to create a catchy/pithy/funny title for an article about a court ruling based on a summary of that ruling.
The title should relate to what the case was about.
Create your title ONLY in Czech.
"""


class CaseTitleGenerator:
    NEXT_CHUNK_SIZE = 4096 - 1000

    def __init__(self) -> None:
        system_message_prompt = SystemMessagePromptTemplate.from_template(
            SYSTEM_MESSAGE_TEMPLATE
        )
        human_message_prompt = HumanMessagePromptTemplate.from_template(
            "Here is the summary: {summary}"
        )

        prompt = ChatPromptTemplate.from_messages(
            [system_message_prompt, human_message_prompt]
        )

        self.llm_chain = LLMChain(llm=standard_llm, prompt=prompt)

    async def generate_title(self, summary: str) -> str:
        try:
            return await self.llm_chain.arun(summary)
        except Exception:
            logger.exception(f"Error while generating title:")
            return ""


title_generator = CaseTitleGenerator()

if __name__ == "__main__":
    test_summary = """
Soud rozhodoval o žalobě, ve které žalobkyně požadovala zaplacení peněz od žalovaného za cestování bez platné jízdenky. Soud posoudil věc podle platného zákona o drahách, který umožňuje uložit cestujícímu, který nemá platný jízdní doklad, zaplacení jízdného a přirážky. V tomto případě byla přirážka stanovena na 1 500 Kč. Žalobkyně a žalovaný uzavřeli smlouvu o přepravě, podle které je žalovaný povinen zaplatit žalobkyni jízdné a přirážku. Žalobkyně má také právo na úrok z prodlení. Soud rozhodl, že žalobkyni náleží náhrada nákladů řízení ve výši 1 489 Kč, která zahrnuje soudní poplatek a odměnu advokáta.
"""

    print(title_generator.generate_title(test_summary))
