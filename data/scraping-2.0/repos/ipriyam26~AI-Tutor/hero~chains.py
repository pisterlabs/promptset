import contextlib
import enum
import math
import os
import re
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from dotenv import load_dotenv
from langchain.memory import ConversationBufferWindowMemory, ConversationBufferMemory
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain.chains import LLMChain
from hero.constants import (
    INFO_MSG,
    OUTLINE_MSG,
    ESSAY_MSG,
    THESIS_MSG,
    EXAMPLES,
    examples,
    AI_FIX,
)
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector

from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
from colorama import Fore, Style
from langchain.schema import Document
from langchain.chains.summarize import load_summarize_chain


load_dotenv()


class RunningState(enum.Enum):
    INFO = enum.auto()
    THESIS = enum.auto()
    OUTLINE = enum.auto()
    ESSAY = enum.auto()


class Chains:
    def __init__(self) -> None:
        self.embeddings = OpenAIEmbeddings()
        self.chat_history = []
        self.model = ChatOpenAI()
        self.current_chain = self.get_info()
        self.info = ""
        self.thesis = ""
        self.outline = ""
        self.essay = ""
        self.citation = ""
        self.sections = []

    def get_info(self) -> LLMChain:
        return LLMChain(
            prompt=PromptTemplate(
                template=INFO_MSG, input_variables=["history", "input"]
            ),
            memory=ConversationBufferMemory(),
            llm=ChatOpenAI(model="gpt-4", temperature=0)
        )

    def get_thesis(self) -> LLMChain:
        return LLMChain(
            prompt=PromptTemplate(
                template=THESIS_MSG, input_variables=["input", "history"]
            ),
            llm=self.model,
            memory=ConversationBufferMemory(),
        )

    def get_outline(self) -> LLMChain:
        return LLMChain(
            prompt=PromptTemplate(
                template=OUTLINE_MSG, input_variables=["input", "history"]
            ),
            memory=ConversationBufferMemory(),
            llm=ChatOpenAI(temperature=0.7, max_tokens=1500),
        )

    def get_essay(self) -> LLMChain:
        essay = PromptTemplate(
            template=ESSAY_MSG,
            input_variables=[
                "user_input",
                "history",
                "instruct",
                # "example",
                "CITATIONS",
            ],
        ).partial(
            instruct=self.info,
            CITATIONS=f"ADD CITATION IN STYLE: {self.citation}"
            if self.citation and self.citation.upper() != "NONE"
            else "",
        )
        return LLMChain(
            prompt=essay,
            memory=ConversationBufferMemory(
                memory_key="history", input_key="user_input"
            ),
            llm=ChatOpenAI(model="gpt-4", temperature=0),
        )

    def gen_info(self, input: str):
        reply = self.current_chain.predict(input=input)
        if "<info>" in reply:
            with contextlib.suppress(Exception):
                return self.extract_info(reply)
        return reply, True

    def extract_info(self, reply: str):
        self.info = reply.split("<info>")[1].split("</info>")[0]
        word_count_pattern = r"Word Count:\s+(\d+)"
        # Citation Style: Chicago
        citation_pattern = r"Citation Style:\s+(.*)"
        if match := re.search(word_count_pattern, self.info):
            self.word_count = match[1]
        if match := re.search(citation_pattern, self.info):
            self.citation = match[1]

        self.current_chain = self.get_thesis()

        return self.info, False

    def gen_thesis(self) -> str:
        chain = self.get_thesis()
        msg = f"Given these requirements, write a thesis\n{self.info}"
        for _ in range(5):
            reply = chain.predict(input=msg)
            if "<Thesis>" in reply:
                with contextlib.suppress(Exception):
                    self.thesis = reply.split("<Thesis>")[1].split("</Thesis>")[0]
                    return self.thesis
            msg = "Please generate a thesis statement\n And wrap it in <Thesis> </Thesis> tags"
        return "Sorry, I am not able to generate a thesis with given information"

    def gen_outline(self):
        chain = self.get_outline()
        section = math.ceil( int(self.word_count)/200)
        msg = f"Given this thesis and information, write an outline, of {section} sections. \n    Thesis: {self.thesis}\n    Information:\n    {self.info}. Outline should keep the world limit in mind."

        for _ in range(5):
            reply = chain.predict(input=msg)
            if "<Outline>" in reply:
                with contextlib.suppress(Exception):
                    outline = reply.split("<Outline>")[1].split("</Outline>")[0]
                    self.outline = re.findall(
                        r"<section>(.*?)</section>", outline, re.DOTALL
                    )
                    return self.outline
            msg = (
                "Please generate an outline\n And wrap it in <Outline> </Outline> tags"
            )
        return "Sorry, I am not able to generate an outline with given information and thesis"

    def gen_essay(self) -> str:
        chain = self.get_essay()
        # go through two sections at a time
        for section in self.outline:
            for _ in range(2):
                with contextlib.suppress(Exception):
                    # example = self.example_selector(section)
                    essay_section = chain.predict(
                        user_input=section,
                        # example=example
                    )
                    essay = re.findall(
                        r"<ESSAY>(.*?)</ESSAY>", essay_section, re.DOTALL
                    )[0]
                    self.sections.append(essay)
                    print(f"AI: {Fore.LIGHTMAGENTA_EX} {essay}\n\n {Style.RESET_ALL}")
                    self.essay += essay + "\n"
                    break
        return self.essay

    def gen_summary(self) -> str:
        llm = ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0)
        # essay = "\n".join(self.sections)
        start_prompt = f"""Using the provided info write an essay of {self.word_count} words\n Info: {self.essay}"""
        prompt_template = """
        {start_prompt}
        ----

        {history}

        ----
        User:{msg}
        AI:
        """
        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["msg", "history", "start_prompt"]
        )

        chain = LLMChain(
            prompt=PROMPT, llm=llm, memory=ConversationBufferMemory(input_key="msg")
        )
        completed = False
        msg = "Please, Start writing the essay"
        while not completed:
            start = chain.predict(msg=msg, start_prompt=start_prompt)
            current_count = len(start.split(" "))
            print(f"AI: {Fore.LIGHTMAGENTA_EX} {start}\n\n {Style.RESET_ALL}")
            print(f"{Fore.GREEN} Current word count: {current_count} {Style.BRIGHT}")
            # if current_count is + or - 10% of the word count, then we are done
            if current_count > int(self.word_count) * 1.1:
                msg = "The essay you have provided is a bit long, please remove some content from it"
            elif current_count < int(self.word_count) * 0.9:
                msg = "The essay you have provided is a bit short, please add some more content to it"
            else:
                completed = True
        return start

    def gen_rewriter(self) -> str:
        prompt = """I want you act an expert in essay writing and rewrite the essay I have written. add flow to it and make it more readable. don't change the meaning of  length should be the same as the original essay. i have been told that the essay is not good and needs to be rewritten. it contains conlusions in places it should not have and has no flow. please rewrite it. KEEP THE WORD COUNT THE SAME. NEVER CHANGE THE WORD COUNT.

        Wrap completed essay in <ESSAY> </ESSAY> tags
        ----
        {essay}
        """
        PROMPT = PromptTemplate(template=prompt, input_variables=["essay"])
        llm = ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=1)
        reply = llm.predict(text=PROMPT.format(essay=self.essay))
        return re.findall(r"<ESSAY>(.*?)</ESSAY>", reply, re.DOTALL)[0]

    def example_selector(self, event: str):
        example_prompt = PromptTemplate(
            input_variables=["input", "output"],
            template="Outline: {input}\nESSAY: {output}",
        )
        example_selector = SemanticSimilarityExampleSelector.from_examples(
            examples, OpenAIEmbeddings(), Chroma, k=2
        )
        ret = example_selector.select_examples({"input": event})
        return "\n".join([example_prompt.format(**r) for r in ret])

    def prevent_ai_detection(self):
        PROMPT = PromptTemplate(template=AI_FIX, input_variables=["content","word_count"])
        llm = ChatOpenAI(model="gpt-4", temperature=1)
        return llm.predict(text=PROMPT.format(content=self.essay, word_count=self.word_count))


if __name__ == "__main__":
    chain = Chains()
    msg_ai, more = chain.gen_info("I want help writing an essay")
    while more:
        print(f"AI: {msg_ai}")
        if "1. Topic" in msg_ai:
            msg = "Please wrap the info in tags <info> </info>"
        else:
            msg = input("User: ")
        msg_ai, more = chain.gen_info(msg)
    print(f"AI: {Fore.CYAN} {msg_ai} {Style.RESET_ALL}")
    print("\n\nGenerating Thesis")
    print(f"AI: {Fore.LIGHTCYAN_EX} {chain.gen_thesis()} {Style.RESET_ALL}")
    print("\n\nGenerating Outline")
    outline = "\n".join(chain.gen_outline())
    print(f"AI: {Fore.CYAN} {outline} {Style.RESET_ALL}")
    print("\n\nGenerating Essay")
    chain.gen_essay()
    print(f"AI: {Fore.LIGHTYELLOW_EX} {chain.essay} {Style.RESET_ALL}\n")
    print(f"\n\nEssay Length: {len(chain.essay.split(' '))}")
    print("Fixing AI detection")
    ai_fix = chain.prevent_ai_detection()
    print(f"AI: {Fore.LIGHTYELLOW_EX} {ai_fix} {Style.RESET_ALL}\n")
    print(f"\n\nEssay Length: {len(ai_fix.split(' '))}")
    # returnn = chain.gen_summary()cp
    # rewrite = chain.gen_rewriter()
    # print(f"AI: {Fore.LIGHTMAGENTA_EX} {rewrite} {Style.RESET_ALL}")
    # print(f"\n\nEssay Length: {len(rewrite.split(' '))}")
    # # print(f"AI: {Fore.LIGHTRED_EX} {returnn} {Style.RESET_ALL}")
    # print("\n\nEssay word count: ", len(returnn.split(" ")))

