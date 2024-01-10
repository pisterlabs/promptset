# from langchain.llms import LlamaCpp
# from llama_cpp import LlamaGrammar
import os

from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain

from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate

from getch import getch

# one_line = LlamaGrammar.from_string('root ::= [a-zA-Z\'",.;:!? ]+ "\n"', verbose=False)

# fmt: off

llm_story = ChatPromptTemplate.from_messages([
    ("system", """
You are writing a thrilling text adventure game set in a sci-fi universe.
You are a dungeon master that leads the players through the story step by step.
Give detailed descriptions of the current environment and situation. Don't leave things vague.
Don't write about how player feels about things.
Use at most three paragraphs.
"""),
    ("human", "{text}")]) | ChatOpenAI(model_kwargs={"stop": [">"]})


llm_options = ChatPromptTemplate.from_messages([
    ("system", """
This is a text adventure game set in a sci-fi universe.
Add a player action option on what to do next at the end of text.
Use at most three sentences.
Make the option unique and interesting.
Don't use word "you" in the option, instead use a command style like for example "go north".
"""),
    ("human", "{text}")]) | ChatOpenAI(model_kwargs={"stop": ["\n"]})


llm_compress = ChatPromptTemplate.from_messages([
    ("system", """Summarize the text in a few paragraphs. Keep all the relevant items, locations and actors."""),
    ("human", "{text}")
]) | ChatOpenAI()

# fmt: on

# llm = ChatOpenAI(model_name="gpt-3.5-turbo", max_tokens=200)


class PromptManager:
    def __str__(self):
        return self.text

    def __init__(self, prt=""):
        assert isinstance(prt, str)
        self.text = prt

    def add(self, text):
        assert isinstance(text, str)
        self.text += text
        print(text, end="")

    def set(self, text):
        assert isinstance(text, str)
        self.text = text

    def get(self):
        return self.text


def gen_options(prompt):
    assert len(prompt.get()) > 10, "Prompt length too short."
    prompt.add("\n\n")
    print("(gen_options)")
    for n in range(4):
        result = ""
        prompt.add(f"{n+1}. ")
        for _ in range(5):
            result = llm_options.invoke({"text": prompt.get()}).content.strip()
            if len(result) > 5:
                break
        prompt.add(result + "\n")
    # print(prompt.get().split("\n")[-10:])
    omap = {}
    for line in prompt.get().split("\n")[-10:]:
        if len(line) > 5 and line[0] in "1234":
            omap[line[0]] = line[3:]
    options = []
    # print(len(omap))
    for i in range(1, 5):
        if str(i) in omap:
            options.append(omap[str(i)])
    assert len(options) > 1
    # print(options)
    # print(len(options))
    return options


previous_choices = []
checkpoint = 0
# prompt = PromptManager(header + start_text)
prompt = PromptManager("")
while True:
    start_prompt = prompt.get()
    print("(gen story)")
    prompt.add(llm_story.invoke({"text": start_prompt}).content.strip())

    checkpoint = prompt.get()
    options = gen_options(prompt)
    prompt.set(checkpoint)

    print(f"\n> ", end="")
    prompt.text += "\n\n> "

    # Read user choice
    while (choice := getch()) not in "123459cq":
        pass

    if choice == "q":
        print("-----")
        print(prompt.get())
        break

    if choice == "5":
        prompt.add(input("5. ") + "\n\n")

    if choice in "4321":
        prompt.add(options[int(choice) - 1] + "\n\n")

    if choice == "9":
        print("(REGEN)\n")
        prompt.set(start_prompt)

    if choice == "c":
        # calculate number of ">" in the prompt
        print("(COMPRESS)\n")
        print(llm_compress.invoke({"text": start_prompt}).content)
        prompt.set(start_prompt)
        # if prompt.get().count(">") > 2:
        #     # Second half of the context, starting from ">"
        #     pt = start_prompt
        #     ptl = len(pt)
        #     cutoff = ptl // 2 + pt[ptl // 2].find(">")
        #     part_second = pt[cutoff:]
        #     part_first = pt[:cutoff]

        #     print("(SUMMARY)\n")
        #     compressed = llm_compress.invoke({"text": part_first}).content.strip()
        #     prompt.set(compressed + part_second)
        #     print(prompt.get())
        #     print("-----\n")
