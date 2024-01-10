from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Type

import langchain
from langchain import LLMChain, LLMMathChain, PromptTemplate
from langchain.cache import InMemoryCache
from langchain.chains import SequentialChain
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from adventure.utils import get_model, game_print_debug, game_print
from adventure.utils import run_with_attempts, verbose


def get_room_by_type(room_type: str) -> Type[GeneralRoom] | Type[MathRoom] | None:
    return {
        "general": GeneralRoom,
        "math": MathRoom,
    }.get(room_type, None)


NUM_ATTEMPTS_PER_ROOM = 10


class RoomLoopAnswer(BaseModel):
    """Room Loop Answer"""
    state: str = Field(
        description="The current state. Must be one of the valid states")
    similarity: float = Field(
        description="The similarity of the main idea of the player's input to the main idea of the answer "
                    "as a value between 0 and 1")
    reply: str = Field(
        description="Your reply to player. Does not contain answer")
    new_state: str = Field(
        description="The new state. Must be one of the valid states")


RoomLoopAnswerParser = PydanticOutputParser(pydantic_object=RoomLoopAnswer)


class Riddle(BaseModel):
    """A riddle that need to be solved."""
    riddle: str = Field(description="the riddle as a question")
    answer: str = Field(description="the answer to the riddle")


RiddleParser = PydanticOutputParser(pydantic_object=Riddle)


class Room(ABC):
    _riddle_generator: LLMChain = None
    """The riddle generator for this room. Generates riddle and answer"""

    _room_chain: LLMChain = None
    """The chain for this room that will be used to check 
    answers for the riddle. Isn't going to be a Conversation, 
    cause we actually don't need memory."""

    room_config: dict = None
    """The config for this room. Contains the topic and the room type"""

    _riddle: dict = None  # {"riddle": str, "answer": str}

    ADVENTURE_GAME_ROOM_PROMPT_TEMPLATE = """
    You are following the following rules:
    - You are playing the question/answer game.
    - You must not refer to yourself in any way. Only to role.
    - You play the role an expert in {topic}.
    - You are talking to a player.
    - You can't pretend a player.
    - Your task is to check the correctness of the player's input or answer the question.    
    - The topic of the conversation is {topic}.
    - You can only reply on questions that are asked in the context of the game.
    {room_specific_instructions}
    - Be short and precise in your replies.
    - Don't provide answer. You can only give clues or hints.
    - Your reply should not contain words delimited by triple backticks: ```{answer}```.
    
    - Precise Task 1: Based on the current state and players' input, choose the next state from available states. 
    The following states are available:
    {states}
    - Precise Task 2: Based on the current state description and players' input, generate Your reply to player .
    """

    def _check_riddle_instructions(self):
        raise NotImplementedError

    @property
    def states_str(self):
        return f"""
        * "start_game": 1. Introduce yourself and the topic of conversation. 2. Give player the riddle
        * "guessing_riddle": {self._check_riddle_instructions()} 
        * "finished": you should stop the game
        """

    # Used to be for memory
    history_concatenation = """Current conversation:
    The current state is "{state}"
    Player's input delimited by triple backticks: ```{input}```
    """

    def __init__(self, room_config: dict):
        self.room_config = room_config

    @property
    def riddle(self):
        if not self._riddle:
            raise ValueError("Riddle is not generated yet")
        return self._riddle["riddle"]

    @property
    def answer(self):
        if not self._riddle:
            raise ValueError("answer is not generated yet")
        return self._riddle["answer"]

    @abstractmethod
    def _get_riddle_generator(self) -> LLMChain:
        raise NotImplementedError

    @property
    def riddle_generator(self) -> LLMChain:
        if not self._riddle_generator:
            self._riddle_generator = self._get_riddle_generator()
        return self._riddle_generator

    @abstractmethod
    def generate_riddle(self):
        raise NotImplementedError

    @property
    def room_prompt_template(self):
        raise NotImplementedError

    def _get_room_chain_from_params(self) -> LLMChain:
        # TODO: check if this is needed
        langchain.llm_cache = InMemoryCache()
        vbs = verbose()

        model_name = "OpenAI"
        # model_name = "ChatOpenAI"  # TODO: adopt code to chat models

        # model_name = "Replicate"
        # model_name = "Cohere"
        # model_name = "HuggingFace_google_flan"
        # model_name = "HuggingFace_mbzai_lamini_flan"
        # model_name = "Local_gpt2"
        # model_name = "Local_lama"
        llm = get_model(model_name, temperature=0.1)

        prompt = PromptTemplate(
            template=self.room_prompt_template,
            input_variables=["input", "state"],
            partial_variables={"format_instructions": RoomLoopAnswerParser.get_format_instructions()},
        )

        # We don't need memory in this case, although technically It's a conversation
        conversation = LLMChain(
            llm=llm,
            prompt=prompt,
            # memory=memory,
            verbose=vbs
        )
        return conversation

    def _get_room_chain(self) -> LLMChain:
        self.generate_riddle()
        game_print_debug(f"Debug ::: Riddle :: {self._riddle}")
        return self._get_room_chain_from_params()

    @property
    def room_chain(self) -> LLMChain:
        if not self._room_chain:
            self._room_chain = self._get_room_chain()
        return self._room_chain

    @run_with_attempts
    def sub_loop(self, p_input, state):
        game_print_debug(f"chain input: {p_input}, state: {state}")
        repl = self.room_chain.run(input=p_input, state=state)
        game_print_debug(f"chain repl: {type(repl)} : {repl}")
        return RoomLoopAnswerParser.parse(repl).dict()

    def loop(self) -> int:
        game_print_debug("Debug::: Room Loop Started")
        hp = self.room_config["hp"]
        topic = self.room_config["topic"]
        game_print(f"Are you ready to talk about {topic}?")
        input(f"You [{' ♥ ' * hp}] : >>>")

        rpl = self.sub_loop("Introduce yourself and give me a riddle.", "start_game")
        game_print(f"Expert: {rpl['reply']}")

        for attempt in range(1, NUM_ATTEMPTS_PER_ROOM+1):
            power = NUM_ATTEMPTS_PER_ROOM - attempt
            rpl = self.sub_loop(input(f"You [{' ♥ '* hp }] [{' ⚡ '* power }]: >>>"), rpl["new_state"])

            if rpl["new_state"] == "finished":
                game_print(f"Expert: {rpl['reply']}")
                return attempt

            if float(rpl["similarity"]) > 0.65:
                game_print("Your guess is almost correct!")
                game_print(f"The answer is {self.answer}")
                return attempt

            game_print(f"Expert: {rpl['reply']}")

        game_print(f"Expert: The number of attempts is exceeded. The answer is {self.answer}")


class GeneralRoom(Room):
    GENERAL_ROOM_SPECIFIC_INSTRUCTIONS = """
    - The player is trying to guess a riddle.
    - You have a riddle that player need to find the solution to. The riddle: "{riddle}". 
    - The riddle is related to {topic}.
    - Compare the player's input with the answer. The correct answer is "{answer}".
    """

    def _get_riddle_generator(self) -> LLMChain:
        sys_prompt = """
        You are a world class algorithm for generating riddles. 
        Your task is to generate a riddle about the topic delimited by triple backticks: ```{topic}```.
        Your knowledge of {topic} should be used to generate riddle.

        Hint: The riddle should be short.
        Hint: The riddle should not contain the answer.
        """
        vbs = verbose()
        llm = get_model("OpenAI", temperature=0.8)

        prompt = PromptTemplate(
            template=sys_prompt + "\n{format_instructions}\n",
            input_variables=["topic"],
            partial_variables={"format_instructions": RiddleParser.get_format_instructions()},
        )
        return LLMChain(llm=llm, prompt=prompt, verbose=vbs)

    @run_with_attempts
    def generate_riddle(self):
        repl = self.riddle_generator.run(topic=self.room_config["topic"])
        game_print_debug(f"chain repl: {type(repl)} : {repl}")
        self._riddle = RiddleParser.parse(repl).dict()

    def _check_riddle_instructions(self):
        return ". ".join([
            "If player makes a guess, compare player's input with the answer. ",
            "If player asked a question, answer it by providing a clue",
            'if player guessed correctly. The next state is "finished"'
        ])

    @property
    def room_prompt_template(self) -> str:
        system_prompt = self.ADVENTURE_GAME_ROOM_PROMPT_TEMPLATE.format(
            topic=self.room_config["topic"],
            states=self.states_str,
            answer=self.answer,
            room_specific_instructions=self.GENERAL_ROOM_SPECIFIC_INSTRUCTIONS.format(
                riddle=self.riddle,
                answer=self.answer,
                topic=self.room_config["topic"],
            )
        )

        template = system_prompt + self.history_concatenation + "\n{format_instructions}\n"
        game_print_debug(f"Room Prompt Template: {template}")
        return template


class MathRoom(Room):
    MATH_ROOM_SPECIFIC_INSTRUCTIONS = """
    - The player is trying to solve a math problem.
    - You have a math problem that player need to solve. The math problem: "{riddle}".
    - You have a correct answer to the math problem. The correct answer is "{answer}". It's a number.
    - You should use the term "riddle", talking about math problem.
    """

    def _get_riddle_generator(self):
        math_question_prompt = """
        You are a world class algorithm for generating math questions. 
        Your task is to generate a math question that requires calculation.

        Hint: The question should not contain the answer. 
        {input}
        """
        vbs = verbose()
        # model_name = "Replicate"  # Provides more interesting questions
        model_name = "OpenAI"  # ALso provides interesting questions
        # but now configured to provide debug simple questions
        q_llm = get_model(model_name, temperature=0.8)

        prompt = PromptTemplate(template=math_question_prompt, input_variables=["input"])
        q_chain = LLMChain(llm=q_llm, prompt=prompt, output_key="question", verbose=vbs)

        m_llm = get_model("OpenAI", temperature=0)
        m_chain = LLMMathChain.from_llm(llm=m_llm, verbose=vbs)

        overall_chain = SequentialChain(
            chains=[q_chain, m_chain],
            input_variables=["input"],
            output_variables=["question", "answer"],
            verbose=vbs
        )

        return overall_chain

    @run_with_attempts
    def generate_riddle(self):
        repl = self.riddle_generator({"input": "Generate a math question"})
        game_print_debug(f"chain repl: {type(repl)} : {repl}")
        self._riddle = {"riddle": repl["question"], "answer": repl["answer"]}

    def _check_riddle_instructions(self):
        return ". ".join([
            'Get the number from the answer.'
            "If player makes a guess, consider player's input as number and compare it with the number from the answer",
            "If player asked a question, answer it by providing a clue",
            'if player guessed correctly. The next state is "finished"',
        ])

    @property
    def room_prompt_template(self) -> str:
        system_prompt = self.ADVENTURE_GAME_ROOM_PROMPT_TEMPLATE.format(
            topic=self.room_config["topic"],
            states=self.states_str,
            answer=self.answer,
            room_specific_instructions=self.MATH_ROOM_SPECIFIC_INSTRUCTIONS.format(
                riddle=self.riddle,
                answer=self.answer
            )
        )

        template = system_prompt + self.history_concatenation + "\n{format_instructions}\n"
        return template
