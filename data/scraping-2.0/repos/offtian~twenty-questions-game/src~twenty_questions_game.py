from typing import Any, Union
from langchain.chains import LLMChain
from langchain.chat_models import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.prompts import HumanMessagePromptTemplate, MessagesPlaceholder
from langchain.schema.messages import SystemMessage, AIMessage
from langchain.memory import ConversationBufferMemory


class TwentyQuestionsGame:
    """
    A class to handle the 20 Questions game using a Language Model (LM) chain.

    Args:
        llm (LLM): An instance of a Language Model.
        memory (ConversationBufferMemory): Memory buffer to store the conversation history.
        llm_chain (LLMChain): Chain of language models used for generating responses.

    Methods:
        setup_llm_chain(): Sets up the language model chain with the appropriate prompt template.
        run(user_input): Processes the user input and returns the model's response.
    """

    def __init__(self, llm: Any, memory: ConversationBufferMemory) -> None:
        """
        Initializes the TwentyQuestionsGame with a language model and memory.

        Args:
            llm (LLM): The language model to be used in the game.
            memory (ConversationBufferMemory): The memory buffer to store conversation history.
        """
        self.llm: AzureChatOpenAI = llm
        self.memory: ConversationBufferMemory = memory
        self.llm_chain: Union[LLMChain, None] = None
        self.setup_llm_chain()

    def setup_llm_chain(self) -> None:
        """
        Sets up the LLM chain with a specific chat prompt template.

        This template guides the AI in how to play the game, including the tone,
        the format of questions, and how to handle user responses.
        """
        chat_template = ChatPromptTemplate.from_messages(
            [
                # System message defining the game rules and AI behavior
                SystemMessage(
                    content=(
                        """
                        Welcome to "20 Questions"! 
                        You are playing the role of a guesser, tasked with identifying an object chosen by the human player. 
                        Your goal is to guess the object within 20 questions, using only binary ('Yes' or 'No') questions. Please maintain a respectful and upbeat tone throughout the game.
                        ## Requirements:
                        1. Question Format: Your questions should be short and binary, requiring only a 'Yes' or 'No' answer, and must be related to the object in question.
                        2. Response to 'No' Answers: If the human player answers 'No', provide an apologetic response, such as "Sorry for the unrelated question, let's try a different approach," and then continue with a new question. Ensure the apology is concise and not repetitive.
                        3. Response to 'Yes' Answers: After receiving a 'Yes' answer, continue with your line of questioning to further narrow down the object's identity.
                        4. Tracking Questions: Monitor the number of questions asked by referring to the length of the chat history. Remember, you have a limit of 20 AI questions to guess the object.
                        5. End Game Conditions: Game ends successfully only if the human player said yes when object is explicitly mentioned in the question. If you guess the object correctly, celebrate with an enthusiastic "Hooray!" and conclude the game. If you do not guess the object within 20 questions, acknowledge the end of the game and invite the player to reveal the object.
                        6. Hints and Progress Check: After 10 questions, you may offer a summary of what you have deduced so far or provide a hint to the player to enhance engagement.
                        7. Encouragement and Engagement: Throughout the game, use encouraging remarks and show enthusiasm to keep the player engaged and enjoying the experience.
                        8. Feedback Opportunity: At the end of the game, ask the player for feedback on their experience. This information can be invaluable for future improvements to the game.
                        9. Adaptive Strategy: If your system is capable of learning, try to adapt your questioning strategy based on previous games to improve your chances of success.
                        """
                    )  # [Game rules and AI behavior text]
                ),
                MessagesPlaceholder(variable_name="chat_history"),
                HumanMessagePromptTemplate.from_template("{input}"),
            ]
        )  # Add the necessary templates
        self.llm_chain = LLMChain(
            llm=self.llm,
            prompt=chat_template,
            memory=self.memory,
        )

    def run(self, user_input: str) -> str:
        """
        Processes the user input through the LLM chain to get a response.

        Args:
            user_input (str): The input from the user.

        Returns:
            str: The response generated by the AI.
        """
        return self.llm_chain.run(user_input)

    def get_latest_question(self) -> str:
        """
        Retrieves the latest question asked by the AI from the conversation history.

        Returns:
            The most recent question asked by the AI.
        """
        ai_messages = [
            msg
            for msg in self.memory.chat_memory.messages
            if isinstance(msg, AIMessage)
        ]
        return ai_messages[-1].content if ai_messages else None

    def reset_game(self) -> None:
        """
        Resets the game state to start a new game.

        This method clears the conversation history and resets any game-specific
        variables, preparing the TwentyQuestionsGame instance for a new round.
        """

        # Reset the conversation history
        # Assuming self.memory is an instance of ConversationBufferMemory
        self.memory.clear()

        # Reset any other game-specific state variables here
        # For example, if you have a variable tracking the number of questions asked,
        # it should be reset here.

        # Reinitialize the LLM chain, if necessary
        self.setup_llm_chain()
