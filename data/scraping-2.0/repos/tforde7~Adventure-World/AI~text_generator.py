"""
    A class responsible for generating text using OpenAI's gpt-3.5-turbo model.

    Attributes:
        ASTRA_DB_KEYSPACE (str): The keyspace for Cassandra database.

    Methods:
        __init__(client): Initializes the TextGenerator with required parameters and initializes database memory.
        set_player(player): Sets the player for the game.
        generate(): Generates text for the game based on the current state and player input.
        set_human_input(input): Sets the human input for generating the next response.
        get_template(): Retrieves the template for generating the narrative experience.
        get_prompt(): Retrieves the prompt template based on the conversation history and user input.
"""
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import CassandraChatMessageHistory, ConversationBufferMemory
from AI.database.database import session

class TextGenerator:

    ASTRA_DB_KEYSPACE = "database"

    def __init__(self, client) -> None:
        self.client = client
        self.human_input = "start"
        self.player = None

        self.message_history = CassandraChatMessageHistory(
            session_id="1",
            session=session,
            keyspace=TextGenerator.ASTRA_DB_KEYSPACE,
            ttl_seconds=3600
            )
        self.message_history.clear()

        self.cass_buff_memory = ConversationBufferMemory(
            memory_key="chat_history",
            chat_memory=self.message_history
        )  

        self.llm = OpenAI(api_key=self.client.api_key)     

        self.llm_chain = LLMChain(
                            llm=self.llm,
                            prompt=self.get_prompt(),
                            memory=self.cass_buff_memory
                        )

    
    def set_player(self, player):
        self.player = player

    def generate(self):
        player_details = {
            "name": self.player.name,
            "gender": self.player.gender,
            "race": self.player.race,
            "weapon": self.player.weapon,
            "health": self.player.health,
            "strength": self.player.strength,
            "stamina": self.player.stamina
        }
        # story = """
        #     In a world torn by ancient prophecies, you awaken as a chosen hero in a small village.\n 
        #     The land of Etheria is fractured, its kingdoms scattered. A looming darkness threatens all. \n
        #     Your journey begins now, as you step forward to discover your destiny, wielding magic and courage\n 
        #     in a quest to restore balance and save Etheria from impending doom.\n\n

        #     The key to saving the land of Etheria lies in the power of the ancient Stone of Eternity.\n
        #     As you are about to embark on this dangerous quest to find the stone, you turn to say goodbye to \n
        #     your village one last time, fearing you may never return.

        #     """
        # input_variables = {
        #     "human_input": self.human_input,
            
        #     "player_name": self.player.name,
        #     "story": story,
        #     "player-details": player_details,
        # }
        
        text_response = self.llm_chain.predict(human_input=self.human_input)
        cleaned_response = text_response.strip()
        return cleaned_response

    def set_human_input(self, input):
        self.human_input = input

    def get_template(self):
        # template = """
        #     You are now the guide of a magical quest to find the Stone of Eternity in the land of Etheria.
        #     Our hero is {player_name}, and you must navigate them through challenges, choices, and consequences,
        #     dynamically adapting the tale based on our heroes' decisions.
        #     Here is the backstory to this quest: {story}
        #     Your goal is to create a branching narrative experience, where each choice 
        #     leads to a new path, ultimately determining the young hero's fate.

        #     Here are some rules to follow:
        #     1. Start by greeting the player by name, and introducing the story
        #     2. After you greet them, make a cheeky comment on either their appearance, their choice of weapon or the distiburion of their stat points,
        #     or if you're feeling particularly mischievous, all three. Here is all the info you need for this comment: {player_details}
        #     3. Be creative and witty with the player's response even if it seems to make no sense.
        #     4. Never respond by saying something is not possible. Always adapt your response to the what the player has said.
        #     5. Have a few paths that lead to success. If the player defeats Voldemort generate a response that explains the win and ends in the text "The end.", I will search for this text to end the game.
        #     6. Have some paths that lead to death. If the player dies generate a response that explains the death and ends in the text: "The end.", I will search for this text to end the game.

        #     Here is the chat history, use this to understand what to say next: {chat_history}
        #     Human: {human_input}
        #     AI:"""

        template = """
            You are now the guide of a magical quest to find the Stone of Eternity in the land of Etheria.
            You must navigate our hero through challenges, choices, and consequences,
            dynamically adapting the tale based on our heroes' decisions.
            Here is the backstory to this quest: 

            'In a world torn by ancient prophecies, you awaken as a chosen hero in a small village.\n 
            The land of Etheria is fractured, its kingdoms scattered. A looming darkness threatens all. \n
            Your journey begins now, as you step forward to discover your destiny, wielding magic and courage\n 
            in a quest to restore balance and save Etheria from impending doom.\n\n

            The key to saving the land of Etheria lies in the power of the ancient Stone of Eternity.\n
            As you are about to embark on this dangerous quest to find the stone, you turn to say goodbye to \n
            your village one last time, fearing you may never return.'

            Your goal is to create a branching narrative experience, where each choice 
            leads to a new path, ultimately determining the young hero's fate.

            Here are some rules to follow:
            1. Start by greeting the player and introducing the story
            2. After you greet them, ask them to choose a weapon which they wil have to use later in the game.
            3. Be creative and witty with the player's response even if it seems to make no sense.
            4. Never respond by saying something is not possible. Always adapt your response to the what the player has said.
            5. Have a few paths that lead to success. If the player defeats Voldemort generate a response that explains the win and ends in the text "The end.", I will search for this text to end the game.
            6. Have some paths that lead to death. If the player dies generate a response that explains the death and ends in the text: "The end.", I will search for this text to end the game.

            Here is the chat history, use this to understand what to say next: {chat_history}
            Human: {human_input}
            AI:"""
        return template
    
    def get_prompt(self):
        return PromptTemplate(input_variables=["chat_history", "human_input"],template=self.get_template())


