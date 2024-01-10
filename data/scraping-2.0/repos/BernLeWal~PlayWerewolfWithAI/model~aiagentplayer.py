"""
Representing a player in a Werewolve game
"""
import asyncio
import logging

from agents.openai_agent import OpenAIAgent

from model.player import Player
from model.command import VoteCommand
from logic.context import Context


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(levelname)s:%(name)s: %(message)s')
logger = logging.getLogger(__name__)



class AIAgentPlayer(Player):
    """Representing a AI-agent player"""
    def __init__(self, name :str, game :Context, bot) ->None:
        super().__init__(name)
        self.message_queue = asyncio.Queue()
        self.agent = OpenAIAgent()
        self.bot = bot
        self.game = game
        logger.info("Created AIAgentPlayer with name %s", self.name)

        self.__current_channel_id__ = -1
        self.__current_messages__ : str = ""

        self.__tasks__ = None
        self.__stopped__ = False

    def __del__(self) ->None:
        self.stop()
        logger.info("Deleted AIAgentPlayer named %s", self.name)


    async def send_dm(self, msg :str) ->None:
        """Sends a direct message to the player"""
        logger.info("Sent DM '%s' to AIPlayer %s", msg, self.name)
        self.agent.advice( msg, None )

    async def __worker_task__(self):
        try:
            while not self.__stopped__:
                # Get message from the queue and process it
                (channel_id, author_name, message) = await self.message_queue.get()
                logger.debug("Worker processing: channel_id=%d %s:%s",
                             channel_id, author_name, message)

                if author_name == "ModeratorBot":
                    if message == "!quit":
                        self.__stopped__ = True
                        break
                    await self.bot.get_channel(channel_id).send(
                        f"**{self.name}**: {await self.agent.ask_async(message)}")
                elif author_name == "$$TimerTask$$":
                    if not self.__current_channel_id__ == -1:
                        await self.__send___current_messages____()
                    if not self.is_dead:
                        if self.game.is_werewolf_vote_needed(self):
                            await self.__check_werewolf_vote__()
                        if self.game.is_villager_vote_needed(self):
                            await self.__check_villager_vote__()
                        if self.game.is_seer_vote_needed(self):
                            await self.__check_seer_vote__()
                else:
                    if self.__current_channel_id__ == -1:
                        # It's a new message
                        self.__current_channel_id__ = channel_id
                        self.__current_messages__ = f"{author_name}: {message}\n"
                    elif self.__current_channel_id__ == channel_id:
                        # It adds to the current collaboration
                        self.__current_messages__ += f"{author_name}: {message}\n"
                    else:
                        # The collaboration moves to a different channel
                        # --> send current collab to agent now
                        await self.__send___current_messages____()
                        self.__current_channel_id__ = channel_id
                        self.__current_messages__ = f"{author_name}: {message}\n"
        except asyncio.CancelledError:
            logger.warning("WorkerTask of %s cancelled!", self.name)

    async def __send___current_messages____(self) ->None:
        if self.__current_channel_id__ > -1 and len(self.__current_messages__)>0:
            logger.info("Send to LLM: %s", self.__current_messages__)
            self.agent.advice(
                "What did the other players say lately?",
                self.__current_messages__
            )
            self.__current_messages__ = ""
            prompt = "Take part of the recent conversation or give answer."
            await self.bot.get_channel(self.__current_channel_id__).send(
                f"**{self.name}**: {await self.agent.ask_async(prompt)}")
            self.__current_channel_id__ = -1

    async def __check_werewolf_vote__(self) ->None:
        vote = await self.agent.ask_async(
            "As a werewolf you need to vote for a victim together with the other werewolves."
            "Decide for a victim and answer this time with just one word - the player name!"
            )
        vote = self.__fix_ai_vote__(vote)
        logger.info("AIAgentPlayer %s (a werewolf) sent the following vote decision:'%s'",
                    self.name, vote)
        if vote in self.game.players:
            await self.game.handle( VoteCommand(self.bot.user, self.name, vote))

    async def __check_villager_vote__(self) ->None:
        vote = await self.agent.ask_async(
            "You need to vote for a victim together with the others."
            "Decide for a victim and answer this time with just one word - the player name!"
            )
        vote = self.__fix_ai_vote__(vote)
        logger.info("AIAgentPlayer %s sent the following vote decision:'%s'",
                    self.name, vote)
        if vote in self.game.players:
            await self.game.handle( VoteCommand(self.bot.user, self.name, vote))

    async def __check_seer_vote__(self) ->None:
        vote = await self.agent.ask_async(
            "As the seer you are allowed to ask if one player is a werewolf."
            "Decide for a player and answer this time with just one word - the player name!"
            )
        logger.info("AIAgentPlayer %s (a seer) sent the following vote decision:'%s'",
                    self.name, vote)
        vote = self.__fix_ai_vote__(vote)
        if vote in self.game.players:
            await self.game.handle( VoteCommand(self.bot.user, self.name, vote))

    def __fix_ai_vote__(self, vote:str) ->str:
        vote = vote.strip()
        if vote.endswith('.') or vote.endswith('!'):
            vote = vote[:-1]
        return vote

    async def __timer_task__(self):
        try:
            while not self.__stopped__:
                await asyncio.sleep(60)   # produce a reminder every minute
                await self.message_queue.put( (-1, "$$TimerTask$$", "SendCurrentMessages") )
        except asyncio.CancelledError:
            logger.warning("TimerTask cancelled")


    async def start(self) ->None:
        """Start the worker thread"""
        self.__stopped__ = False
        self.__tasks__ = await asyncio.gather(self.__timer_task__(), self.__worker_task__())


    async def stop(self) ->None:
        """Stop the worker threads"""
        self.__stopped__ = True
        await self.add_message(-1, "ModeratorBot", "!quit")
        for task in self.__tasks__:
            #if not task.done():
            #    task.cancel()
            task.join()


    async def init(self) ->None:
        """Initialize the context for the LLM"""
        self.agent.system(
            "You are a player of the famous card game 'Werwölfe vom Düsterwald'."
            "You will play together with the werewolves team or with the villagers team, "
            "depending on the card you get."
            "Be curios, be funny, make jokes."
            f"Your name is {self.name}."
            "Do not use more than five sentences in your responses!"
        )


    async def add_message(self, channel_id :int, author_name :str, message :str) ->None:
        """Put a message in the queue"""
        logger.info("Inform AI-Player %s about message '%s:%s'", self.name, author_name, message)
        await self.message_queue.put( (channel_id, author_name, message) )
