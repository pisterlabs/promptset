from src.custom_logger import CustomLogger
from .character_data import CharacterDetails
from .agent_memory_manager import AgentMemoryManager
from .agent_memory.agent_memory import AgentMemory
from .agent_memory.generative_memory import GenerativeAgentMemory
from .agent_memory.memory import MemoryEntry
from .decision_making.mood_analyzer import MoodAnalyzer
from .decision_making.decision_processor import DecisionProcessor
from .decision_making.thread_decorator import threaded
from .openai_helpers.chat_completion import chat_completion
from dotenv import load_dotenv

import time
import threading
import datetime
import textwrap
import os
import openai
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")


class Character:
  """ A character with personal data, memories, and decision-making capabilities. """

  def __init__(self, name: str, bio: str, abilities: str, memories: str, traits: str, initial_location: str = 'club room') -> None:
    """
    Initialize the Character instance with personal data and memories.

    Parameters
    ----------
    name : str
      The name of the character.

    bio : str
      The biography of the character.

    abilities : str
      The abilities or skills of the character.

    memories : str
      The memories of the character, separated by semicolons.

    traits : str
      The traits of the character.

    initial_location : str, optional
      The initial location of the character, by default 'club room'.
    """
    self._memory_db = AgentMemoryManager(name, 'json')

    self._character_data = CharacterDetails(name, bio, traits, abilities, initial_location)

    self._logger = CustomLogger(self._character_data)

    memories = [memory.strip() for memory in memories.split(';')]

    self._agent_memory = AgentMemory(memories, self._character_data, self._logger, self._memory_db)

    self.character_data.status = self._memory_db.get_agent_status()

    if self.character_data.status is None:
      self.character_data.status = "Monika have afraid and doesn't know what is happening, she is trying to figure out what is happening."

    self._mood_analyzer = MoodAnalyzer(self._character_data, self._logger)

    self._conversation_history = ''

    initial_time = time.time()

    self._decision_processor = DecisionProcessor(self._logger, self._agent_memory, self._character_data)

    self._generative_memory = GenerativeAgentMemory(self._character_data, self._agent_memory, self._logger)

    self._character_data.bio = self._generate_bio()

    self._generate_bio_thread = threading.Thread(target=self._generate_bio, daemon=True, name='Generate Bio Thread')

    self._status_thread = threading.Thread(target=self._generate_status, daemon=True, name='Generate Status Thread')

    if self._agent_memory._is_initial_run:
      self._generative_memory.generate_reflections()

    self._logger.agent_info(f'Finished initializing character in {time.time() - initial_time} seconds')

  @property
  def character_data(self) -> CharacterDetails:
    return self._character_data

  @property
  def memories(self) -> list[MemoryEntry]:
    return self._agent_memory.memories

  def _generate_bio(self) -> str:
    """
    Generate the biography of the character based on its memories and personal data.

    Returns
    -------
    str
      The generated biography of the character.
    """
    self._logger.agent_info('Generating bio...')

    questions = (
      f'Key features of {self._character_data.name}, what makes it unique.',
      f'Current daily occupation of {self._character_data.name}.',
      f'How is {self._character_data.name} feeling about their recent progress in life.'
    )

    prompts = (
      textwrap.dedent("""
      How would one describe the key features of {} given the following statements?
      Use a maximum of 120 words. Include only the summary, do not add a title or the like.

      Only use the information provided below:
      {}
      """),
      textwrap.dedent("""
      How would one describe the daily occupation of {} given the following statements?
      Use a maximum of 120 words. Include only the summary, do not add a title or the like.

      Only use the information provided below:
      {}
      """),
      textwrap.dedent("""
      How would one describe the recent progress in {}'s life given the following statements?
      Use a maximum of 120 words. Include only the summary, do not add a title or the like.

      Only use the information provided below:
      {}
      """)
    )

    @threaded
    def generate_summary(args) -> str:
      (prompt, question) = args
      memories = self._agent_memory.retrieve(question)

      list_of_memories = '\n'.join([f'- {memory.access()}.' for memory in memories])

      summary, _ = chat_completion(prompt.format(self._character_data.name, list_of_memories))

      self._logger.agent_info(f'Generated summary for > {question}\nSummary: {summary}')

      return summary

    threads = [generate_summary((prompt, question)) for prompt, question in zip(prompts, questions)]

    summaries = [result for result in threads]

    new_description = textwrap.dedent("""
    You are a person named {}.
    Your bio is the following:
    {}
    
    Your abilities are the following:
    {}
    
    Your traits are the following:
    {}    
    """).format(self._character_data.name, '\n\n'.join(summaries), self._character_data.abilities, self._character_data.traits)

    self._logger.agent_info(f'Generated bio: {new_description}')

    return new_description

  def _generate_status(self) -> str:
    """
    Generate the current status of the character based on recent memories.

    Returns
    -------
    str
      The generated status of the character.
    """
    self._logger.agent_info('Generating status...')

    recent_memories = self._agent_memory.memories[:30]

    list_of_memories = '\n'.join([f'{i + 1}. {memory.access()}' for i, memory in enumerate(recent_memories)])

    prompt = textwrap.dedent("""
    Information (Records):
    {}

    What would be the current emotional state of {} based on the statements above?
    Use a maximum of 10 words and follow the format below.

    The result should be in the third person, specifying who the person being referred to is.

    Format:
    Status: <FILL IN>
    """).format(list_of_memories, self._character_data.name)

    new_status, _ = chat_completion(prompt)

    new_status = new_status.split(':')[1].strip()

    self._logger.agent_info(f'Generated new_status: {new_status}')

    self._memory_db.set_agent_status(new_status)

    return new_status

  def chat(self, speaker: str, message: str) -> list[list[str]]:
    """
    Engage in a conversation with the character, processing the speaker's message.

    Parameters
    ----------
    speaker : str
      The name of the speaker engaging with the character.

    message : str
      The message or statement made by the speaker.

    Returns
    -------
    tuple(str, str)
      The response and pose of the character.
    """
    initial_time = time.time()

    if len(self._agent_memory.memories) % 40 == 0:
      self._generate_bio_thread.start()

    self._conversation_history += f'{speaker}: {message.strip()}\n'

    @threaded
    def generate_speaker_action(speaker: str, speaker_message: str) -> str:
      return self._decision_processor.determine_speaker_action(speaker, speaker_message)

    @threaded
    def generate_observation(speaker: str, conversation_history: str) -> str:
      return self._decision_processor.generate_observation(speaker, conversation_history)

    speaker_action = generate_speaker_action(speaker, message)
    observation = generate_observation(speaker, self._conversation_history)

    questions = [f'What is the relationship between {self._character_data.name} and {speaker}?', speaker_action]

    memory_summaries = self._decision_processor.generate_memory_summaries(questions)

    posible_action = self._decision_processor.determine_possible_action(observation, memory_summaries)

    self._logger.agent_info(f'Generating response...')

    prompt = textwrap.dedent("""
    Current Date: {}
    {} State: {}
    Current Location: {}

    Observation:
    {}

    Summary of {} and their relationship with {}:
    {}

    Possible action to take:
    {}
    
    What should {} say? Remember to use only the information provided to you. Respond in English.

    Below is the conversation history up to this point:
    {}
    
    Format:
    Response: <FILL IN>
    """).format(
      datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
      f"{self._character_data.name}'s" if not self._character_data.name.endswith(
        's') else f"{self._character_data.name}'",
      self._character_data.status,
      self._character_data.position,
      observation,
      self._character_data.name,
      speaker,
      '\n\n'.join([summary for summary in memory_summaries]),
      posible_action,
      self._character_data.name,
      self._conversation_history
    )

    self._logger.agent_info(f'Generated prompt: {prompt}')

    response, tokens = chat_completion(prompt, self._character_data.bio, '16k')
    response = response[response.find(':') + 1:].strip()
    response = response.replace("\"", "")

    self._logger.agent_info(f'Generated response: {response} \nTokens: {tokens}')

    self._conversation_history += f'{self._character_data.name}: {response}\n'

    response_chunks = [m.strip() if m.endswith('?') or m.endswith('!') else m.strip() +
                               '.' for m in response.split('.') if m.strip() != '']

    @threaded
    def _determine_pose(chunk):
      return self._mood_analyzer.determine_pose(chunk)

    full_response = [[_determine_pose(chunk), chunk] for chunk in response_chunks]

    if tokens > 3500:
      self._conversation_history = ''
      self._generative_memory.generate_reflections()
      self._status_thread.start()

    self._agent_memory.record_memory(observation)

    self._logger.agent_info(f'Finished generating response in {time.time() - initial_time} seconds')

    return full_response
