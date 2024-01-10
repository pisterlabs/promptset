# This file incorporates modifications based on the algorithm implemented by windszzlang.
# Source: https://github.com/windszzlang/DiagGPT

import streamlit as st
import os
import re
from typing import Dict, List, Tuple
from langchain import LLMChain, PromptTemplate
from langchain.chat_models import AzureChatOpenAI  # azure specific.
from langchain.docstore.document import Document
from langchain.llms import AzureOpenAI  # azure specific.
from langchain.memory import ConversationSummaryBufferMemory, ReadOnlySharedMemory
from langchain.prompts.chat import (
    ChatPromptTemplate, 
    SystemMessagePromptTemplate, 
    HumanMessagePromptTemplate
)
from langchain.schema import HumanMessage  # azure specific.

from utilities import tool
from prompts.chat_agent import CHAT_PROMPT_TEMPLATE
from prompts.enrich_agent import ENRICH_TOPIC_PROMPT
from prompts.manager_agent import MANAGE_TOPIC_PROMPT
from prompts.intro import AI_INTRO, USER_INTRO
from prompts.intro import AI_INTRO, USER_INTRO
from task_loading import load_predefined_tasks

from langchain import chat_models
import textwrap

my_api_type =  st.secrets["OPENAI_API_TYPE"]
my_api_version =  st.secrets["OPENAI_API_VERSION"]
my_api_based = st.secrets["OPENAI_API_BASE"]
my_api_key =   st.secrets["OPENAI_API_KEY"]
my_api_deploy = st.secrets["AZURE_DEPLOY"]

class CocoGPT():
  """
    CocoGPT is the core class for the CocoGPT 04 application, encompassing conversation management,
    topic handling, and interaction with the language model.
    
    Attributes:
    predefined_tasks (dict): Loaded tasks for conversation topics.
    topic_stack (list): A stack to manage conversation topics.
    tool_list (list): List of tools/methods for conversation handling.
    chat_model (LLMChain): The chat model used for generating responses.
    topic_enricher (LLMChain): Model for enriching conversation topics.
    topic_manage_agent (LLMChain): Model for managing and switching topics.
    beginning_topic (str): The initial topic for conversation.
    current_task (dict): The current task being discussed.
    llm (AzureChatOpenAI): Instance of the language model for response generation.
  """
  wrapper = textwrap.TextWrapper(width=70)  # Set the desired text width
  predefined_tasks = load_predefined_tasks()

  def wrap_print(self, message, is_checkpoint=False, indent_level=0):
    """Prints formatted messages with optional checkpoint indicators.

      Args:
          message (str): The message to be printed.
          is_checkpoint (bool): Flag to indicate if the message is a checkpoint.
          indent_level (int): The level of indentation for the message.
    """
    indent = "    " * indent_level
    if is_checkpoint:
        formatted_message = f"\n--- {message} ---\n"
    else:
        formatted_message = indent + message
    print(self.wrapper.fill(formatted_message))


  # Initialization method
  def __init__(self):
      """
        Initializes the CocoGPT instance, setting up models, conversation topics, and tools.
      """
      # streaming_callback = StreamingQueueCallbackHandler()
      # if streaming_callbacks is None:
      # streaming_callbacks = [StreamingStdOutCallbackHandler()]
      # self.streaming_buffer = streaming_callback.q

      self.topic_stack = []
      self.tool_list = [self.stay_at_the_current_topic,
                        self.create_a_new_topic,
                        self.finish_the_current_topic,
                        self.finish_the_current_topic_and_create_a_new_topic_together,
                        self.finish_the_current_topic_and_jump_to_an_existing_topic_together,
                        self.jump_to_an_existing_topic,
                        self.load_topics_from_a_predefined_task]

      self.chat_model = self._init_chat_model()
      self.topic_enricher = self._init_topic_enricher(self.chat_model.memory)
      self.topic_manage_agent = self._init_topic_manage_agent(self.chat_model.memory)

      # Define the beginning topic for the conversation
      self.beginning_topic = self.introduction_topic = 'Introduce yourself to user'

      # Define topic types for clarity in conversation flow
      self.topic_type = {
              'ask': 'Asking user: ',
              'answer': 'Answering user: ',
              'goal': 'Completing goal: '
        }

      self.topics_initialized = False  # Add a flag to check if topics are already initialized
      self.current_task = {} # Initialize the current task with default values
      self.init_topics() # Initialize the topics
      self.llm = self._init_llm() # Initialize the large language model (LLM)


  def _init_llm(self):
    """Initializes the Large Language Model (LLM).

      Returns:
        An instance of the LLM configured with specific deployment settings.
    """
    # return AzureChatOpenAI(deployment_name=my_api_deploy, temperature=0.7)
    # return AzureChatOpenAI(deployment_name= my_api_deploy , temperature=0.3)
    # return AzureChatOpenAI(deployment_name= my_api_deploy , temperature=0.0)
    return AzureChatOpenAI(deployment_name=my_api_deploy , temperature=0.0) # First choice.

  def _init_chat_model(self) -> LLMChain:
    """Initializes the chat model for conversation.

    Returns:
        LLMChain: A chain of language models configured for conversation handling.
    """
    llm = AzureChatOpenAI(deployment_name=my_api_deploy, temperature=0.0, streaming=True)
    template = CHAT_PROMPT_TEMPLATE

    # Prepare chat prompts and memory settings
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt])
    memory = ConversationSummaryBufferMemory(max_token_limit=2000, input_key='human_input',
                                              memory_key="chat_history",
                                              return_messages=True,
                                              human_prefix='User:',
                                              ai_prefix='COCO (Mental Health Expert): ',
                                              llm = AzureChatOpenAI(deployment_name=my_api_deploy, temperature=0))

    memory.save_context({'human_input': USER_INTRO}, {'output': AI_INTRO})
    # Create and return the chat model chain
    chat_model = LLMChain(llm=llm, prompt=chat_prompt, memory=memory)
    return chat_model

    
  def is_topic_stack_empty(self):
      """Check if the topic stack is empty."""
      return len(self.topic_stack) == 0

    
  def _init_topic_enricher(self, memory) -> LLMChain:
      llm = AzureChatOpenAI(deployment_name=my_api_deploy, temperature=0)
      prompt = PromptTemplate(input_variables=['original_topic', 'chat_history'], template=ENRICH_TOPIC_PROMPT)
      chain = LLMChain(llm=llm, prompt=prompt, memory=ReadOnlySharedMemory(memory=memory))
      return chain

    
  def _init_topic_manage_agent(self, memory) -> LLMChain:
      llm = AzureChatOpenAI(deployment_name=my_api_deploy, temperature=0)
      prompt = PromptTemplate(input_variables=['topic_list', 'current_topic', 'tool_description', 'tool_names', 'human_input', 'chat_history'], template=MANAGE_TOPIC_PROMPT)
      chain = LLMChain(llm=llm, prompt=prompt, memory=ReadOnlySharedMemory(memory=memory))
      return chain

    
  @tool(
      name='Stay At the Current Topic',
      description='useful when you think the user still want to stay at the current topic and will talk more about this topic.'
                  'This tool does not have any input.'
  )
  # @tool(
  # name='Stay At the Current Topic',
  # description='useful when you think the user want to stay at the current topic and will talk more about this topic. specially in cases where the user’s response is short or when the conversation is reaching a natural conclusion.'
  #             'This tool does not have any input.'
  # )
  def stay_at_the_current_topic(self):
    """Keeps the conversation on the current topic.

    Raises:
        IndexError: If the topic stack is empty, indicating no current topic.

    Returns:
        The last topic from the stack, indicating continued focus on it.
    """
    if self.is_topic_stack_empty():
        raise IndexError("Cannot stay at the current topic because the topic stack is empty.")
    # print(f"Checkpoint - stay_at_the_current_topic: Returning last topic: {self.topic_stack[-1]}") # Checkpoint
    return self.topic_stack[-1]

  @tool(
      name='Create a New Topic',
      description='useful when you think the user starts a new topic which is different from the current topic, and will discuss this topic next.'
                  'Use when the current topic is relevant and not sensitive.' ## new
                  'If you want to create a new topic, but the new topic is similar to the current topic, please do not use this tool and use the tool: Stay At the Current Topic'
                  'If you want to create a new topic, but the new topic is similar to an existing topic on the topic list , please do not use this tool and use the tool: Finish the Current Topic and Jump To an Existing Topic Together'
                  'The input to this tool should be a string representing the name of the new topic.'
  )
    
  def create_a_new_topic(self, topic_name: str):
      topic_name = self.topic_type['answer'] + topic_name
      self.topic_stack.append(topic_name)
      # print(f"Checkpoint - create_a_new_topic: Topic stack after adding new topic: {self.topic_stack}")  # Checkpoint
      return topic_name

  @tool(
      name='Finish the Current Topic',
      description='useful when you think the user has already known about the answer of current topic and wants to finish the current topic,'
                  'or the user has already answered the question you ask in the current topic.'
                  'or the user does not want to talk more about the current topic and wants to finish it'
                  'This tool does not have any input.'
  )
    
  def finish_the_current_topic(self):
    if self.is_topic_stack_empty():
        raise IndexError("Cannot finish the current topic because the topic stack is empty.")
    finished_topic = self.topic_stack.pop()
    # print(f"Checkpoint - finish_the_current_topic: Topic stack after finishing topic: {self.topic_stack}, Finished Topic: {finished_topic}")
    return finished_topic

  @tool(
      name='Finish the Current Topic and Create a New topic Together',
      description='useful when you think the user want to finish the current topic and create a new safe topic in one round of dialogue'
                  'If you want to create a new topic, but the new topic is similar to an existing topic on the topic list , please do not use this tool'
                  'The input to this tool should be a string representing the name of the new created topic.'
  )
    
  def finish_the_current_topic_and_create_a_new_topic_together(self, topic_name: str):
    try:
        self.finish_the_current_topic()
        return self.create_a_new_topic(topic_name)
    except IndexError as e:
        raise e
        
  @tool(
      name='Finish the Current Topic and Jump To an Existing Topic Together',
      description='useful when you think the user want to finish the current topic and jump to an exisiting topic in one round of dialogue'
                  'The input to this tool should be a string representing the name of an existing topic in the topic list, which must be one topic from the topic list'
  )
    
  def finish_the_current_topic_and_jump_to_an_existing_topic_together(self, topic_name: str):
    try:
        self.finish_the_current_topic()
        return self.jump_to_an_existing_topic(topic_name)
    except IndexError as e:
        raise e

  @tool(
      name='Jump To an Existing Topic',
      description='useful when you think the user wants to jump to an exisiting topic (recall a previous topic) which is in the topic list.'
                  'The input to this tool should be a string representing the name of an existing topic in the topic list, which must be one topic from the topic list. If the topic does not exist, do not to the non-existent topic.'
  )
    
  def jump_to_an_existing_topic(self, topic_name: str):
    # if not topic_name:
    #     raise ValueError("The topic name cannot be empty.")
    # if topic_name not in self.topic_stack:
    #     raise ValueError(f"Unknown existing topic: `{topic_name}`")
    # # self.topic_stack.remove(topic_name)
    # self.topic_stack.append(topic_name)
    # return topic_name
      if topic_name in self.topic_stack:
        self.topic_stack.remove(topic_name) 
      else:
        raise ValueError(f"Unknown existing topic: `{topic_name}`")
      self.topic_stack.append(topic_name)
      return topic_name
      
  @tool(
      name='Load Topics From a Predefined Task',
      description='useful when you think the user starts a predefined task (a complex topics group).'
                  'All predefined task includes: (separated by comma): ' + ', '.join(predefined_tasks.keys()) +
                  'A predefined task contains a group dialogue topics we define for you, you should distinguish it from topics which are already in topic list'
                  'The input to this tool should be a string representing the name of a predefined task, which must be from (separated by comma): ' + ', '.join(predefined_tasks.keys()) +
                  'You can just use this tool once.'
  )
    
  def load_topics_from_a_predefined_task(self, task_name: str):
    if task_name not in self.predefined_tasks:
        raise ValueError(f"The task '{task_name}' does not exist in predefined tasks.")
      task = self.predefined_tasks[task_name]
    # self.wrap_print(f"Checkpoint - Loading Task: {task_name} with details: {task}", is_checkpoint=True) # Checkpoint
    if not isinstance(task['checklist'], list):
        raise TypeError(f"The checklist for task '{task_name}' is not a list.")
    main_topic_name = self.topic_type['goal'] + task['goal']
    self.topic_stack.append(main_topic_name)
      for topic_name in task['checklist'][::-1]:
        topic_name = self.topic_type['ask'] + topic_name
        self.topic_stack.append(topic_name)
    return main_topic_name + ', ' + ', '.join(self.topic_stack)


  @property
  def tool_description(self) -> str:
    return "\n".join([f'{tool.name}: {tool.description}' for tool in self.tool_list])

  @property
  def tool_names(self) -> str:
    return ", ".join([tool.name for tool in self.tool_list])

  @property
  def tool_by_names(self) -> Dict:
    return {tool.name: tool for tool in self.tool_list}

  @property
  def topic_list(self) -> str:
    return '; '.join([topic for topic in self.topic_stack])

    
  def enrich_topic(self, original_topic):
    """Enriches a given topic for better conversation flow.

    Args:
        original_topic (str): The topic to be enriched.

    Returns:
        A new, enriched topic based on the original.
    """
    # Check if there's any topic to enrich
    if self.is_topic_stack_empty():
        raise ValueError("Cannot enrich topic because the topic stack is empty.")
        
    # Attempt to enrich the topic and handle possible errors
    try:
        new_topic = self.topic_enricher.predict(original_topic=original_topic)
        # self.wrap_print(f"Checkpoint - Enriched Topic: Original - {original_topic}, New - {new_topic}", is_checkpoint=True) # Checkpoint
        return new_topic
    except IndexError as e:
        print(f"Error during topic enrichment: {e}")
        return original_topic
    except Exception as e:
        print(f"An unexpected error occurred during topic enrichment: {e}")
        raise


  def chat(self, query: str) -> str:
    """Generates a chat response based on the current topic or dynamically using the LLM.

    Args:
        query (str): The user's input question or statement.

    Returns:
        str: The generated response to the query.
    """
    # self.wrap_print("Chat Response", is_checkpoint=True) # Checkpoint
    if self.is_topic_stack_empty():
      # Generate a dynamic response using the LLM
      llm_response = self.generate_dynamic_llm_response(query)
      return llm_response

    try:
      current_topic = self.enrich_topic(self.topic_stack[-1])
      print()
      # self.wrap_print(f"Checkpoint [chat - 1]: Current topic after enrichment: {current_topic}", is_checkpoint=True) # Checkpoint
      # Fallback to LLM response when the current topic is invalid
      if not current_topic:
        # self.wrap_print("Checkpoint [chat - Fallback]: Using LLM for dynamic response.", is_checkpoint=True) # Checkpoint
        return self.generate_llm_response(query, current_topic)
    except IndexError as e:
      self.wrap_print(f"Checkpoint [chat - Error]: Error in enriching topic: {e}", is_checkpoint=True)
      return self.generate_llm_response(query, "I encountered an error. Let's continue our conversation.")

    try:
      result = self.chat_model.predict(human_input=query,
                                        current_topic=current_topic,
                                        task_overview=self.current_task['overview'],
                                        final_goal=self.current_task['goal'])
      # Fallback to LLM response when the chat model returns an invalid response
      if not result:
        self.wrap_print("Checkpoint [chat - Error]: The chat model returned an invalid response.", is_checkpoint=True) # Checkpoint
        return self.generate_llm_response(query, current_topic)
    except Exception as e:
      self.wrap_print(f"Checkpoint [chat - Error]: An unexpected error occurred: {e}", is_checkpoint=True)  # Checkpoint
      return self.generate_llm_response(query, "I encountered an unexpected error.")

    # self.wrap_print("Checkpoint [chat - End]: Chat response generated\n", is_checkpoint=True)  # Checkpoint
    # stripName = result.replace("COCO: ", "").strip()   # Remove the "COCO:" prefix from the ui response output
    response = result.replace("COCO:", "").replace("AI:", "").strip()
    return response

    
  def init_topics(self):
    # topics = load_saft_topics()
    # self.topic_stack.extend(topics)

    # Check if topics are already initialized
    if not self.topics_initialized:
      # self.wrap_print("Checkpoint [init_topics - Start]: Initializing topics", is_checkpoint=True) # Checkpoint
        
      if not self.beginning_topic:
        raise ValueError("Beginning topic is not set.")
      self.topic_stack.append(self.beginning_topic)
      self.current_task = {
          'overview': None,
          'goal': None,
          'checklist': [],
          'subtasks' : []
      }
      # self.wrap_print(f"Checkpoint - init_topics: Topic stack after init: {self.topic_stack}", is_checkpoint=True) # Checkpoint
      self.topics_initialized = True # Set the flag to True as topics are now initialized
    
    else:
        self.wrap_print("Checkpoint - init_topics: Topics already initialized", is_checkpoint=True) # Checkpoint

    
  def parse_agent_output(self, agent_output: str):
    # self.wrap_print("Checkpoint [parse_agent_output]: Parsing agent output", is_checkpoint=True) # Checkpoint
    print('######\n<Agent output>:') # Checkpoint
    print('Thought: ' + agent_output) # Checkpoint
    print('######') # Checkpoint

    regex = r"Action: [\[]?(.*?)[\]]?[\n]*Action Input:[\s]*(.*)[\n]*Observation: "
    match = re.search(regex, agent_output, re.DOTALL)
    if not match:
        raise ValueError(f"Output of LLM is not parsable for next tool use: `{agent_output}`")
    tool = match.group(1).strip()
    tool_input = match.group(2)
    tool_input = tool_input.replace('"', '').replace("'", '').strip()
    return tool, tool_input

    
  # In the do_agent_action method, handle the creation of a new topic if necessary
  def do_agent_action(self, tool, tool_input):
    # self.wrap_print(f"Checkpoint [do_agent_action - Start]: Executing tool '{tool}' with input '{tool_input}'", is_checkpoint=True) # Checkpoint
    if tool_input == 'None' or tool_input is None:
        tool_result = self.tool_by_names[tool]()
    else:
        if tool == 'Create a New Topic':
            tool_result = self.create_a_new_topic(tool_input)
        else:
            tool_result = self.tool_by_names[tool](tool_input)
    return tool_result

    
  def run_agent(self, query: str):
    # self.wrap_print("Agent Operation", is_checkpoint=True) # Checkpoint
    if self.is_topic_stack_empty():
      raise IndexError("Cannot run agent because the topic stack is empty.")
    # self.wrap_print(f"Checkpoint [run_agent - Current Stack]: {self.topic_stack}", is_checkpoint=True) # Checkpoint
    # self.wrap_print(f"Checkpoint [run_agent - User Query]: {query}", is_checkpoint=True) # Checkpoint
    agent_output = self.topic_manage_agent.predict(
        topic_list=self.topic_list,
        current_topic=self.topic_stack[-1],
        tool_description=self.tool_description,
        tool_names=self.tool_names,
        human_input=query
    )
    # self.wrap_print(f"Checkpoint - Agent Output: {agent_output}", is_checkpoint=True) # Checkpoint
      tool, tool_input = self.parse_agent_output(agent_output)

    # if tool == 'Jump To an Existing Topic' and tool_input not in self.topic_stack:
    #   self.finish_the_current_topic()
    #   self.jump_to_an_existing_topic(tool_input)

    if tool == 'Create a New Topic':
      self.create_a_new_topic(tool_input)
    elif tool == 'Finish the Current Topic':
      self.finish_the_current_topic()
    elif tool in ['Divert Topic', 'Adapt Conversation']:
      # self.wrap_print(f"Checkpoint [run_agent - Diversion/Adaptation]: Adapting or diverting conversation based on user input.", is_checkpoint=True) # Checkpoint
      self.handle_diversion_or_adaptation(query, tool_input)
    else:
      # Execute the standard tool action
      tool_result = self.do_agent_action(tool, tool_input)
      # self.wrap_print(f"Checkpoint [run_agent - End]: Completed agent operation with result: {tool_result}", is_checkpoint=True) # Checkpoint
      return tool_result
    self.remove_redundant_topics()
    # self.wrap_print(f"Checkpoint [run_agent - End]: Completed agent operation. Current Stack: {self.topic_stack}", is_checkpoint=True) # Checkpoint

    
  def remove_redundant_topics(self, round_threshold: int=3):
      # self.wrap_print("Checkpoint [remove_redundant_topics - Start]: Removing redundant topics", is_checkpoint=True) # Checkpoint
      if len(self.topic_stack) <= round_threshold:
          # If the stack is not larger than the threshold, we cannot remove anything.
          return

      new_topic_stack = []
      for i, topic_name in enumerate(self.topic_stack):
          # Check if the topic is an 'answer' type and within the last 'round_threshold' topics
          if topic_name.startswith(self.topic_type['answer']) and len(self.topic_stack) - i > round_threshold:
              continue
              
          new_topic_stack.append(topic_name)
      self.topic_stack = new_topic_stack


  def handle_diversion_or_adaptation(self, query, tool_input):
    """
    Handles conversation diversions or adaptations based on user input.

    Args:
        query (str): The current user input that led to the diversion or adaptation.
        tool_input (str): Additional input or instruction for handling the diversion or adaptation.
    """
    # self.wrap_print(f"Handling diversion or adaptation based on query: '{query}' and tool_input: '{tool_input}'", is_checkpoint=True) # Checkpoint
    # Logic to adapt or divert conversation
    if tool_input == 'adapt_to_user_interest':
      # If the user is showing interest in a different but relevant topic, adapt the conversation accordingly
      new_topic = 'Adapted Topic based on user interest'
      self.topic_stack.append(new_topic)
      # self.wrap_print(f"Adapted to new topic: {new_topic}", is_checkpoint=True) # Checkpoint

    elif tool_input == 'general_inquiry':
      # If the user's query is general or vague, direct the conversation to a more general topic
      general_topic = 'General Mental Health Discussion'
      self.topic_stack.append(general_topic)
      # self.wrap_print(f"Switched to general topic: {general_topic}", is_checkpoint=True) # Checkpoint
      
    else:
      self.wrap_print(f"No specific adaptation for tool_input: '{tool_input}'. Continuing with current topic.", is_checkpoint=True) # Checkpoint

    
  def generate_dynamic_llm_response(self, user_input):
    """Generates a dynamic response using the large language model (LLM).

      Args:
          user_input (str): The user's input that requires a dynamic response.

      Returns:
          str: The generated response from the LLM.
    """
    try:
      response = self.llm.predict(user_input)
    except Exception as e:
      self.wrap_print(f"LLM Error: {e}", is_checkpoint=True)
      return "I'm sorry, I encountered an error processing your request."

    return response

    
  def run(self, query: str):
    # self.wrap_print("Processing Query", is_checkpoint=True) # Checkpoint
    # print('<Stack status 1>: ' + self.topic_list) # Checkpoint
    if self.topic_stack:
      agent_action_result = self.run_agent(query)
      # print('<Stack status 2>:' + '; '.join(self.topic_stack)) # Checkpoint
      chat_response = self.chat(query)
      try:
        self.remove_redundant_topics(3)
        # self.wrap_print('Checkpoint - run: After removing redundant topics ' + str(self.topic_stack), is_checkpoint=True) # Checkpoint
      except IndexError as e:
        print(f"Error during post-chat processing: {e}")
        raise
    
    else:
      chat_response = "I'm sorry, I'm not sure how to respond to that."
    
    output = chat_response.replace("COCO:", "").strip()
    return output

    
  def reset_conversation(self):
    """
    Resets the conversation to default settings.
    """
    self.topic_stack = [self.beginning_topic]
    self.current_task = {
        'overview': None,
        'goal': None,
        'checklist': [],
        'subtasks': []
    }
    print("Conversation has been reset to default settings.") # Checkpoint
