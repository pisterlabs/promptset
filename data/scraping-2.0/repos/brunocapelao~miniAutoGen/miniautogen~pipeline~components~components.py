from openai import OpenAI
import openai
import os
import logging
from dotenv import load_dotenv
import time
from jinja2 import Environment, select_autoescape
import json
from miniautogen.pipeline.components.pipelinecomponent import PipelineComponent

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UserExitException(Exception):
    pass


class UserResponseComponent(PipelineComponent):
    """
    A pipeline component that captures user response and updates the state.

    Attributes:
        None

    Methods:
        process(state): Captures user response and updates the state with the reply.

    """

    def process(self, state):
        """
        Captures user response and updates the state with the reply.

        Args:
            state (dict): The current state of the pipeline.

        Returns:
            str: The user's response.

        """
        try:
            reply = input("Enter the response: ")
            state.update_state(reply=reply)
        except Exception as e:
            print(f"Error capturing user response: {e}")

        return reply


class UserInputNextAgent(PipelineComponent):
    def process(self, state):
        """
        Process the user input to select the next agent.

        Args:
            state (State): The current state of the pipeline.

        Returns:
            State: The updated state with the selected agent.

        Raises:
            ValueError: If the group chat is invalid or no agents are available.
            UserExitException: If the user chooses to exit.
        """
        group_chat = state.get_state().get('group_chat')
        if not group_chat or not group_chat.agentList:
            raise ValueError("Invalid GroupChat or no agents available.")

        for i, agent in enumerate(group_chat.agentList):
            print(f"{i + 1}: {agent.name}")

        next_agent = input(
            "Enter the number corresponding to the agent (or 'exit' to cancel): ")

        if next_agent.lower() == 'exit':
            raise UserExitException("User chose to exit")

        try:
            next_agent_index = int(next_agent) - 1
            if 0 <= next_agent_index < len(group_chat.agentList):
                next_agent = group_chat.agentList[next_agent_index]
                state.update_state(selected_agent=next_agent)
            else:
                print("Invalid choice.")
        except ValueError:
            print("Invalid input. Please enter a number.")

        return state


class NextAgentSelectorComponent(PipelineComponent):
    """
    Component responsible for selecting the next agent in a group chat.

    This component processes the state and selects the next agent based on the last message sender.
    If there are no messages, it starts with the first agent in the list.

    Args:
        PipelineComponent (class): Base class for pipeline components.

    Returns:
        State: Updated state with the selected agent.

    Raises:
        ValueError: If the group chat is invalid or agents are not found.
    """

    def process(self, state):
        """
        Process the state and select the next agent.

        Args:
            state (State): Current state of the pipeline.

        Returns:
            State: Updated state with the selected agent.
        """

        group_chat = state.get_state().get('group_chat')
        if not group_chat or not hasattr(group_chat, 'agentList'):
            raise ValueError("Invalid GroupChat or agents not found.")

        try:
            next_agent = self.select_next_agent(group_chat)
            state.update_state(selected_agent=next_agent)
        except Exception as e:
            print(f"Error selecting the next agent: {e}")

        return state

    def select_next_agent(self, group_chat):
        """
        Select the next agent based on the last message sender.

        Args:
            group_chat (GroupChat): Group chat object.

        Returns:
            Agent: Next agent to be selected.
        """

        last_message = group_chat.get_messages(
            type='json')[-1] if group_chat.get_messages(type='json') else None
        if last_message:
            last_sender_id = last_message['sender_id']
            # Find the index of the last agent who sent a message
            last_agent_index = next((i for i, agent in enumerate(
                group_chat.agentList) if agent.agent_id == last_sender_id), -1)
            # Select the next agent in the list
            next_agent_index = (last_agent_index +
                                1) % len(group_chat.agentList)
        else:
            # If there are no messages, start with the first agent
            next_agent_index = 0

        return group_chat.agentList[next_agent_index]


class AgentReplyComponent(PipelineComponent):
    def process(self, state):
        """
        Processes the response from the current agent and adds that response to the group chat.

        Args:
            state (PipelineState): Current state of the pipeline.

        Returns:
            PipelineState: Updated state of the pipeline.
        """
        # Access the current state to get necessary information
        agent = state.get_state().get('selected_agent')
        group_chat = state.get_state().get('group_chat')
        if not agent or not group_chat:
            raise ValueError(
                "Agent and GroupChat are required for AgentReplyComponent.")
        # Implementation of agent reply generation
        try:
            reply = agent.generate_reply(state)
            print(reply)
            group_chat.add_message(sender_id=agent.agent_id, message=reply)
        except Exception as e:
            print(f"Error processing agent reply: {e}")

        return state


class TerminateChatComponent(PipelineComponent):
    """
    Pipeline component that checks if the word 'TERMINATE' is present
    in the last message and, if so, terminates the chat.
    """

    def process(self, state):
        chat_admin = state.get_state().get('chat_admin')
        chat = state.get_state().get('group_chat')
        if not chat:
            raise ValueError(
                "Chat is required for TerminateChatComponent.")
        messages = chat.get_messages()
        last_message = messages.iloc[-1].message if not messages.empty else None

        if last_message and "TERMINATE" in last_message.upper():
            print("Terminating chat...")
            chat_admin.stop()  # Make sure the stop() method is implemented in GroupChat
            return "TERMINATE"

        # Continue the pipeline flow normally if TERMINATE is not present.
        return state


class OpenAIChatComponent(PipelineComponent):
    """
    Pipeline component to generate responses using the OpenAI language model.
    """

    def __init__(self):
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.logger = logging.getLogger(__name__)

    def process(self, state):
        try:
            chat = state.get_state().get('group_chat')
            agent = state.get_state().get('selected_agent')

            if not chat or not agent:
                raise ValueError(
                    "groupchat and agent are required for OpenAIResponseComponent.")
            messages = chat.get_messages('json')
            prompt = self._construct_prompt(agent, messages)
            response = self._call_openai_api(prompt)
            return response.choices[0].message.content
        except Exception as e:
            self.logger.error(f"Error in OpenAIResponseComponent: {e}")
            raise

    def _construct_prompt(self, agent, messages):
        formatted_messages = [{'role': 'system', 'content': agent.role}]
        for item in messages:
            role = 'assistant' if item['sender_id'] == agent.agent_id else 'user'
            content = item['message']
            formatted = {'role': role, 'content': content}
            formatted_messages.append(formatted)
        return formatted_messages

    def _call_openai_api(self, prompt):
        """ Calls the OpenAI API. """
        try:
            return self.client.chat.completions.create(
                model="gpt-4-1106-preview",
                messages=prompt,
                temperature=1
            )
        except Exception as e:
            self.logger.error(f"Error calling the OpenAI API: {e}")
            raise


class OpenAIThreadComponent(PipelineComponent):
    """
    A pipeline component that interacts with the OpenAI assistant's thread.

    Attributes:
        client (openai.OpenAI): The OpenAI client.
        assistant_id (str): The OpenAI assistant's ID.
        thread (Thread): The OpenAI assistant's thread.
        logger (logging.Logger): The logger for logging errors.

    Methods:
        set_assistant_id: Sets the assistant ID for the component.
        set_thread: Sets the thread ID for the component.
        process: Processes the current state to interact with the OpenAI assistant's thread.
        _create_thread: Creates a new thread for the OpenAI assistant.
        _construct_prompt: Constructs the prompt for the OpenAI assistant.
        _submit_message: Submits a message to the OpenAI assistant's thread.
        _get_response: Retrieves the response from the OpenAI assistant's thread.
        _wait_on_run: Waits for the completion of a run in the OpenAI assistant's thread.
    """

    def __init__(self):
        self.client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.assistant_id = None  # Initialize without an assistant_id
        self.thread = None
        self.logger = logging.getLogger(__name__)

    def set_assistant_id(self, assistant_id):
        """
        Sets the assistant ID for the component.

        Args:
            assistant_id (str): OpenAI assistant's ID.
        """
        self.assistant_id = assistant_id

    def set_thread(self, thread):
        """
        Sets the thread ID for the component.

        Args:
            assistant_id (str): OpenAI assistant's ID.
        """
        self.assistant_id = thread

    def process(self, state):
        """
        Processes the current state to interact with the OpenAI assistant's thread.

        Args:
            state (PipelineState): Current pipeline state.

        Returns:
            PipelineState: Updated pipeline state.
        """
        if not self.assistant_id:
            raise ValueError(
                "Assistant ID not set for OpenAIThreadComponent.")

        if not self.thread:
            self.thread = self._create_thread()

        try:
            # Get necessary information from the state
            chat = state.get_state().get('group_chat')
            agent = state.get_state().get('selected_agent')

            if not chat or not agent:
                raise ValueError(
                    "group_chat and selected_agent are required for OpenAIThreadComponent.")

            # Construct the prompt
            prompt = self._construct_prompt(chat)
            # Submit the message
            run = self._submit_message(self.assistant_id, self.thread, prompt)
            # Wait for the response
            run = self._wait_on_run(run, self.thread)
            # Get the response
            response = self._get_response(self.thread, self.assistant_id)

        except Exception as e:
            self.logger.error(f"Error in OpenAIThreadComponent: {e}")
            raise

        return response

    def _create_thread(self):
        thread = self.client.beta.threads.create()
        return thread

    def _construct_prompt(self, chat):
        message = chat.get_messages('json')[-1]['message']
        return message

    def _submit_message(self, assistant_id, thread, message):
        self.client.beta.threads.messages.create(
            thread_id=thread.id, role="user", content=message
        )
        return self.client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=assistant_id,
        )

    def _get_response(self, thread, assistant_id):
        messages = self.client.beta.threads.messages.list(
            thread_id=thread.id, order="asc")
        for message in messages:
            if message.assistant_id == assistant_id:
                return message.content[0].text.value
        return None

    def _wait_on_run(self, run, thread):
        while run.status == "queued" or run.status == "in_progress":
            run = self.client.beta.threads.runs.retrieve(
                thread_id=thread.id,
                run_id=run.id,
            )
            time.sleep(0.5)
        return run


class Jinja2TemplatesComponent(PipelineComponent):
    """
    A pipeline component that uses Jinja2 templates to generate combined JSON output.

    Attributes:
        templates (list): List of templates to be rendered.
        variables (dict): Variables to be passed to the templates.
        env (Environment): Jinja2 environment for template rendering.

    Methods:
        add_template(template_str, role): Add a template to the list with its respective role.
        set_variables(variables): Set the variables to be passed to the templates.
        _generate_combined_result(): Generate the combined result of rendered templates.
        process(state): Process the state and update it with the combined JSON result.
    """

    def __init__(self):
        self.templates = []
        self.variables = {}
        self.env = Environment(autoescape=select_autoescape())

    def add_template(self, template_str, role):
        """
        Add a template to the list with its respective role.

        Args:
            template_str (str): String containing the Jinja2 template.
            role (str): Template's role ('system', 'user', 'assistant').
        """
        self.templates.append({'template': template_str, 'role': role})

    def set_variables(self, variables):
        """
        Set the variables to be passed to the templates.

        Args:
            variables (dict): Variables to be passed to the templates.
        """
        self.variables = variables

    def _generate_combined_result(self):
        """
        Generate the combined result of rendered templates.

        Returns:
            str: Combined JSON string.
        """
        combined_result = []
        for item in self.templates:
            template_str = item['template']
            role = item['role']
            template = self.env.from_string(template_str)
            rendered_content = template.render(self.variables)
            # Here, we ensure that each item is a valid dictionary for JSON
            combined_result.append({"role": role, "content": rendered_content})

        # Converting the list of dictionaries into a JSON string
        return json.dumps(combined_result)

    def process(self, state):
        """
        Process the state and update it with the combined JSON result.

        Args:
            state (State): The current state of the pipeline.

        Returns:
            State: The updated state.
        """
        chat = state.get_state().get('group_chat')
        agent = state.get_state().get('selected_agent')
        messages = json.loads(chat.get_messages()[
                              ['sender_id', 'message']].to_json(orient='records'))

        # Check if variables have been defined
        if self.variables is None:
            self.variables = state.get_state().get('variables', {})

        self.variables['chat'] = chat
        self.variables['agent'] = agent
        self.variables['messages'] = messages
        combined_json_str = self._generate_combined_result()
        # Convert the JSON string into a Python object to update the state
        combined_json = json.loads(combined_json_str)
        state.update_state(prompt=combined_json)

        return state


class NextAgentMessageComponent(PipelineComponent):
    def __init__(self):
        self.alternative_next = NextAgentSelectorComponent()

    def set_alternative_next(self, alternative_next):
        """
        Set the next component to be executed if no agent is found
        in the last message.

        Args:
            alternative_next (PipelineComponent): Next component to be executed.
        """
        self.alternative_next = alternative_next

    def process(self, state):
        """
        Process the state and select the next agent based on the last message.

        Args:
            state (State): The current state of the pipeline.

        Returns:
            State: The updated state after selecting the next agent.
        """
        chat = state.get_state().get('group_chat')
        agents = chat.agentList

        # Get the last message from the chat
        messages = chat.get_messages()
        last_message = messages.iloc[-1].message if not messages.empty else None

        next_agent = None
        if last_message:
            # Search for the agent_id of each agent in the last message
            for agent in agents:
                if agent.agent_id in last_message:
                    next_agent = agent
                    break

        # Update the state with the selected next agent, if found
        if next_agent:
            state.update_state(selected_agent=next_agent)
        else:
            print("No matching agent found in the last message.")
            self.alternative_next.process(state)

        return state


class UpdateNextAgentComponent(PipelineComponent):
    """
    A pipeline component that updates the state to indicate the next agent based on the provided agent ID.
    """

    def __init__(self):
        self.next_agent_id = None

    def set_next_agent_id(self, next_agent_id):
        """
        Sets the ID of the next agent to be set in the state.

        Args:
            next_agent_id (str): ID of the next agent.
        """
        self.next_agent_id = next_agent_id

    def process(self, state):
        """
        Updates the state to indicate the next agent based on the provided agent ID.

        Args:
            state (PipelineState): Current state of the pipeline.

        Returns:
            PipelineState: Updated state of the pipeline.

        Raises:
            ValueError: If the agent ID is not found among the available agents.
        """
        chat = state.get_state().get('group_chat')
        agents = chat.agentList

        for agent in agents:
            if agent.agent_id in self.next_agent_id:
                next_agent = agent
                break

        if next_agent:
            state.update_state(selected_agent=next_agent)
            print(f"Next agent updated to: {self.next_agent_id}")
        else:
            raise ValueError(
                f"Agent ID '{self.next_agent_id}' not found among the available agents.")

        return state


class Jinja2SingleTemplateComponent(PipelineComponent):
    """
    A pipeline component that uses Jinja2 templating engine to render a single template.

    Attributes:
        template_str (str): The template string to be rendered.
        variables (dict): The variables to be used in the template.
        env (Environment): The Jinja2 environment.

    Methods:
        set_template_str(template_str): Sets the template string for the component.
        set_variables(variables): Sets the variables for the component.
        process(state): Processes the component and renders the template.
    """

    def __init__(self):
        self.template_str = None
        self.variables = None
        self.env = Environment(autoescape=select_autoescape())

    def set_template_str(self, template_str):
        """
        Sets the template string for the component.

        Args:
            template_str (str): String containing the Jinja2 template.
        """
        self.template_str = template_str

    def set_variables(self, variables):
        """
        Sets the variables for the component.

        Args:
            variables (dict): Dictionary of variables for the template.
        """
        self.variables = variables

    def process(self, state):
        """
        Processes the component and renders the template.

        Args:
            state (PipelineState): The current state of the pipeline.

        Returns:
            PipelineState: The updated state of the pipeline.
        """
        template = self.env.from_string(self.template_str)
        chat = state.get_state().get('group_chat')
        agent = state.get_state().get('selected_agent')
        messages = json.loads(chat.get_messages()[
                              ['sender_id', 'message']].to_json(orient='records'))

        # Check if variables have been defined
        if self.variables is None:
            self.variables = state.get_state().get('variables', {})

        self.variables['chat'] = chat
        self.variables['agent'] = agent
        self.variables['messages'] = messages
        # Render the template with the provided variables
        prompt = template.render(self.variables)
        prompt = json.loads(prompt)

        # Update the pipeline state with the rendered output
        state.update_state(prompt=prompt)

        return state


class LLMResponseComponent(PipelineComponent):
    """
    A pipeline component that uses the LLM client to generate a response based on a given prompt.

    Args:
        llm_client (LLMClient): The LLM client used to interact with the language model.
        model_name (str, optional): The name of the language model to use. Defaults to "gpt-4".
    """

    def __init__(self, llm_client, model_name="gpt-4"):
        self.llm_client = llm_client
        self.model_name = model_name
        self.logger = logging.getLogger(__name__)

    def process(self, state):
        prompt = state.get_state().get('prompt')
        if not prompt:
            self.logger.error("Prompt missing in pipeline state.")
            return state

        response = self.llm_client.get_model_response(prompt, self.model_name)
        if response:
            return response
        else:
            self.logger.error("Failed to get response from LLM.")
            state.update_state({'error': "Failed to get response from LLM."})
            return state
