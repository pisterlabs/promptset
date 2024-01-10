# /Agents/Agency.py

import json
import openai

from .Team import Team
from .Session import SessionManager, Session
from .AgentConfig import AgentConfigurationManager
from Agents import SprAgent, UserAgent
from Utilities.Config import GetClient, current_model
from Utilities.Log import Log, Debug, type
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Agents.BaseAgent import BaseAgent

class Agency:
    def __init__(
        self, prompt=None, new_session: bool = False, rebuild_agents: bool = False
    ):
        self.client: openai = GetClient()
        self.thread = None
        self.agents: ["BaseAgent"] = []
        
        self.team_instructions = None
        self.plan = None
        self.prompt = prompt
        
        self.active_agent: 'BaseAgent' = None
        self.running_tool = False
        self.message_queue = []
        self.delegation_count = 0

        self.session_manager:SessionManager = SessionManager(client=self.client,prompt=prompt, new_session=new_session)
        self.session: Session = self.session_manager.active_session
        
        self.agents_manager: AgentConfigurationManager = AgentConfigurationManager(agency=self, rebuild_agents=rebuild_agents)
        self.agents = self.agents_manager.agents
        
    def get_agent(self, name) -> "BaseAgent":
        
        for agent in self.agents:
            if agent.name == name:
                return agent

        # An invalid name was supplied, use GPT to find the correct agent name
        Log(type.ERROR, f"Agent named '{name}' not found in agency. Engaging fall back...")

        list_of_agent_names = [agent.name for agent in self.agents]
        Log(type.ERROR, f"Actual agent names: {', '.join(list_of_agent_names)}")

        completion = self.client.chat.completions.create(
            model=current_model,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": """
                        Task: Match closest valid agent name.
                        Input: Valid names, one invalid.
                        Output: JSON with closest match.
                        Method: String similarity analysis.
                        JSON Format: {"Name": "Closest valid agent"}.
                    """.strip(),
                },
                {
                    "role": "user",
                    "content": f"Valid names: {', '.join(list_of_agent_names)}.\nInvalid name:{name}",
                },
            ],
        )

        actualAgentName = json.loads(completion.choices[0].message.content)["Name"]
        Log(type.ERROR, f"Agent name fallback determined: {actualAgentName}")

        for agent in self.agents:
            if agent.name == actualAgentName:
                return agent

        Log(type.ERROR, f"Requested Agent could still not be found in agency... Returning user agent")
        return self.get_agent(UserAgent.NAME)

    def UpdatePlan(self, plan):
        # todo
        self.plan = plan

    def _queue_message(self, message):
        self.message_queue.append(message)

    def add_message(self, message):
        if self.running_tool:
            self._queue_message(message)
            return

        self.waiting_on_response = False

        # todo: support seed
        # appears to currently not be supported: https://github.com/openai/openai-python/blob/790df765d41f27b9a6b88ce7b8af713939f8dc22/src/openai/resources/beta/threads/messages/messages.py#L39
        # reported issue: https://community.openai.com/t/seed-param-and-reproducible-output-do-not-work/487245

        return self.client.beta.threads.messages.create(
            thread_id=self.thread.id,
            role="user",
            content=message,
        )
        
    def get_team_instruction(self):
        if self.team_instructions is None:
            
            verbose_team_instructions = Team.build_agents_list_and_arsenal()
            Debug(f"Verbose Team instructions:\n{verbose_team_instructions}")
            
            # Use SPR Writer to compress the instructions
            completion = self.client.chat.completions.create(
                model=current_model,
                messages=[
                    {"role": "system", "content": SprAgent.INSTRUCTIONS},
                    {"role": "user", "content": verbose_team_instructions},
                ],
            )
            
            compressed_team_instructions = completion.choices[0].message.content
            Debug(f"Concise Team instructions:\n{compressed_team_instructions}")
            
            self.team_instructions = compressed_team_instructions
            
        return self.team_instructions

    # main method to get the agency to work the prompt
    def complete(
        self, mission_prompt, single_agent, stop_word="exit", continue_phrase=None
    ):
        if self.prompt is None:
            self.prompt = mission_prompt
        
        if self.thread is None:
            self.thread = self.session_manager.get_session(self.prompt)

        if continue_phrase is None:
            continue_phrase = ""

        prompt = mission_prompt

        if self.active_agent is None:
            self.active_agent = self.get_agent(UserAgent.NAME)

        while True:

            if single_agent:
                agent_name = self.active_agent.name
                prompt += "\nWork on this alone, do not delegate.\n"
                response = self.active_agent.get_completion(message=prompt)
                Debug(f"{agent_name} responded with:\n{response}")
                Debug(f"Active agent: {self.active_agent.name}")
            else:
                response = self._operate(prompt)

            Log(type.RESULT, f"{self.active_agent.name}:\n{response}")

            message = f"Waiting for reply from user. Or type '{stop_word}'"
            if continue_phrase is not None:
                message += f" to {continue_phrase}"
            message += ":\n\n"

            Log(type.PROMPT, message)

            prompt = input("> ")

            if prompt.lower() == stop_word.lower():
                if continue_phrase is not None:
                    Log(type.ACTION, f"\t{continue_phrase}")
                break

        return response

    # Used to have user agent delegate and auto respond to agency
    def _operate(self, prompt):
        # Trigger the initial delegation
        Debug(f"Starting operation. User provided prompt: {prompt}")
        response = self.active_agent.get_completion(prompt)
        Debug(f"Initial response: {response}")

        user_agent = self.get_agent(UserAgent.NAME)
        user_agent.task_delegated = False

        while self.active_agent.waiting_on_response == False:
            Debug(f"Active agent: {self.active_agent.name}")

            # Store the name so that we can recognize who the previous agent was after a delegation
            active_agent_name = self.active_agent.name

            response = self.active_agent.get_completion()
            Log(type.COMMUNICATION, f"{self.active_agent.name}:\n{response}")

            previous_agent = self.get_agent(active_agent_name)

            if previous_agent.task_delegated == True:
                # Turn this flag off now that delegation is completed
                previous_agent.task_delegated == False

            # Get user agent to handle the response in order to automate the next step if an agent response instead of tool usage
            elif (
                previous_agent.task_delegated == False
                and active_agent_name != UserAgent.NAME
            ):
                prompt = f"{response}\n\n In regards to the overall plan. What do we do now leader?"
                Debug(
                    f"{active_agent_name} has responded and is addressing user agent:\n{prompt}"
                )
                self.active_agent.waiting_on_response = False
                self.active_agent = user_agent
                # Attempt to delegate
                response = user_agent.get_completion(message=prompt)
                Debug(
                    f"User agent is expected to have delegated. This was its response:{response}"
                )
                Debug(f"The new active agent is: {self.active_agent.name}")
                # If the user agent is still active, this will get the response sent back to the user
                if self.active_agent.name == UserAgent.NAME:
                    self.active_agent.waiting_on_response = True
                # When successfully delegated, loop will restart, causing the next agent to pick up the delegate instruction message

        Debug(
            f"{self.active_agent.name} is returning back to the user with: {response}"
        )
        return response
