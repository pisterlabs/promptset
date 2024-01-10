import openai 
import json

class SwarmArchitect:

    def __init__(self):

        self.client = openai.Client()

        self.swarm_architect = "asst_HmxHtEXXuAAzBJYen2udhoFJ"

        self.agent_creator = "asst_47ywjR3NxwhYOhRhllYplRs9"

        self.swarm_architect_agent = self.client.beta.assistants.retrieve(self.swarm_architect)

        self.agent_creator_agent = self.client.beta.assistants.retrieve(self.agent_creator)

        self.tools = {

            "create_agent": self.create_agent

        }

        self.swarm = self.client.beta.assistants.list()

  

        self.agents_by_name = {}

        for agent in self.swarm:

            self.agents_by_name[agent.name] = agent

  

    def create_agent(self, name, system_prompt):

        if name in self.agents_by_name:

            return self.agents_by_name[name].id

        assistant = self.client.beta.assistants.create(

            name=name,

            instructions=system_prompt,

            model="gpt-4-1106-preview"

        )

        self.agents_by_name[name] = assistant

        return assistant.id

    def create_agents_for_goal(self, goal):

        run, thread = self.swarm_architect_thread(goal)

        run = self.creator_agent(thread,run)

        self.handle_tool_calls(run, thread)

  

    def swarm_architect_thread(self, goal): ## create a new chat, and run the swarm architect agent on it

        thread = self.client.beta.threads.create()

        self.client.beta.threads.messages.create(

            thread_id=thread.id,

            role="user",

            content=goal

        )

        run = self.client.beta.threads.runs.create(

            thread_id=thread.id,

            assistant_id=self.swarm_architect_agent.id

        )

        while run.status != "completed":

            run = self.client.beta.threads.runs.retrieve(

            run_id=run.id,

            thread_id=thread.id

            )

            messages = self.client.beta.threads.messages.list(thread_id=thread.id)

            print(run.status)

  

        for message in messages:

            print(message)

        return run, thread

    def creator_agent(self, thread, run):

        ## add the creator agent to the chat by creating a new run

        self.client.beta.threads.messages.create(

            thread_id=thread.id,

            role="user",

            content="Please create the agents that the swarm architect agent suggested."

        )

        run = self.client.beta.threads.runs.create(

            thread_id=thread.id,

            assistant_id=self.agent_creator_agent.id

        )

        return run

  

    def handle_tool_calls(self, run, thread):

        ## waiting for the run status to be completed

        while run.status != "completed":

            run = self.client.beta.threads.runs.retrieve(

            run_id=run.id,

            thread_id=thread.id

            )

            print(run.status)

            ## if the run status is requires_action, then we need to run the tools

            if run.status == "requires_action":

                tool_outputs = []

                tool_calls = run.required_action.submit_tool_outputs.tool_calls

                ## for each tool call, run the tool and collect the output

                for tool_call in tool_calls:

                    tool_call_id = tool_call.id

                    tool = self.tools[tool_call.function.name]

                    tool_kwargs = json.loads(tool_call.function.arguments)

                    result = tool(**tool_kwargs)

                    tool_outputs.append({

                        "tool_call_id": tool_call_id,

                        "output": "success"

                    })

                ## submit the tool outputs to the run

                run = self.client.beta.threads.runs.submit_tool_outputs(

                    thread_id=thread.id,

                    run_id=run.id,

                    tool_outputs=tool_outputs

                )

  

        while run.status != "completed":

                run = self.client.beta.threads.runs.retrieve(

                run_id=run.id,

                thread_id=thread.id

            )

  

        messages = self.client.beta.threads.messages.list(thread_id=thread.id)

        for message in messages:

            print(message)

  
  

swarm_architect = SwarmArchitect()

# swarm_architect.create_agents_for_goal("Please create a simple pair of agents that will function as a recruiter and reviewer. The recruiter agent will review the list of available agets and choose some that match the problem, then the critical agent is a reviewer to ensure we don't recruit too many agents- beyond 3 we encounter diminishing returns, so please only 3 agents can be returned. Thank you!")
# swarm_architect.create_agents_for_goal("Please create a swarm of agents that will make tools for a swarm of agents by making a plan for the tools, reviewing the plan, then creating the tools themselves.")

swarm_architect.create_agents_for_goal("Please create a swarm of agents that will help Multiverse School students collaborate to find housing, and will assist in the housing-finding process through things like searching for housing, grants for schools to provide housing, and castle rehabilitation grants in europe, etc. We need agents to somehow get us a castle.")