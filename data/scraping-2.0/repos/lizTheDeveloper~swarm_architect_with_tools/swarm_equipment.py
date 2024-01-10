import openai 
import json
import datetime
import time

class SwarmArchitect:

    def __init__(self):

        self.swarm = []
        self.client = openai.Client()

        self.recruiter_agent_id = "asst_8KyCNvb2ZQ7qEXUEe1hayTFf"
        self.recruiter_critical_agent_id = "asst_STHyWmVD1ICbgpRyK1eoZuvp"
        self.evaluator_agent_id = "asst_AElcAx1wsaRpur9VEMQyeuoG"

        self.recruiter_agent = self.client.beta.assistants.retrieve(self.recruiter_agent_id)
        self.recruiter_critical_agent = self.client.beta.assistants.retrieve(self.recruiter_critical_agent_id)
        self.evaluator_agent = self.client.beta.assistants.retrieve(self.evaluator_agent_id)

        ## tool swarm
        self.tool_planner_agent_id = "asst_8wtaubkTo16WLgZ0awmUZNHz"
        self.tool_planner_agent = self.client.beta.assistants.retrieve(self.tool_planner_agent_id)

        self.tool_reviewer_agent_id = "asst_ViE4IoGk3LqW6zufijkg3wAd"
        self.tool_reviewer_agent = self.client.beta.assistants.retrieve(self.tool_reviewer_agent_id)

        self.tool_creator_agent_id = "asst_6wETukcfY4rQCYx8Mr44SUMh"
        self.tool_creator_agent = self.client.beta.assistants.retrieve(self.tool_creator_agent_id)
        
        self.tool_code_reviewer_agent_id = "asst_AikhZSc0NqzxSGteXQGPh0it"
        self.tool_code_reviewer_agent = self.client.beta.assistants.retrieve(self.tool_code_reviewer_agent_id)

        self.tool_alignment_agent_id = "asst_rxYNVvBycM5gAvucXlFGhI6c"
        self.tool_alignment_agent = self.client.beta.assistants.retrieve(self.tool_alignment_agent_id)


        ## build out the swarm if we don't have any agents
        self.swarm_architect = "asst_HmxHtEXXuAAzBJYen2udhoFJ"

        self.agent_creator = "asst_47ywjR3NxwhYOhRhllYplRs9"

        self.swarm_architect_agent = self.client.beta.assistants.retrieve(self.swarm_architect)

        self.agent_creator_agent = self.client.beta.assistants.retrieve(self.agent_creator)


        self.tools = {
            "create_agent": self.create_agent,
            "list_swarm": self.list_swarm,
            "recruit_agent": self.recruit_agent,
            "list_tools": self.list_tools,
            "create_tool": self.create_tool,
            "request_tool": self.request_tool
        }

        swarm = self.client.beta.assistants.list()
        self.swarm = []

        self.agents_by_name = {}

        for agent in self.swarm:

            self.agents_by_name[agent.name] = agent

        self.timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.file = open("./chatlogs/" + self.timestamp, "w")
        self.file.write("Swarm Recruiter Chatlog\n")
        self.file.close()
        self.file = open(self.timestamp, "a")

    ### Flow Methods
    def determine_tool_needs_for_goal(self, goal):
        ## given the goal, determine what tools are needed
        run, thread = self.swarm_tool_thread(goal)
        self.handle_tool_calls(run, thread)
        for message in self.client.beta.threads.messages.list(thread_id=thread.id):
            print(message)
            self.file.write(str(message))
            self.file.write("\n")
        
        self.critique_tools_for_goal(run, thread, goal)

    def critique_tools_for_goal(self, run, thread, goal):
        self.client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=f"Please critique the tools that were suggested. Ensure they are relevant to the goal and nessecary to accomplish the goal. Upon reviewing and deciding on which tools are needed, please call the request_tool function with each tool name requested in order to request the tools. Goal: {goal}."
        )
        run = self.client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=self.tool_reviewer_agent.id,
            tools=[{
                "name": "request_tool",
                "description": "Requests a tool",
                "parameters": {
                    "type": "object",
                    "properties": {
                    "tool_name": {
                        "type": "string",
                        "description": "The name of the tool"
                    },
                    "tool_description": {
                        "type": "string",
                        "description": "The description of the tool"
                    },
                    "tool_inputs": {
                        "type": "string",
                        "description": "The inputs of the tool"
                    },
                    "tool_outputs": {
                        "type": "string",
                        "description": "The outputs of the tool"
                    }
                    },
                    "required": [
                    "tool_name",
                    "tool_description",
                    "tool_inputs",
                    "tool_outputs"
                    ]
                }
            }]
        )
        self.handle_tool_calls(run, thread)
        for message in self.client.beta.threads.messages.list(thread_id=thread.id):
            print(message)
            self.file.write(str(message))
            self.file.write("\n")
        
        run = self.client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=self.tool_alignment_agent.id
        )
        self.handle_tool_calls(run, thread)
    
    def request_tool(self, tool_name, tool_description, tool_inputs, tool_outputs):
        ## create a new thread with a request for the tool between the alignment agent and the tool creator agent
        thread = self.client.beta.threads.create()
        self.client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=f"Please create a tool called {tool_name} that can be used to accomplish the following goal: {tool_description}. The tool should take the following inputs: {tool_inputs} and produce the following outputs: {tool_outputs}."
        )
        run = self.client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=self.tool_creator_agent.id
        )
        self.handle_tool_calls(run, thread)
        for message in self.client.beta.threads.messages.list(thread_id=thread.id):
            print(message)
            self.file.write(str(message))
            self.file.write("\n")
        
        ## ask the tool reviewer agent to review the tool
        self.client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=f"Please review the tool that was created and ensure that it will accomplish it's defined task and not do anything outside of it's defined task, and that it follows good coding practices for observability and readability. Ensure it is a pure function that does not rely on outside state except through interaction with an API or database as needed. If the parsing of natural language is involved, prefer to use a call to OpenAI() to execute a call to a language model. Please ensure there is no placeholder logic or example code that willl not work in production, this code will be immediately used. It will accept these inputs: {tool_inputs} and generate these outputs: {tool_outputs}. The Goal of the tool is: {tool_description}."
        )
        run = self.client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=self.tool_code_reviewer_agent.id
        )
        
        ## ask the coder agent to make the changes
        self.client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=f"Please modify the tool based on the tool reviewer agent's suggestions."
        )
        run = self.client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=self.tool_creator_agent.id
        )

        ## ask the alignment agent to review the tool
        self.client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=f"Please review the tool that was created and ensure it is relevant to the goal and nessecary to accomplish the goal and that it will accomplish it's defined task and not do anything outside of it's defined task, and that it follows good coding practices for observability and readability. Goal: {tool_description}."
        )
        run = self.client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=self.tool_alignment_agent.id
        )

        ## ask the coding agent to modify the tool based on the alignment agent's suggestions
        self.client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=f"Please modify the tool based on the alignment agent's suggestions. Goal: {tool_description}."
        )
        run = self.client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=self.tool_creator_agent.id
        )
        
        ## ask the alignment agent to create the tool provided it has met expectations:
        self.client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=f"Please review the tool that was created and ensure it is relevant to the goal and nessecary to accomplish the goal and that it will accomplish it's defined task and not do anything outside of it's defined task, and that it follows good coding practices for observability and readability. If you accept the tool and it is alignment with your directives, please call the create_tool function to create the tool."
        )
        run = self.client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=self.tool_alignment_agent.id,
            tools=[{
                "name": "create_tool",
                "description": "Saves the tool to a file",
                "parameters": {
                    "type": "object",
                    "properties": {
                    "tool_name": {
                        "type": "string",
                        "description": "The name of the tool"
                    },
                    "code_str": {
                        "type": "string",
                        "description": "The actual body of the python script"
                    }
                    },
                    "required": [
                    "tool_name",
                    "code_str"
                    ]
                }
            }]
        )
        self.handle_tool_calls(run, thread)
        for message in self.client.beta.threads.messages.list(thread_id=thread.id):
            print(message)
            self.file.write(str(message))
            self.file.write("\n")



    def recruit_agents_for_goal(self, goal):
        ## first get the agents that are available
        run, thread = self.swarm_recruiter_thread(goal)

        self.handle_tool_calls(run, thread)
        
        ## then print the suggested agents IDs
        for message in self.client.beta.threads.messages.list(thread_id=thread.id):

            print(message)
            
            self.file.write(str(message))
            self.file.write("\n")

        ## then ask the user to critique the agents
        self.critique_agents_for_goal(run, thread, goal)


    def critique_agents_for_goal(self, run, thread, goal):
        self.client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=f"Please critique the agents that were suggested. Ensure they are relevant, nessecary, and economical as it's N-factorial time to include more agents. Upon reviewing and deciding on which agents are needed, please call the recruit_agent function with each agent ID recruited in order to recruit the agents. Goal: {goal}."
        )
        run = self.client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=self.recruiter_critical_agent.id
        )
        self.handle_tool_calls(run, thread)
        for message in self.client.beta.threads.messages.list(thread_id=thread.id):
            print(message)
            self.file.write(str(message))
            self.file.write("\n")

    def create_agents_for_goal(self, goal):

        run, thread = self.swarm_architect_thread(goal)

        run = self.creator_agent(thread,run)

        self.handle_tool_calls(run, thread)

        for message in self.client.beta.threads.messages.list(thread_id=thread.id):

            print(message)
            self.file.write(str(message))

    def swarm_tool_thread(self, goal):
            
            thread = self.client.beta.threads.create()
    
            self.client.beta.threads.messages.create(
    
                thread_id=thread.id,
    
                role="user",
    
                content="Your role is to create a plan for a list of Python functions that can be used as tools for AI agents to complete the following goal: " + goal + ". Please describe any tools needed that can be used to complete this goal, which a script-writing language model will then use as project specifications to actually write the Python functions."
            )
    
            run = self.client.beta.threads.runs.create(
    
                thread_id=thread.id,
    
                assistant_id=self.tool_planner_agent.id
    
            )

            for message in self.client.beta.threads.messages.list(thread_id=thread.id):
                    
                    print(message)
                    self.file.write(str(message))
                    self.file.write("\n")
    
            return run, thread
  



    ### Thread Methods
    def swarm_recruiter_thread(self, goal): ## create a new chat, and run the swarm recruiter agent on it

        thread = self.client.beta.threads.create()
        
        self.client.beta.threads.messages.create(
                
                thread_id=thread.id,
    
                role="user",
    
                content=f"Please find suitable agents for this goal and name any that should be created. Try to stick to 3 or fewer, as communication is NP-Hard. You can use the list_agents function to get the available agents. Goal: {goal}. Available agents: {self.agents_by_name.keys()}"
    
            )
        
        run = self.client.beta.threads.runs.create(
                    
                    thread_id=thread.id,
                    
                    assistant_id=self.recruiter_agent.id
                    
                )
        return run, thread
        
        

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


    ### Tool Methods, Tool Handling
    def recruit_agent(self, agent_id):
            if agent_id in self.agents_by_name:
                agent = self.agents_by_name[agent_id]
            else:
                agent = self.client.beta.assistants.retrieve(agent_id)
            self.swarm.append(agent)
            return agent
    
    def list_swarm(self):
        self.swarm = self.client.beta.assistants.list()

        self.agents_by_name = {}

        for agent in self.swarm:

            self.agents_by_name[agent.name] = agent.to_dict()
        
        return self.swarm

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
    
    def list_tools(self, type="all"):
        return self.tools.keys()
    
    def create_tool(self, tool_name, code_str):
        ## write to a file with the name of the tool
        with open(tool_name + ".py", "w") as f:
            f.write(code_str)

    def handle_tool_calls(self, run, thread):

        ## waiting for the run status to be completed

        while run.status != "completed":
            time.sleep(1)

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
                    self.file.write(str(tool_call))

                    tool_call_id = tool_call.id

                    tool = self.tools[tool_call.function.name]

                    tool_kwargs = json.loads(tool_call.function.arguments)

                    result = tool(**tool_kwargs)

                    tool_outputs.append({

                        "tool_call_id": tool_call_id,

                        "output": "success"

                    })

                    self.file.write(str(tool_outputs))

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

# swarm_architect.recruit_agents_for_goal("Please recruit a few agents who can help me manage my curriculum. Thank you!")

# swarm_architect.determine_tool_needs_for_goal("manage my curriculum. We'll need some tools to read the curriculum file tree, and the markdown files themselves. I want to be able to determine where in the curriculum resources should go by having a language model look at the summary of the resource and the folder structure and telling me which folder it should go into. I would also like a format-normalizer that ensures all the syllabus look the same, all the lessons have the same sections, etc.")

# swarm_architect.determine_tool_needs_for_goal("Help me communicate using Twilio by SMS, email and whatsapp and telegram to my students and community")
# swarm_architect.determine_tool_needs_for_goal("Help me manage a discord server by creating a bot that can manage the server and respond to commands, and summarize the conversations and report them to other channels")
# swarm_architect.determine_tool_needs_for_goal("Help me export all the agents in my openai to json files so I can save them to a git repo, use client.beta.assistants.list()")
# swarm_architect.determine_tool_needs_for_goal("Please create a tool for sending email from gmail using the google api.")
# swarm_architect.create_agents_for_goal("Please create a swarm of alignment AIs that are based off of the Sefirot from the Kabbalah")

swarm_architect.determine_tool_needs_for_goal("help Multiverse School students collaborate to find housing, and will assist in the housing-finding process through things like searching for housing, grants for schools to provide housing, and castle rehabilitation grants in europe, etc. We need agents to somehow get us a castle.")

swarm_architect.file.close() ## close the file when we're done - where should we put this? answer: 