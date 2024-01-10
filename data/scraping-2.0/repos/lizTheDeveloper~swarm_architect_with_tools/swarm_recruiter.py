import openai 
import json
import datetime

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

        ## build out the swarm if we don't have any agents
        self.swarm_architect = "asst_HmxHtEXXuAAzBJYen2udhoFJ"

        self.agent_creator = "asst_47ywjR3NxwhYOhRhllYplRs9"

        self.swarm_architect_agent = self.client.beta.assistants.retrieve(self.swarm_architect)

        self.agent_creator_agent = self.client.beta.assistants.retrieve(self.agent_creator)


        self.tools = {

            "create_agent": self.create_agent,
            "list_swarm": self.list_swarm,
            "recruit_agent": self.recruit_agent

        }

        self.swarm = self.client.beta.assistants.list()

        self.agents_by_name = {}

        for agent in self.swarm:

            self.agents_by_name[agent.name] = agent

        self.timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.file = open(self.timestamp, "w")
        self.file.write("Swarm Recruiter Chatlog\n")
        self.file.close()
        self.file = open(self.timestamp, "a")

    ### Flow Methods
    def recruit_agents_for_goal(self, goal):
        ## first get the agents that are available
        run, thread = self.swarm_recruiter_thread(goal)

        self.handle_tool_calls(run, thread)
        ## create the file with the timestamp 
        
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

swarm_architect.recruit_agents_for_goal("Please recruit a few agents who can help me make arbitrary python tools given a text-based request. Thank you!")







swarm_architect.file.close() ## close the file when we're done - where should we put this? answer: 