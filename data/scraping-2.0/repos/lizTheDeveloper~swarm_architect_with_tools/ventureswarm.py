import openai 
import json
import datetime
import time

class VentureSwarm:
    def __init__(self):

        self.swarm = []
        self.client = openai.Client()

        self.brainstorming_agent_id = "asst_8KyCNvb2ZQ7qEXUEe1hayTFf"
        self.manager_agent_id = "asst_STHyWmVD1ICbgpRyK1eoZuvp"
        self.reviewer_agent_id = "asst_AElcAx1wsaRpur9VEMQyeuoG"
        self.vc_agent_id = "asst_8wtaubkTo16WLgZ0awmUZNHz"

        self.brainstorming_agent = self.client.beta.assistants.retrieve(self.brainstorming_agent_id)
        self.manager_agent = self.client.beta.assistants.retrieve(self.manager_agent_id)
        self.reviewer_agent = self.client.beta.assistants.retrieve(self.reviewer_agent_id)
        self.vc_agent = self.client.beta.assistants.retrieve(self.vc_agent_id)

        
        self.tools = {
            "create_agent": self.create_agent,
            "list_swarm": self.list_swarm,
            "recruit_agent": self.recruit_agent,
            "list_tools": self.list_tools,
            "create_tool": self.create_tool,
            "request_tool": self.request_tool
        }


        self.timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.file = open("./chatlogs/" + self.timestamp, "w")
        self.file.write("Swarm Recruiter Chatlog\n")
        self.file.close()
        self.file = open(self.timestamp, "a")

    
    
    def brainstorm_venture(self,category):
        ## create a new thread with a request for the tool between the alignment agent and the tool creator agent
        thread = self.client.beta.threads.create()
        
        self.client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=f"Please brainstorm a venture in the {category} category."
        )

        run = self.client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=self.brainstorming_agent.id
        )
        self.handle_tool_calls(run, thread)
        for message in self.client.beta.threads.messages.list(thread_id=thread.id):
            print(message)
            self.file.write(str(message))
            self.file.write("\n")
        
        ## ask the reviewer agent to review the venture
        self.client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=f"Please review the venture that was created and ensure it is relevant to the category and nessecary to accomplish the goal and that it will accomplish it's defined task and not do anything outside of it's defined task, and that it follows good coding practices for observability and readability."
        )
        run = self.client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=self.reviewer_agent.id
        )
        self.handle_tool_calls(run, thread)
        for message in self.client.beta.threads.messages.list(thread_id=thread.id):
            print(message)
            self.file.write(str(message))
            self.file.write("\n")
        
        ## ask the vc agent to review the venture
        self.client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=f"Please review the venture that was created and ensure it is relevant to the category and nessecary to accomplish the goal and that it will accomplish it's defined task and not do anything outside of it's defined task, and that it follows good coding practices for observability and readability."
        )
        run = self.client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=self.vc_agent.id
        )
        self.handle_tool_calls(run, thread)
        for message in self.client.beta.threads.messages.list(thread_id=thread.id):
            print(message)
            self.file.write(str(message))
            self.file.write("\n")
        
        ## ask the manager agent to review the venture
        self.client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=f"Please review the venture that was created and ensure it is relevant to the category and nessecary to accomplish the goal and that it will accomplish it's defined task and not do anything outside of it's defined task, and that it follows good coding practices for observability and readability."
        )
        run = self.client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=self.manager_agent.id
        )
        self.handle_tool_calls(run, thread)
        for message in self.client.beta.threads.messages.list(thread_id=thread.id):
            print(message)
            self.file.write(str(message))
            self.file.write("\n")



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
  


swarm = VentureSwarm()
swarm.brainstorm_venture("Medical Devices")
