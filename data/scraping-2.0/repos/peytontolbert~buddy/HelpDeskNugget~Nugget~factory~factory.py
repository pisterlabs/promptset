import openai
import os
from dotenv import load_dotenv
from gpt.chatgpt import ChatGPT
from memory.memory import Memory
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
llm_model = os.getenv("LLM_MODEL")

class ToolManager:
    def __init__(self):
        self.tools = {}

    def add_tool(self, tool_name, tool):
        self.tools[tool_name] = tool

    def get_tool(self, tool_name):
        return self.tools.get(tool_name)



class ToolRetriever:
    def __init__(self, tool_manager):
        self.tool_manager = tool_manager

    def remember_all_tools(self):
        # Retrieve all tools
        return self.tool_manager.get_tools()

    def remember_relevant_tools(self, task):
        # Placeholder for logic to retrieve relevant tools based on the task
        relevant_tools = []
        return relevant_tools

class Factory:
    def __init__(self):
        self._creators = {}
        self.tool_manager = ToolManager()

    def run_conversation(self,messages=[],results=None):
        og_messages = messages.copy()
        prompt={"role":"system","content":"Your job is only decide if this is a task or a question. Simply respond 'task' or 'question'."}
        checkmessages = messages.append(prompt)
        print(messages)
        response = ChatGPT.chat_with_gpt3(messages)
        print(response)
        if "question" in response:
            results = self.run_question(og_messages)
        elif "task" in response:
            results = self.run_task(og_messages)
        else:
            results = self.run_clarify(og_messages)
        return results
    
    def run_question(self, question):
        print(question)
        response = ChatGPT.chat_with_gpt3(question)
        return response
    
    def run_task(self, task):
        response = ChatGPT.chat_with_gpt3(task)
        return response
    
    def run_clarify(self, clarify):
        question={"role":"assistant","content":"I am not sure what you mean. Please clarify."}
        clarify+=question
        return clarify
    
    def create_agent(self,task):
        agent = Agent(task, self.tool_manager)
        return agent
    

class Agent:
    def __init__ (self, goal, tool_manager):
        self.goal = goal
        self.tasklist = []
        self.tool_manager = tool_manager
        self.state = None
        self.memory = Memory()
        self.tool_retriever = ToolRetriever(tool_manager)

    def create_task(self, task):
        # Retrieve related past episodes, knowledge, and tools
        related_past_episodes = self.memory.remember_related_episodes(task, k=3)
        related_knowledge = self.memory.remember_related_knowledge(task, k=3)
        tools = self.tool_retriever.remember_relevant_tools(task)

        # Prepare the tools' information
        tool_info = "\n".join([f"{tool.name}: {tool.description}" for tool in tools])

        newprompt = """
        You are AutoGPT, an autonomous agent with the ability to make decisions independently. Your goal is to complete the task assigned to you with minimal user intervention.

        [Step 1: Task Understanding]
        You have been assigned the following task:
        {task}

        [Step 2: Tool Identification]
        You can only use one tool at a time. Try to use the most appropriate tool for each step of the task. Remember to mark the task as completed using 'task_complete' and provide your answer in the 'args' field if you think you have finished the task. 

        Available Tools:
        {tools}

        [Step 3: Recall Past Experiences & Knowledge]
        {related_past_episodes}

        [RELATED KNOWLEDGE]
        {related_knowledge}

        [Step 4: Generate Idea]
        Come up with a strategy to complete the task. Consider the task requirements, the available tools, and the lessons from past tasks.

        [Step 5: Criticize Your Idea]
        Review your strategy. Is there any potential issue? Can you think of a better approach? Criticize your idea constructively.

        [Step 6: Explain Your Strategy]
        Explain your final strategy to the user. Why did you choose this approach? Why did you select these tools? 

        [Step 7: Execute the Task]
        Perform the task according to your strategy. Remember to adhere to the JSON RESPONSE FORMAT when describing your actions and their outcomes.

        JSON RESPONSE FORMAT:
        {{
            "observation": "(observation of [RECENT EPISODES])",
            "thoughts": {{
                "task": "(description of [YOUR TASK] assigned to you)",
                "knowledge": "(if there is any helpful knowledge in [RELATED KNOWLEDGE] for the task, summarize the key points here)",
                "past_events": "(if there is any helpful past events in [RELATED PAST EPISODES] for the task, summarize the key points here)",
                "idea": "(thought to perform the task)",
                "reasoning": "(reasoning of the thought)",
                "criticism": "(constructive self-criticism)",
                "summary": "(thoughts summary to say to user)"
            }},
            "action": {{
                "tool_name": "(One of the tool names included in [TOOLS])",
                "args": {{
                    "arg name": "value",
                    "arg name": "value"
                }}
            }}
        }}
        """.format(task=task, tools=tool_info, related_past_episodes=related_past_episodes, related_knowledge=related_knowledge)

        response = self.run_chat(newprompt)
        
        if response == "I give up and restart":
            return self.create_task(task)
        else:
            self.state = response
            return self.execute_task()
        
    def execute_task(self):
        while self.state != "done":
            response = self.run_chat("What's your next step?")
            if response == "I give up and restart":
                return self.create_task(self.goal)
            else:
                self.state = response
        return self.run_chat("What's your final answer?")
        
    def run_chat(self, prompt):
        return ChatGPT.chat_with_gpt3(prompt)