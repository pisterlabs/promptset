import json 
import openai
from .Agent import Agent
import importlib
import glob
from .PromptTemplate import PromptTemplate


# TODO:
# Add handle exceptions
# Constructor should take in the list of tools as part of the custom information

class ToolSelectionAgent(Agent):
    '''
    Chooses the best tool to answer a question
    '''

    def __init__(self, company_info) -> None:
        self.tools = company_info["tools"]
        self.company_desc = company_info["company_desc"]
        self.company_info = company_info
        AGENTS = {}
        files = glob.glob("Agents/*Agent.py")
        for file in files: # Import all the agents 
            class_name = file.replace(".py", "").replace("Agents/", "")
            module = importlib.import_module("Agents." + class_name)
            class_ = getattr(module, class_name)
            AGENTS[class_name] = class_
        self.agents = AGENTS

    def __str__(self) -> str:
        return "Agent for tool selection"

    @property
    def prompt(self) -> None:
        s = ''' You are a Task Manager for {company_desc}. You are tasked with choosing the best tool to assist a cusomter given the conversation history so far. Output only the name of the tool and nothing else. Your output has to be exactly the same as one of the options below. Do not give me any reasoning.
        
        You have access to the following tools:
        {tools_list}

        Given the entire conversation history, the question and the tools available to you, choose the best tool to assist the customer.
        The conversation history might have used other tools. However, you should choose the tool to answer the most recent query given the context of the entire conversation history.
        If the conversation history is empty, this is a new conversation. Your choice should then be only determined on the query. Always choose your best answer even if you are not confident.
        If the user is replying to a question from the assistant previously, put more attention on the previous question and the user's reply to it when choosing the tool.
        
        Conversation History:
        {conversaton_history}

        Output your chain of thought not just the final answer. When you have a final answer end your chain of thought with "ANSWER: [TOOL NAME]". If you think the question will need 2 tools sequentially, output the first one. 

        Thus, your output should end with "ANSWER: [TOOL NAME]" where TOOL NAME is one of the tools in {tools_master}
        If there is no clear question, choose the tool that is most relevant to the conversation history.
        '''
        return PromptTemplate(s)

    def generate_answer(self, query, chat_history = list()) -> dict:
        '''
        Generate answer to a query, chooses the right task
        '''
        with open('Agents/Tools.json', 'r') as f:
            tools_master = json.load(f)

        if len(self.tools) == 1:
            return f"ANSWER: {self.tools[0]}", {} # If there is only one tool, just use that tool
        
        if self.tools == []:
            self.tools = tools_master.keys()

        usingtools = '\n\n'.join([tools_master[tool] for tool in self.tools])
        formatted_history = "\n".join([f"{message['role']}: {message['content']}" for message in (chat_history + [{"role": "user", "content": query}])])
        processed_prompt = self.prompt.format({
            "company_desc": self.company_desc,
            "tools_list": usingtools,
            "conversaton_history": formatted_history,
            "tools_master": ", ".join([tools_master[tool] for tool in self.tools])
        })

        ans = openai.ChatCompletion.create(
        model= "gpt-3.5-turbo",
        messages = [{"role": "system", "content": processed_prompt},
                {"role": "user", "content": query + " \nWhat tool should I use?"}
            ],
        temperature=0,
        # max_tokens=5,
        )
        return ans["choices"][0]["message"]["content"], ans["usage"]
    
    def route(self, query: str, chat_history: list) -> tuple:
        '''
        Route a query to the right tool
        '''
        ans, usage = self.generate_answer(query, chat_history)
        chosen = ans
        if "ANSWER:" in chosen:
            chosen = chosen.split("ANSWER: ")[1].replace(".", "").strip()
        else:
            chosen = chosen.replace(".", "").strip() 
        return self.agents[chosen + "Agent"](company_info = self.company_info), usage
        
