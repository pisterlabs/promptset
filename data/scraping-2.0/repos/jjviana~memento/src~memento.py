import os
import openai
from prompt_toolkit import prompt
from settings import Settings
from prompts import PromptsManager
from language_models import OpenAIChat,BedrockClaude
from language_models import LoopDetector
from language_models import Message
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from tools_manager import ToolsManager
import argparse
from exceptions import exception_handler
from protocol import EmbeddedMessage,parse_message
class Memento:
    def __init__(self, tools_manager, prompts_manager,model,logger):
        self.tools_manager = tools_manager
        self.prompts_manager = prompts_manager
        self.model = model
        self.logger = logger
        self.model.set_gc_manager(self.manage_gc)
        self.session_cost = 0.0
        self.iteraction_costs = []
    
    def format_tool_output(self, tool_name, tool_output):
        return "{{FROM:"+tool_name+" TO:memento}}\n" + tool_output + "\n{{END}}"

    def start(self) -> str:
        starting_prompt = self.prompts_manager.get_prompt("start")
        response = self.model.initial_prompt(starting_prompt)
        return self.process_response(response)
    
    def process(self,message:str) -> str:
        
        response = self.model.input(self.format_tool_output("user",message))
        return self.process_response(response)
    
    def process_response(self,response:str) -> str:

        interaction_cost = 0.0
        while True:
            # Some models tend to return a response with leading whitespace
            messages = self.model.get_messages()
            response_cost = messages[-1].cost
            interaction_cost = interaction_cost + response_cost

            embedded_messages = []
            try:
                embedded_messages = parse_message(response)
            except Exception as e:
                if response == LoopDetector.LOOP_DETECTED_SENTINEL:
                    response = self.model.input(self.format_tool_output("system","You are repeating yourself. Please check for any error messages, check command syntax or try a different approach."))  
                    continue
                # Some agents tend to respond without the proper formatting.
                # Usually this happens when formatting a response to the user
                # (the prompt doesn't always override the model tendency to reply directly)
                # In this case we will cheat and format the message as if the agent sent the 
                # message to the user.
                response = f"{{FROM:memento TO:user}}{response}{{END}}"
                embedded_messages = parse_message(response)

            # If for some reason the agent sends more than one embedded response message 
            # (shouldn't happen given generation stop condition) 
            # we will consider just the first one.
            embedded_message = embedded_messages[0]
            tool_name,tool_output = self.tools_manager.process_command(embedded_message.message)
            if tool_name == "user":
                self.session_cost = self.session_cost + interaction_cost
                self.iteraction_costs.append(interaction_cost)
                return tool_output

            tool_output = self.format_tool_output(tool_name, tool_output)
            response = self.model.input(tool_output)

    def manage_gc(self,model):
       
        raise NotImplementedError("Maximum tokens used.")

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    available_models = ["gpt-3.5-turbo","gpt-3.5-turbo-16k","gpt-4","anthropic.claude-v2"]
    parser.add_argument("--model", type=str, required=True, help=f"Model type. Available models: {available_models}")

    language_model_factory = None
    args = parser.parse_args()
    if args.model.startswith("gpt-"):
        language_model_factory =lambda logger: OpenAIChat(args.model,logger)
    elif args.model == "anthropic.claude-v2":
        language_model_factory = lambda logger: BedrockClaude("anthropic.claude-v2",logger)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    session = PromptSession(history=FileHistory("history.txt"))

    log_file= open("memento.log", "a")
    def log(message: str):
        log_file.write(message + "\n")
        log_file.flush()
    def logger(instance: str):
        return lambda message: log(instance+"->"+message)

    prompts_manager = PromptsManager(".")

    all_agents = []
    agent_session_costs =[]

    def new_agent(tools_manager,prompts_manager,instance: str):
        model = LoopDetector(language_model_factory(logger(instance)))
        agent = Memento(tools_manager,prompts_manager,model,logger(instance))
        all_agents.append(agent)
        agent_session_costs.append(0.0)
        return agent


    tools_manager = ToolsManager("tools",new_agent)


    top_level_agent = new_agent(tools_manager,prompts_manager,"top_level")
    response = top_level_agent.start()

    while True:
        print(response)
        interaction_costs = [all_agents[i].session_cost - agent_session_costs[i] for i in range(0,len(all_agents))]
        agent_session_costs = [all_agents[i].session_cost for i in range(0,len(all_agents))]
        total_iteraction_cost = sum(interaction_costs)
        total_session_cost = sum(agent_session_costs)
        iteraction_cost = "{:.6f}".format(total_iteraction_cost)
        session_cost = "{:.6f}".format(total_session_cost)
        print(f"====\nIteraction cost: {iteraction_cost}\nSession cost: {session_cost}\n====")
        message = session.prompt(">")
        response = top_level_agent.process(message)
