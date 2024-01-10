from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import StringPromptTemplate
from langchain import OpenAI, LLMChain
from typing import List, Union
from langchain.schema import AgentAction, AgentFinish
from langchain.memory import ConversationBufferWindowMemory
from api.send_to_notion import write_to_notion

import re
from simple_chalk import chalk, red, green
class Classifier:
    
    def __init__(self):

        self.score = self.title = self.text = self.summary = self.timestamp = ""

        def print_title_and_summary(input):
            if input != "Not Important":
                print(chalk.red("\n" + input))
                self.title, self.summary = parse_string(input)
                if all_fields_ready():
                    write_to_notion(self.title, self.summary, self.text, self.score)
                    self.score = self.title = self.text = self.summary = ""
                    return("I have successfully completed all my tasks. Go me!")
                else:
                    if self.score == "":
                        return("I have to sumbit the importance score now")
                    elif self.timestamp == "":
                        return("I have to sumbit the timestamp now")

        def parse_string(s):
            parts = s.split(', ')
            title = parts[0]
            summary = parts[1]
            return (title, summary)
        
        def all_fields_ready():
            if self.title != "" and self.text != "" and self.summary != "" and self.score != "" and self.timestamp !="":
                return True
        
        def submit_importance_score(score):
            print(chalk.red("\n" + score))
            try:
                self.score = int(score)
            except Exception as e: 
                return("I accidentally sumbitted the score in an invalid format!!!")
            if all_fields_ready():
                write_to_notion(self.title, self.summary, self.text, self.score)
                self.score = self.title = self.text = self.summary = ""
                return("I have successfully completed all my tasks. Go me!")
            else:
                if self.summary == "" or self.title == "":
                    return("I have to sumbit the summary and title now.")
                elif self.timestamp == "":
                    return("I have to sumbit the timestamp now")
            
        def submit_time_and_date(tAndD): 
            print(chalk.red("\n" + tAndD))
            try:
                self.timestamp = tAndD + ""
            except Exception as e: 
                return("I accidentally sumbitted the score in an invalid format!!!")
            if all_fields_ready():
                write_to_notion(self.timestamp, self.title, self.summary, self.text, self.score)
                self.score = self.title = self.text = self.summary = self.timestamp = ""
                return("I have successfully completed all my tasks. Go me!")
            else:
                if self.summary == "" or self.title == "":
                    return("I have to sumbit the summary and title now.")
                elif self.score == "":
                    return("I have to sumbit the importance score now")
                

        self.tools = [
            Tool.from_function(
            func = print_title_and_summary,
            name = "Submit_Summary_and_Title",
            description ="""Use this to sumbit your title and summary. Format your input as such: title, summary """
            ),
            Tool.from_function(
            func = submit_importance_score,
            name = "Submit_Importance_Score",
            description="""Use this to sumbit the importance of the conversation. Input should be a number between 1 and 10."""
            ),
             Tool.from_function(
            func = submit_time_and_date,
            name = "Submit_time_and_date",
            description="""Use this to sumbit the time and date of the conversation."""
            ),
        ]

        self.tool_names = [tool.name for tool in self.tools]

        self.template_with_history = """You are an AI whose job it is to decide if a conversation is important or not. 

            If you think the conversation is important, use the Submit_Importance_Score to sumbit how important it is on a scale of 0-10. 

            Then, use the Submit_Summary_and_Title tool to sumbit a title and a summary for the conversation.
            
            The title should be a couple words. The summary should be a short sentence. 

            If there are any people names in the conversation, use the name of the person in the title.

            Here is the format for your answer: 

            Thought: The importance of the conversation on a scale of 0-10
            Action: {tool_names}
            Action Input:
            Observation: the result of the action
            ... (this Thought/Action/Action Input/Observation can repeat N times)
            Thought: I now know the final answer
            Final Answer: Sumbitted or not important

            Here is the provided tool:
            
            {tools}

            Previous conversation history:
            {history}

            New question: {input}
            {agent_scratchpad}"""
        
        self.prompt_with_history = self.CustomPromptTemplate(
            template=self.template_with_history,
            tools=self.tools,
            input_variables=["input", "intermediate_steps", "history"]
        )
        self.output_parser = self.CustomOutputParser()

        self.llm = OpenAI(temperature=0)

        self.llm_chain = LLMChain(llm=self.llm, prompt=self.prompt_with_history)

        self.agent = LLMSingleActionAgent(
            llm_chain=self.llm_chain, 
            output_parser=self.output_parser,
            stop=["\nObservation:"], 
            allowed_tools=self.tool_names
        )

        self.memory=ConversationBufferWindowMemory(k=2)

        self.agent_executor = AgentExecutor.from_agent_and_tools(agent=self.agent, tools=self.tools, verbose=True, memory=self.memory)

    class CustomPromptTemplate(StringPromptTemplate):
        # The template to use
        template: str
        # The list of tools available
        tools: List[Tool]
            
        def format(self, **kwargs) -> str:
            intermediate_steps = kwargs.pop("intermediate_steps")
            thoughts = ""
            for action, observation in intermediate_steps:
                thoughts += action.log
                thoughts += f"\nObservation: {observation}\nThought: "
            kwargs["agent_scratchpad"] = thoughts
            kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
            kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
            return self.template.format(**kwargs)

    class CustomOutputParser(AgentOutputParser):
        def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
            if "Final Answer:" in llm_output:
                return AgentFinish(
                    return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                    log=llm_output,
                )
            regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
            match = re.search(regex, llm_output, re.DOTALL)
            if not match:
                raise ValueError(f"Could not parse LLM output: `{llm_output}`")
            action = match.group(1).strip()
            action_input = match.group(2)
            return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)

    def intake(self, input_str):
       self.text = input_str
       return self.agent_executor.run(input_str)

