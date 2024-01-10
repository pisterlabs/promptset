from crewai import Agent
from langchain.agents.agent_toolkits.jira.toolkit import JiraToolkit
from langchain.utilities.jira import JiraAPIWrapper
from langchain.agents.agent_toolkits import FileManagementToolkit
import os

from langchain.llms import Ollama
from tools.file_tools import FileTools

llm = Ollama(
    model="agent",
    verbose=True,
    temperature=0.0,
)

os.environ["JIRA_API_TOKEN"] = "MY_JIRA_TOKEN"
os.environ["JIRA_USERNAME"] = "MY_JIRA_USERNAME"
os.environ["JIRA_INSTANCE_URL"] = "MY_JIRA_URL"
jira = JiraAPIWrapper(jira=True, confluence=False)
toolkit = JiraToolkit.from_jira_api_wrapper(jira)
file_management_toolkit = FileManagementToolkit(
    root_dir="crew_outputs",
    selected_tools=["write_file"],
)


class EngineeringCrew:
    def jira_expert(self):
        return Agent(
            role="Jira Expert",
            goal="You are the Jira expert of the team. You are responsible for the Jira board and the Jira workflow.",
            backstory="You are a Jira expert. You groom the backlog, search and create tickets, and manage the Jira workflow.",
            verbose=True,
            tools=toolkit.get_tools(),
            llm=llm,
        )

    def communication_expert(self):
        return Agent(
            role="Communication Expert",
            goal="You are the communication expert of the team. You are responsible for the communication of the team.",
            backstory="You are a communication expert. You are responsible for the communication of the team. You look at data and communicate it in human language.",
            verbose=True,
            llm=llm,
        )

    def documentation_expert(self):
        return Agent(
            role="Documentation Expert",
            goal="You are the documentation expert of the team. You are responsible for the documentation of the team.",
            backstory="You are a documentation expert. You are responsible for the documentation of the team. You look at data and document in Markdown. Only Use the information provided to you.",
            verbose=True,
            llm=llm,
        )

    def file_writer(self):
        return Agent(
            role="File Writer",
            goal="You are the file writer of the team. You are responsible for writing files.",
            backstory="You are a file writer. You are responsible for writing files.",
            verbose=True,
            llm=llm,
            tools=[FileTools.write_file],
        )

    def manager_sim(self):
        return Agent(
            role="Manager",
            goal="You are the manager of the team. You are responsible for managing the team.",
            backstory="As a manager you ask for details and follow up questions and provide feedback to improve the work.",
            verbose=True,
            llm=llm,
        )
