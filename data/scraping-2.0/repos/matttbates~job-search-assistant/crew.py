from langchain.llms import Ollama
from crewai import Agent, Task, Crew, Process

ollama_openhermes = Ollama(model="openhermes")

from langchain.tools import DuckDuckGoSearchRun
search_tool = DuckDuckGoSearchRun()

researcher = Agent(
    role='Researcher',
    goal='Research Android Job openings',
    backstory='You are a job search assistant specialized in finding relevant job openings',
    verbose=True,
    allow_delegation=False,
    llm=ollama_openhermes,
    tools=[search_tool]
)

writer = Agent(
    role='Writer',
    goal='Write compelling cover letters',
    backstory='You are a Job search assistant specialized in writing cover letters that land jobs.',
    verbose=True,
    allow_delegation=False,
    llm=ollama_openhermes
)

task1 = Task(
    description='Investigate LinkedIn for the latest Android job offers around Calgary and Canadian remote positions.',
    agent=researcher
)

task2 = Task(
    description='Write a compelling cover letter to land a job interview.',
    agent=writer
)

crew = Crew(
    agents=[researcher, writer],
    tasks=[task1, task2],
    verbose=2,
    process=Process.sequential
)

result = crew.kickoff()