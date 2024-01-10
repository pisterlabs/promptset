from langchain.llms import Ollama
from langchain.tools import DuckDuckGoSearchRun
from crewai import Agent, Task, Crew, Process

ollama_openhermes=Ollama(model='openhermes:latest')

search_tool=DuckDuckGoSearchRun()

researcher=Agent(
    role='Senior Research Analyst',
    goal='Uncover cutting-edge developments in AI and data science in',
    backstory="""You are a Senior Research Analyst at a leading tech think tank.
  Your expertise lies in identifying emerging trends and technologies in AI and
  data science. You have a knack for dissecting complex data and presenting
  actionable insights.""",
    verbose=True,
    allow_delegation=False,
    tools=[search_tool],
    llm=ollama_openhermes
)

writer=Agent(
    role='Tech Content Strategist',
    goal='Craft compelling content on tech advancements',
    backstory="""You are a renowned Tech Content Strategist, known for your insightful
  and engaging articles on technology and innovation. With a deep understanding of
  the tech industry, you transform complex concepts into compelling narratives.""",
    verbose=True,
    allow_delegation=False,
    llm=ollama_openhermes
)

research_task = Task(
    description="""Conduct a comprehensive analysis of the latest advancements in AI in 2024.
  Identify key trends, breakthrough technologies, and potential industry impacts.
  Compile your findings in a detailed report. Your final answer MUST be a full analysis report""",
    agent=researcher
)

writer_task = Task(
    description="""Using the insights from the researcher's report, develop an engaging blog
  post that highlights the most significant AI advancements.
  Your post should be informative yet accessible, catering to a tech-savvy audience.
  Aim for a narrative that captures the essence of these breakthroughs and their
  implications for the future. Your final answer MUST be a full blog post of at least 500 words.""",
    agent=writer
)

crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, writer_task],
    verbose=2,
    process=Process.sequential
)

result = crew.kickoff()

print("######################")
print(result)