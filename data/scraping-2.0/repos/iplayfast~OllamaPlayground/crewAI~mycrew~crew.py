import os
from crewai import Agent, Task, Crew, Process
from langchain.llms import Ollama

import duckduckgo_search
from langchain.tools import tool
import requests
from bs4 import BeautifulSoup

@tool
def latest_trends(topic: str) -> str:
    """
    Search news.google.com for trends on a given topic.
    This function takes a topic as a string, queries news.google.com for trends related to this topic,
    and returns the trends found as a string.
    """
    print(f"latest_trends tool called with topic: {topic}")  # Debugging statement
    # [Rest of the tool code]

    # Format the topic for URL encoding
    formatted_topic = topic.replace(" ", "%20")

    # Construct the search URL
    url = f"https://news.google.com/search?q=Trends%20on%20{formatted_topic}"

    # Perform the web request
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code != 200:
        return "Failed to retrieve data from news.google.com"

    # Parse the HTML content
    soup = BeautifulSoup(response.text, 'html.parser')

    # Extract trend data
    # The following is a placeholder; you'll need to adjust it based on the page structure
    trends = []
    for article in soup.find_all("article"):
        title = article.find("h3").get_text() if article.find("h3") else "No title"
        summary = article.find("p").get_text() if article.find("p") else "No summary"
        trends.append(f"Title: {title}\nSummary: {summary}\n")

    # Format the extracted trends
    trends_output = "\n".join(trends) if trends else "No trends found for the topic."

    return trends_output


ollama_mistral = Ollama(model="mistral")
# Pass Ollama Model to Agents: When creating your agents within the CrewAI framework, you can pass the Ollama model as an argument to the Agent constructor. For instance:

# Define your agents with roles and goals
researcher = Agent(
  role='Researcher',
  goal='Discover new insights',
  backstory="You're a world class researcher working on a major data science company",
  verbose=True,
  allow_delegation=False,
  llm=ollama_mistral, # Ollama model passed here
  # llm=OpenAI(temperature=0.7, model_name="gpt-4"). It uses langchain.chat_models, default is GPT4
  tools=[latest_trends,duckduckgo_search],
)
writer = Agent(
  role='Writer',
  goal='Create engaging content',
  backstory="You're a famous technical writer, specialized on writing data related content",
  verbose=True,
  allow_delegation=False,
  llm=ollama_mistral, # Ollama model passed here
)

# Create tasks for your agents
task1 = Task(description='Investigate the latest AI trends', agent=researcher)
task2 = Task(description='Write a blog post on AI advancements', agent=writer)

# Instantiate your crew with a sequential process
crew = Crew(
  agents=[researcher, writer],
  tasks=[task1, task2],
  verbose=True, # Crew verbose more will let you know what tasks are being worked on
  process=Process.sequential # Sequential process will have tasks executed one after the other and the outcome of the previous one is passed as extra content into this next.
)

# Get your crew to work!
result = crew.kickoff()
