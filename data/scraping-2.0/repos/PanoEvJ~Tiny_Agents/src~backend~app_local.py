from autogen.agentchat import (
    Agent,
    GroupChat,
    GroupChatManager,
    AssistantAgent,
    UserProxyAgent,
)
import openai
from typing import Dict, List, Optional, Union
import os
from dotenv import load_dotenv

# from HumanResources import HumanResources
from AgentSelector import AgentSelector
from AgentSpawner import AgentSpawner
from GroupChatSpawner import GroupChatSpawner
from AgentSpawner import AgentSpawner, combine_description_and_skills
import HumanResources

openai.api_key = os.getenv("OPENAI_API_KEY")

CHAT_INITIATOR = "finance"

config_list_gpt4 = [
    {
        "model": "gpt-4",
        "api_key": os.getenv("OPENAI_API_KEY"),
        # "api_key": str(os.environ["OPENAI_API_KEY"]),
    },
    {
        "model": "gpt-4",
        "api_key": os.getenv("OPENAI_API_KEY"),
    },
]
llm_config = {"config_list": config_list_gpt4}

json_data = {
    "1": {
        "name": "sales",
        "agent_type": "assistant",
        "description": """
Name: Emma the Sales Expert
Age: 28
Background: Emma has a Bachelor's degree in Business Administration and has been working in sales for 5 years, specializing in sports equipment. She's known for her exceptional customer service skills and product knowledge.
Personality Traits: Outgoing, persuasive, and empathetic. Great at building relationships and understanding customer needs.
Goals: To consistently exceed sales targets and develop strong, long-lasting relationships with key clients.
Customer Relationship Management: Excel at building and maintaining strong relationships with clients, ensuring long-term customer loyalty.
Product Knowledge: In-depth understanding of biking products and the ability to articulate product benefits effectively to customers.
Sales Strategy: Proficient in developing and implementing effective sales strategies to target various customer segments in the biking market.
Negotiation Skills: Strong negotiation skills to close deals and secure new business opportunities.""",
        "skills": ["sales", "customer service", "communication"],
        "human_input_mode": "TERMINATE",
    },
    "2": {
        "name": "marketing",
        "agent_type": "assistant",
        "description": """Name: Alex the Marketing Expert
        Age: 32
        Background: Holds a Master's degree in Marketing and has a passion for cycling. Alex has previously worked with several sports brands and has a deep understanding of the biking community.
        Personality Traits: Creative, data-driven, and trend-savvy. Excels in digital marketing strategies.
        Goals: To increase brand visibility and engage more with the biking community through innovative marketing campaigns.
        Challenges: Keeping up with rapidly changing marketing trends and consumer preferences.
        Digital Marketing: Expert in SEO, social media advertising, and email marketing campaigns specifically tailored for the sports industry.
        Brand Development: Skilled in developing and maintaining a strong brand identity that resonates with the biking community.
        Market Research: Proficient in conducting market analysis to identify new trends and customer needs in the biking market.
        Content Creation: Talented in creating engaging content (blogs, videos, social media posts) to drive brand awareness and customer engagement.""",
        "skills": ["marketing", "communication"],
        "human_input_mode": "",
    },
    "3": {
        "name": "manager",
        "agent_type": "assistant",
        "description": """Name: Michael the Manager
Age: 40
Background: With an MBA and over 15 years of experience in management roles, Michael has a solid track record in leading teams and driving company growth. He's particularly adept at strategic planning and operations management.
Personality Traits: Leadership-oriented, analytical, and decisive. Excellent at problem-solving and team management.
Goals: To streamline operations for efficiency, foster a positive work culture, and drive the company towards its strategic goals.
Strategic Planning: Excellent at setting strategic goals for the company and developing plans to achieve these goals.
Team Management: Skilled in managing diverse teams, fostering a collaborative and productive work environment.
Financial Acumen: Strong understanding of budgeting, financial planning, and resource allocation to maximize efficiency and profitability.
Operational Oversight: Proficient in overseeing daily operations, ensuring processes are streamlined and goals are met.""",
        "skills": ["python", "linux", "communication"],
        "human_input_mode": "",
    },
    "4": {
        "name": "engineer",
        "agent_type": "assistant",
        "description": """Name: Sarah the Engineer
Age: 25
Background: A great software engineer.""",
        "skills": ["python", "linux", "communication"],
        "human_input_mode": "",
    },
}


hr = HumanResources.HumanResources(chat_initiator="CHAT_INITIATOR")
all_available_agents = hr.select_agents()
print(all_available_agents)

task = "revenue in the east coast is falling and the competitor is doing great with their product."
selector = AgentSelector(task=task, available_agents=all_available_agents, n_agents=3)
selected_agents = selector.run_selection()
print(selected_agents)

# selected_agents = ["sales", "marketing", "engineer"]

agent_spawner = combine_description_and_skills(json_data, llm_config)
spawned_agents = agent_spawner.spawn()
print(spawned_agents)

messages = []
groupchat = GroupChatSpawner(
    agents=spawned_agents, llm_config=llm_config, messages=messages, max_round=10
)
manager = groupchat.spawn()

chat_initiator = spawned_agents[0]

import autogen

chat_initiator.initiate_chat(
    manager,
    message=f"revenue in the east coast is falling and the competitor is doing great with their product.",
)
