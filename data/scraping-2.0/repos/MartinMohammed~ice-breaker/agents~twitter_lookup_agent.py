from langchain.prompts import PromptTemplate
from langchain.agents import initialize_agent, AgentType, Tool
from langchain.chat_models import ChatOpenAI
from tools.tools import get_profile_url


def twitter_lookup_agent(name: str) -> str:
    # Setting up the language model (LLM) from ChatOpenAI with specific configurations
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

    # The template is the expected response format that the system needs to generate – in this case, a LinkedIn URL.
    template = """
    given the full name {name_of_person} I want you to get me the link to their Twitter profile page, and extract from it their username.
    In your response the URL should be inside square braces e.g. '[@elonmusk]' """  # <-- Output indicator (how the response should look like)

    # Tool creation for the Agent - a function needed to be called when solving the main problem.
    tools_for_agent = [
        Tool(
            name="Crawl Google 4 Twitter profile page",
            func=get_profile_url,  # The specific function to retrieve the LinkedIn URL needs to be defined here.
            description="useful for when you need to get the Twitter page URL",
        )
    ]

    # Explanation of paradigms used: Zero Shot and React for reasoning and acting.
    # Zero Shot allows for more creative results, but potentially less accuracy.
    # React is a paradigm for controlling the reasoning process to enhance correct responses.

    # Prompt template for generating the request for the LinkedIn profile URL.
    linkedin_profile_url_prompt_template = PromptTemplate(
        input_variables=["name_of_person"],
        template=template,
    )

    # Initializing the Agent with specific characteristics, including logging for transparency (verbose=True).
    agent = initialize_agent(
        tools=tools_for_agent,
        verbose=True,  # Log every step the agent takes for transparency.
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        llm=llm,
    )

    # Running the Agent to obtain the LinkedIn profile URL for the provided individual's name.
    response = agent.run(
        linkedin_profile_url_prompt_template.format_prompt(name_of_person=name)
    )
    twitter_username = response.split("[")[1].split("]")[0]
    return twitter_username
