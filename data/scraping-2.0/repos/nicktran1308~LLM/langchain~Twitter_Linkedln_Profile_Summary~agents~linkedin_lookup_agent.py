# ------------------------------- Import Libraries -------------------------------
from tools.tools import get_profile_url
from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, Tool, AgentType

# ------------------------------- Linkedln Lookup Function ------------------------------- 
def linkedin_lookup(name: str) -> str:
    # Initialize the ChatOpenAI model with specific parameters
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    
    # Define the prompt template for the agent's request
    template = """
       Given the name {name_of_person}, find a link to their LinkedIn profile page.
       The final answer should contain only the URL."""
    
    # Define tools available for the agent, in this case to crawl Google for LinkedIn profiles
    tools_for_agent_linkedin = [
        Tool(
            name="Google search for LinkedIn profile page",
            func=get_profile_url,
            description="Useful for retrieving the LinkedIn Page URL",
        ),
    ]
    
    # Initialize the agent with specified tools and settings
    agent = initialize_agent(
        tools_for_agent_linkedin,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )
    
    # Create a prompt template object with the given format
    prompt_template = PromptTemplate(
        input_variables=["name_of_person"], template=template
    )
    
    # Use the agent to run the prompt and get the LinkedIn profile URL
    linkedin_url = agent.run(prompt_template.format_prompt(name_of_person=name))
    
    return linkedin_url

