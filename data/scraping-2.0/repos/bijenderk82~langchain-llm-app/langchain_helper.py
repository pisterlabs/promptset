from langchain.llms import OpenAI
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType

load_dotenv()

def generate_pet_names(animal_type, pet_color):
    llm = OpenAI(temperature=0.7)
    
    prompt_template_name = PromptTemplate(
        input_variables=['animal_type', 'pet_color'],
        template="I have a {animal_type} pet, its has {pet_color} color, and I want a cool name for it. Suggest me five cool names for my pet"
    )
    
    name_chain = LLMChain(
        llm=llm, 
        prompt=prompt_template_name,
        output_key="pet_name"
    )

    response = name_chain({'animal_type': animal_type, 'pet_color': pet_color})
    
    return response


def langchain_agents():
    llm = OpenAI(temperature=0.5)
    tools = load_tools(["wikipedia", "llm-math"], llm=llm)
    
    agent = initialize_agent(tools=tools,llm=llm,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
    
    agent.run("What is the average age of dog? Multiply the age by 3")
    

if __name__ == "__main__":
    print(langchain_agents())
    # print(generate_pet_names("human", "black"))