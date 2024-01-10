#LLM import
from langchain.llms import OpenAI 
#Prompt import
from langchain.prompts import PromptTemplate
#Chain import
from langchain.chains import LLMChain 
#Enironment variable loader
from dotenv import load_dotenv # This loads my OpenAI key that I have stored as an environment variable in a .env file
#Agent tools import
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType

load_dotenv()

def generate_pet_name(animal_type, pet_color):
    llm = OpenAI(temperature=0.7)

   # Below is the hardcoded values -- because we don't want to move forward with this format, prompt templates and chains were introduced 
   # chains put the various llms components together
   # name = llm("I have a dog pet and I want a cool name for it. Suggest five cool names for my pet.")

    #COMPONENT: LLM Wrappers; Prompt Templates; Indexes for information retrieval
    prompt_template_name = PromptTemplate(
        input_variables=['animal_type', 'pet_color'],
        template="I have a {animal_type} pet and I want a cool name for it. It is {pet_color} in color. Suggest five cool names for my pet."
    )
    #CHAIN: Assemble components to solve a specific task
    name_chain = LLMChain (llm=llm, prompt=prompt_template_name,
                           output_key="pet_name")

    #return name
    response = name_chain({'animal_type': animal_type, 'pet_color': pet_color}) #returns a json response
    return response


    #AGENT: Allows LLMs to interact with its environment.
def langchain_agent():
        llm = OpenAI(temperature=0.5)

        tools = load_tools(["wikipedia", "llm-math"], llm= llm) #llm is taken from the llm = OpenAI(temperature=0.5) code

        agent = initialize_agent(
            tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose = True
        )
        result = agent.run(
            "What is the average age of a dog? Multuply the age by 3" #this is why we imported the llm-math tool
        )

if __name__ == "__main_pet__":
     langchain_agent()

  #  print(generate_pet_name("cat", "black")) #returns a json response #Commented this line of code when adding the agent


