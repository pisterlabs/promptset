import langchain
from langchain.loaders import PublicWyscoutLoader
from langchain.templates import PromptTemplate
from langchain.models import LLM

# Create a new LangChain project.
project = langchain.Project()

# Add the PublicWyscoutLoader class to your project.
project.add_loader(PublicWyscoutLoader())

# Create a PromptTemplate that uses the PublicWyscoutLoader class to load data from Wyscout.
prompt_template = PromptTemplate(
    "What can you tell me about the player with the ID {player_id}?",
    loader=PublicWyscoutLoader(),
)

# Create an LLM that can be used to generate responses using the data loaded from Wyscout.
llm = LLM()

# Create a chain that uses the PromptTemplate and LLM to give a response using the data available in Wyscout.
chain = langchain.Chain(
    prompt_template=prompt_template,
    llm=llm,
)

# Run the chain.
response = chain.run()

# Print the response.
print(response)