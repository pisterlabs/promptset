import os
from dotenv import load_dotenv  
import sys

from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.indexes import VectorstoreIndexCreator
from langchain.document_loaders import TextLoader


from langchain.chains import LLMChain
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)

desired_credits = sys.argv[1]

# get api key from environment variable
load_dotenv()
API_KEY = os.getenv("API_KEY")

query = "generate a question for each credit (in the 'credits' array) to ask a stakeholder if they meet the requirements. Return the questions and nothing else."
loader = TextLoader("data.json")
#loader = DirectoryLoader(".", glob="*.txt")
index = VectorstoreIndexCreator().from_loaders([loader])

# print(index.query(query))
questions = index.query(query, llm=ChatOpenAI()) 
numquestions = index.query("how ")
print(questions)


# LLM
llm = ChatOpenAI(api_key=API_KEY, model="gpt-3.5-turbo")

numquestionsString = llm.invoke("here are a series of questions: "+ questions + "how many are there? return the number of questions in digit format and nothing else e.g. 13").content
numquestions = int(numquestionsString)

def initialize(protagonist, antagonist):

    # Prompt
    prompt = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(
                "You are a story generating RPG that will ask a series of inputs to the user in format:" 
                + 
                "[A: 'first option', B: 'second option', C: 'third option', D: 'fourth option']"
                +
                f"The protagonist is {protagonist} and the antagonist is {antagonist}.\n"
                +
                "There will be 10 mcq questions (each with for options as listed above) asked and each will be a point in the story.\n"
                +
                "The 10 sections are: \n['title', 'Inciting Event', 'First Plot Point', 'First Pinch Point', 'Midpoint', 'Second Pinch Point', 'Third Plot Point', 'Climax', 'Climactic Moment', 'Resolution']"
                +
                "ask the series of questions one by one in format: "
                +
                "[question: 'context', A: 'first option', B: 'second option', C: 'third option', D: 'fourth option']"
                
            ),
            # The `variable_name` here is what must align with memory
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{question}"),
        ]
    )

    # Notice that we `return_messages=True` to fit into the MessagesPlaceholder
    # Notice that `"chat_history"` aligns with the MessagesPlaceholder name
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation = LLMChain(llm=llm, prompt=prompt, verbose=True, memory=memory)