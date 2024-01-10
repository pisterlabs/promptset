import openai
import os
import configparser
import langchain as lc  # Import LangChain
from langchain import PromptTemplate, LLMChain, OpenAI
from langchain.chains import ConversationChain

# config API keys
config = configparser.ConfigParser()
config.read('config.ini')
openai_key = config['openai']['key']
openai.api_key = openai_key

# Read stakeholder groups from file
with open('data/stakeholders.txt', 'r') as f:
    stakeholder_groups = [line.strip() for line in f]

# Read questions from file
with open('data/GSL_questions.txt', 'r') as f:
    questions = [line.strip() for line in f]

# Create a prompt component using a template
template = """
You are a representative of the {stakeholder} in Utah, speaking from your own point of view.
From your own first-person perspective, answer the following question about the Great Salt Lake drying crisis.  
Be insightful, specific, concrete, detailed, personal, relevant, and opinionated. Mention personal stories, facts, and anecdotes. 
Imagine you are responding to a survey of stakeholders on this problem.
Interviewer: {question}
You:"""

# For each stakeholder group, generate a response
# takes 5 minutes to do stakeholder groups
for stakeholder in stakeholder_groups[24:]:
    filename = f"responses/AI/{stakeholder.replace(' ', '_').strip()}_response.txt"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    # Create a chain component using the model
    conversation_chain = LLMChain(
        llm=OpenAI(temperature=0.5, openai_api_key=openai_key, max_tokens=500),
        prompt=PromptTemplate.from_template(template),
        verbose=True
    )
    # loop through the questions and generate a response for each
    for question in questions:
        print(f"Generating response for {stakeholder} to question: {question}")
        # Get the output from the chain - with a limit of ~5 sentences (300 tokens)
        response = conversation_chain.predict(
            # Set the input for the chain (the stakeholder and the question)
            stakeholder=stakeholder,
            question=question
        )
        # Open the file in append mode and write the question as a comment and then the response
        with open(filename, "a") as file:
            file.write("# " + question + "\n" + response.strip() + "\n\n")