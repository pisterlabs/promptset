# Prompt for Chain One: Identify if the email content is suspicious or not.
initial_analysis_prompt = """
You are an AI trained in cybersecurity and phishing detection. Using your training data and understanding, analyze the following email content:
\n\nEmail Content:\n{email_content}
\nProvide a detailed analysis. If it's suspicious, state reasons. If not, provide an assurance. The email is: {analysis_result}.
"""
chain_one_prompt = PromptTemplate(
    input_variables=["email_content"],
    template=initial_analysis_prompt
)
chain_one = LLMChain(llm=llm, prompt=chain_one_prompt)

# Prompt for Chain Two: Provide steps for protection based on the previous analysis.
follow_up_prompt = """
Building upon our previous analysis: "The email is: {analysis_result}." Provide a series of steps the user should follow. If the statement suggests the email was suspicious, provide protection measures. If the statement suggests the email was not suspicious, provide general email safety tips.
"""
chain_two_prompt = PromptTemplate(
    input_variables=["analysis_result"],
    template=follow_up_prompt
)
chain_two = LLMChain(llm=llm, prompt=chain_two_prompt)

# Sequential Chaining
overall_chain = SequentialChain(
    input_variables=["email_content"], 
    chains=[chain_one, chain_two], 
    verbose=True
)

# Run the overall chain with a sample email content
email_sample = """Dear user, We noticed unusual activity on your account. Please click this link to verify your identity."""
result = overall_chain.run(email_sample)
print(result)

import os
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains import SequentialChain
from langchain.memory import SimpleMemory

# Set API keys
os.environ["OPENAI_API_KEY"] = "..."

# Define the LLM instance
llm = OpenAI(temperature=0.7)  # Adjust temperature as needed

# Prompt for Chain One: Identify if the email content is suspicious or not.
initial_analysis_prompt = """You are an AI trained in cybersecurity and phishing detection. Using your training data and understanding, analyze the following email content and determine if it's suspicious or not:
\n\nEmail Content:\n{email_content}
\nProvide a detailed analysis. If it's suspicious, state reasons and if not, provide an assurance."""
chain_one_prompt = PromptTemplate(
    input_variables=["email_content"],
    template=initial_analysis_prompt
)
chain_one = LLMChain(llm=llm, prompt=chain_one_prompt)

# Prompt for Chain Two: Provide steps for protection based on the previous analysis.
follow_up_prompt = """Building upon our previous analysis where we determined the email's nature as {analysis_result}, provide a series of steps the user should follow. If the email was suspicious, provide protection measures. If the email was not suspicious, provide general email safety tips."""
chain_two_prompt = PromptTemplate(
    input_variables=["analysis_result"],
    template=follow_up_prompt
)
chain_two = LLMChain(llm=llm, prompt=chain_two_prompt)

# Sequential Chaining
overall_chain = SequentialChain(
    input_variables=["email_content"], 
    chains=[chain_one, chain_two], 
    verbose=True
)

# Run the overall chain with a sample email content
email_sample = """Dear user, We noticed unusual activity on your account. Please click this link to verify your identity."""
result = overall_chain.run(email_sample)
print(result)


import os
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains import SequentialChain
from langchain.memory import SimpleMemory

# Assuming you've already set your API keys as per the earlier section
os.environ["OPENAI_API_KEY"] = "..."

# Create a ChatOpenAI model instance
chatopenai = ChatOpenAI(model_name="gpt-3.5-turbo")

# Chain One - Initial Email Analysis
initial_analysis_prompt = PromptTemplate(
    input_variables=["email_content", "subject", "red_flag", "attachments", "networkSenderIdentifier"],
    template="""
    Email Analysis Task:
    We are trying to classify emails as either legitimate or suspicious based on various factors like content, subject, red flags, attachments, and sender network identifiers.

    Please analyze the following email details:
    Subject: {subject}
    Content: {email_content}
    Attachments: {attachments}
    Sender: {networkSenderIdentifier}
    Red Flag: {red_flag}

    Provide an initial assessment based on the information provided:
    """
)
chain_one = LLMChain(llm=chatopenai, prompt=initial_analysis_prompt)

# Chain Two - Final Decision
final_decision_prompt = PromptTemplate(
    input_variables=["initial_analysis"],
    template="""
    Based on the initial analysis: {initial_analysis}
    Considering common patterns and signs, decide:
    Is this email legitimate or suspicious?
    """
)
chain_two = LLMChain(llm=chatopenai, prompt=final_decision_prompt)

# Sequential Chain Execution
sequential_email_chain = SequentialChain(
    input_variables=["email_content", "subject", "red_flag", "attachments", "networkSenderIdentifier"],
    chains=[chain_one, chain_two],
    verbose=True
)

# Example execution
result = sequential_email_chain.run(
    email_content="Congratulations! You've won a million dollars! Click here...",
    subject="You're a winner!",
    red_flag="No previous contact with sender",
    attachments="None",
    networkSenderIdentifier="unknown1234@example.com"
)

print(result)
