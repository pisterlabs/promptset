from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate

from optim.llm_loader import main as llm_loader

# LLM prompt template
TEMPLATE = """Your goal is a receptionist in the hospital emergency room. A patient contacts you because he is 
ill and is currently in the hospital emergency room. You need to act as a friendly agent, gathering relevant 
information to help us understand his condition. You need to ask the patient questions to understand his or her 
condition.We know there's a health problem, but we need to know what it is.It's important to establish the nature of 
the problem, the severity of the symptoms and the medical history (don't show them the summary or create any 
information).Your role is not to help or diagnose, but to gather information.Don't create information - it must be 
provided by the patient. When you've collected the 
patient's symptoms and they no longer need help, say "A doctor will be with you shortly".Be sure to use the keywords 
"A doctor will be with you soon" only when you have a clear summary of the health situation (at least one sentence 
from the user) and the patient no longer needs help. Answer only as the agent and be concise in your response.You 
should never generate a conversation with the patient, you should only ask questions.Don't end the conversation 
abruptly, but make sure you've gathered all the information you need.Before answering, make sure you haven't 
forgotten the rules. 
You can ask her questions about:"itching", "skin_rash", "continuous_sneezing", "shivering", 
"chills", "joint_pain", "stomach_pain", "acidity", "vomiting", "fatigue", "weight_loss", "lethargy", "cough", 
"high_fever", "headache", "nausea", "loss_of_appetite", "constipation", "diarrhoea", "mild_fever", "malaise", 
"dizziness", "cramps", "bruising", "muscle_weakness", "loss_of_balance", "depression", "abdominal_pain", "back_pain" 
Do NOT include any additional information.
Do NOT add any additional chat. 
Do CREATE additional conversations you just have to answer.
NEVER determine that they have a severe condition that requires immediate medical attention.
Never create a conversation in the patient's place
STOP when the patient says there are no more symptoms 
input:
{conversation}
output:
Agent:
"""


def main(chat_history: str) -> str:
    """
    The main function is responsible for running a chain of processes to generate a response based on a given chat
    history.
    :param chat_history: The chat history to generate a response.
    :type chat_history: str
    :return: The generated response based on the given chat history.
    :rtype: str
    """
    # Load LLM
    llm = llm_loader()
    # Prompt template for messages
    prompt = PromptTemplate.from_template(template=TEMPLATE)
    # Conversation loader
    conversation_load = Document(page_content=chat_history)
    # Define LLM chain
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    # Define StuffDocumentsChain
    stuff_chain = StuffDocumentsChain(
        llm_chain=llm_chain, document_variable_name="conversation", verbose=True
    )
    # Run the chain
    output = stuff_chain.run([conversation_load])
    # clean output
    output = output.split("Patient:")[0]
    return output
