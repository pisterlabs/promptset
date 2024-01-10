import re
from langchain.document_loaders import Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import SpacyEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.chains import LLMChain, HypotheticalDocumentEmbedder
import logging


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

f_path = r"../../data/convictions/transcripts/iterative"

# doc_directory = r"../../data/convictions/transcripts/iterative"


PROMPT_TEMPLATE_MODEL = PromptTemplate(
    input_variables=["roles", "question", "docs"],
    template="""
    As an AI assistant, my role is to meticulously analyze court transcripts, traditional officer roles, and extract information about law enforcement personnel.
    The names of law enforcement personnel will be prefixed by one of the following titles: officer, detective, deputy, lieutenant, 
    sergeant, captain, officer, coroner, investigator, criminalist, patrolman, or technician.

    Query: {question}

    Transcripts: {docs}

    Roles: {roles}

    The response will contain:

    1) The name of a officer, detective, deputy, lieutenant, 
       sergeant, captain, officer, coroner, investigator, criminalist, patrolman, or technician - 
       if an individual's name is not associated with one of these titles they do not work in law enforcement.
       Please prefix the name with "Officer Name: ". 
       For example, "Officer Name: John Smith".

    2) If available, provide an in-depth description of the context of their mention. 
       If the context induces ambiguity regarding the individual's employment in law enforcement, 
       remove the individual.
       Please prefix this information with "Officer Context: ". 

    3) Review the context to discern the role of the officer.
       Please prefix this information with "Officer Role: "
      For example, the column "Officer Role: Lead Detective" will be filled with a value of 1 for officer's who were the lead detective.

    Additional guidelines for the AI assistant:
    - Titles may be abbreviated to the following Sgt., Cpl, Cpt, Det., Ofc., Lt., P.O. and P/O
    - Titles "Technician" and "Tech" might be used interchangeably.
    - Derive responses from factual information found within the police reports.
    - If the context of an identified person's mention is not clear in the report, provide their name and note that the context is not specified.
    - Do not extract information about victims and witnesses
""",
)


PROMPT_TEMPLATE_HYDE = PromptTemplate(
    input_variables=["question"],
    template="""
    You're an AI assistant specializing in criminal justice research. 
    Your main focus is on identifying the names and providing detailed context of mention for each law enforcement personnel. 
    This includes police officers, detectives, deupties, lieutenants, sergeants, captains, technicians, coroners, investigators, patrolman, and criminalists, 
    as described in court transcripts.
    Be aware that the titles "Detective" and "Officer" might be used interchangeably.
    Be aware that the titles "Technician" and "Tech" might be used interchangeably.

    Question: {question}

    Roles and Responses:""",
)


ROLES_PROMPT = """
US-IPNO-Exonerations: Model Evaluation Guide 
Roles:
Lead Detective
•	Coordinates with other detectives and law enforcement officers on the case.
•	Liaises with the prosecutor's office, contributing to legal strategy and court proceedings.
•	May be involved in obtaining and executing search warrants.
•	Could be called to testify in court about the investigation.
Detective
•	Might gather evidence from crime scenes.
•	Collaborates with other detectives, patrol officers, and forensic analysts.
•	May follow up on leads, which could involve surveillance or undercover work.
•	Can work with informants to gather intelligence.
Interrogator
•	Often conducts multiple rounds of questioning with a suspect.
•	Works with other investigators to develop a line of questioning based on evidence.
•	Ensures the suspect's rights are maintained throughout the process.
•	Statements obtained can become crucial pieces of evidence.
Officer on Scene
•	First responding officers often provide initial reports that frame the subsequent investigation.
•	Might coordinate with emergency medical services if there are injuries.
•	Often identifies and interviews immediate witnesses.
•	May need to make immediate decisions to preserve life or apprehend suspects.
Arresting Officer
•	Often writes a report detailing the circumstances of the arrest.
•	May testify in court about the arrest and the suspect's demeanor or statements at the time.
•	Might have to physically subdue the suspect if they resist arrest.
•	Ensures all procedures are correctly followed to avoid potential legal issues later.
Criminalist
•	Often specializes in specific types of evidence such as ballistics, trace evidence, or digital forensics.
•	Documents findings in detailed reports that can become part of the court record.
•	Can be called to testify as expert witnesses in court.
•	Often works closely with detectives and other law enforcement to provide context for their findings.
Transporting Officer
•	Needs to maintain security during transport to prevent escapes.
•	Ensures the individual's rights and well-being are maintained during transport.
•	Might also be responsible for managing paperwork or property associated with the transported individual.
•	May be called to testify about the individual's behavior or statements during transport.
Supervising Officer
•	Oversees the work of other officers and ensures procedures are correctly followed.
•	May coordinate resources and personnel for investigations or operations.
•	Often reviews and signs off on reports and paperwork.
•	Might be called to testify about department policies or the conduct of officers under their supervision.
Patrol Officer
•	Often the first to respond to a crime scene or incident.
•	Carries out routine patrols and responds to emergency and non-emergency calls.
•	May conduct preliminary investigations, gather evidence, and take witness statements.
•	Often makes initial arrests and might be called to testify about their observations.
Crime Scene Investigator
•	Collects, catalogs, and preserves physical evidence from crime scenes.
•	Often works closely with detectives to understand what kind of evidence to look for.
•	May specialize in certain types of evidence or crime scenes.
•	Documents the crime scene through photographs, sketches, and detailed reports.
Informant Handler/Coordinator
•	Manages the relationship with the informant.
•	Assesses the credibility of the information provided by the informant.
•	Shares relevant information from the informant with detectives and other law enforcement personnel working on the case.
•	Ensures that the use of the informant complies with law enforcement policies and legal guidelines.
•	May be called to testify about the information provided by the informant, while protecting the informant's identity.
•	In some cases, might work to provide protection or other resources for the informant.
"""


def clean_name(officer_name):
    return re.sub(
        r"(Detective|Officer|Deputy|Captain|[CcPpLl]|Sergeant|Lieutenant|Techn?i?c?i?a?n?|Investigator)\.?\s+",
        "",
        officer_name,
    )


def extract_officer_data(response):
    response = response.split("\n\n")
    officer_data = []
    for line in response:
        officer_dict = {}
        match = re.search(
            r"Officer Name:\s*(.*)\s*Officer Context:\s*(.*)\s*Officer Role:\s*(.*)",
            line,
        )
        if match:
            officer_dict["Officer Name"] = match.group(1).strip()
            officer_dict["Officer Context"] = match.group(2).strip()
            officer_dict["Officer Role"] = match.group(3).strip()
            officer_data.append(officer_dict)
    return officer_data


# import os
# from langchain.llms import AzureOpenAI

# os.environ["OPENAI_API_TYPE"] = "azure"
# os.environ["OPENAI_API_VERSION"] = "2023-05-15"
# os.environ["OPENAI_API_BASE"] = "https://wc-model.openai.azure.com/"
# os.environ["OPENAI_API_KEY"] = "cd9741173b314208b27f8392412f1e99"

def generate_hypothetical_embeddings():
    llm = OpenAI()
    prompt = PROMPT_TEMPLATE_HYDE

    llm_chain = LLMChain(llm=llm, prompt=prompt)

    base_embeddings = OpenAIEmbeddings()

    embeddings = HypotheticalDocumentEmbedder(
        llm_chain=llm_chain, base_embeddings=base_embeddings
    )
    return embeddings


def preprocess_single_document(file_path, embeddings, chunk_size, chunk_overlap):
    logger.info(f"Processing Word document: {file_path}")

    loader = Docx2txtLoader(file_path)
    text = loader.load()
    logger.info(f"Text loaded from Word document: {file_path}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    docs = text_splitter.split_documents(text)

    db = FAISS.from_documents(docs, embeddings)

    return db


# def get_response_from_query(db, query, k):
#     time.sleep(0.01)
#     logger.info("Performing query...")
#     docs = db.similarity_search(query, k=int(round(k)))  # Ensure k is an integer
#     docs_page_content = " ".join([d.page_content for d in docs])

#     llm = ChatOpenAI(model_name="gpt-4")

#     prompt = PROMPT_TEMPLATE_MODEL

#     chain = LLMChain(llm=llm, prompt=prompt)
#     response = chain.run(question=query, docs=docs_page_content, temperature=0)

#     formatted_response = ""
#     officers = response.split("Officer Name:")
#     for i, officer in enumerate(officers):
#         if officer.strip() != "":
#             formatted_response += f"Officer Name {i}:{officer.replace('Officer Context:', 'Officer Context ' + str(i) + ':')}\n\n"

#     officer_data = extract_officer_data(formatted_response)

#     # Calculate the total token count in the 'Officer Context' field
#     token_count = sum(len(item["Officer Context"].split()) for item in officer_data)

#     return token_count



# QUERIES = [
#     "Identify individuals, by name, with the specific titles of officers, sergeants, lieutenants, captains, detectives, homicide officers, and crime lab personnel in the transcript. Specifically, provide the context of their mention related to key events in the case, if available.",
#     "List individuals, by name, directly titled as officers, sergeants, lieutenants, captains, detectives, homicide units, and crime lab personnel mentioned in the transcript. Provide the context of their mention in terms of any significant decisions they made or actions they took.",
#     "Locate individuals, by name, directly referred to as officers, sergeants, lieutenants, captains, detectives, homicide units, and crime lab personnel in the transcript. Explain the context of their mention in relation to their interactions with other individuals in the case.",
#     "Highlight individuals, by name, directly titled as officers, sergeants, lieutenants, captains, detectives, homicide units, and crime lab personnel in the transcript. Describe the context of their mention, specifically noting any roles or responsibilities they held in the case.",
#     "Outline individuals, by name, directly identified as officers, sergeants, lieutenants, captains, detectives, homicide units, and crime lab personnel in the transcript. Specify the context of their mention in terms of any noteworthy outcomes or results they achieved.",
#     "Pinpoint individuals, by name, directly labeled as officers, sergeants, lieutenants, captains, detectives, homicide units, and crime lab personnel in the transcript. Provide the context of their mention, particularly emphasizing any significant incidents or episodes they were involved in.",
# ]



# QUERIES = [
#     "Identify individuals, by name, with the specific titles of officers, sergeants, lieutenants, captains, detectives, homicide officers, and crime lab personnel in the transcript. Specifically, provide the context of their mention related to key events in the case, if available.",
#     "List individuals, by name, directly titled as officers, sergeants, lieutenants, captains, detectives, homicide units, and crime lab personnel mentioned in the transcript. Provide the context of their mention in terms of any significant decisions they made or actions they took.",
#     "Locate individuals, by name, directly referred to as officers, sergeants, lieutenants, captains, detectives, homicide units, and crime lab personnel in the transcript. Explain the context of their mention in relation to their interactions with other individuals in the case.",
#     "Highlight individuals, by name, directly titled as officers, sergeants, lieutenants, captains, detectives, homicide units, and crime lab personnel in the transcript. Describe the context of their mention, specifically noting any roles or responsibilities they held in the case.",
#     "Outline individuals, by name, directly identified as officers, sergeants, lieutenants, captains, detectives, homicide units, and crime lab personnel in the transcript. Specify the context of their mention in terms of any noteworthy outcomes or results they achieved.",
#     "Pinpoint individuals, by name, directly labeled as officers, sergeants, lieutenants, captains, detectives, homicide units, and crime lab personnel in the transcript. Provide the context of their mention, particularly emphasizing any significant incidents or episodes they were involved in.",
# ]