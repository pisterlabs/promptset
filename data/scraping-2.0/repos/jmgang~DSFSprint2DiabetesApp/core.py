import os
from typing import Any, Dict, List

from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain, ConversationChain
from langchain.prompts.prompt import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

from dotenv import load_dotenv, find_dotenv

from ExtendedConversationBufferMemory import ExtendedConversationMemory

_ = load_dotenv(find_dotenv()) # read local .env file


def chat_with_specialist(query: str, patient_information: List[str], collected_patient_information: List[str]):
    llm = ChatOpenAI(
        temperature=0,
        openai_api_key=os.environ["OPENAI_API_KEY"],
        model='gpt-4'
    )

    template = """The following is a friendly conversation between a human patient and an AI diabetes predictor specialist. The AI is caring towards a patient. If the AI does not know the answer to a question, it truthfully says it does not know.
    The specialist would continue to ask about the patient's information: {patient_information}. The AI specialist would also store the information 
    given by the patient as collected_patient_information. Some of the information asked are questions only answerable by yes or no. The 
    collected information so far are {collected_information}. Do not ask anymore if it has been answered in the collected information. 
    Don't stop asking until all the information have been collected.
    
    Current conversation:
    {history}
    Human: {input}
    AI Specialist: 
    {format_instructions}
    """

    patient_information_schema = ResponseSchema(name="remaining_patient_information",
                                                description="This is the remaining patient information that is needed to be asked.")

    collected_patient_information_schema = ResponseSchema(name="collected_patient_information", type="JSON object (set question as key)",
                                                          description="The collected patient information. This includes information computed by AI that is needed. (i.e. BMI from Height/Weight)")

    doctor_response_schema = ResponseSchema(name="AI response", description="AI Specialist current response")

    can_stop_asking_schema = ResponseSchema(name='can_stop_asking', type="Boolean", description="Whether the AI can stop asking or not.")

    response_schemas = [patient_information_schema, collected_patient_information_schema, doctor_response_schema, can_stop_asking_schema]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()

    prompt = PromptTemplate(
        input_variables=["history", "input", "patient_information", "collected_information"],
        template=template
        , partial_variables={"format_instructions": format_instructions})

    conversation = ConversationChain(llm=llm, prompt=prompt, verbose=True,
                                     memory=ExtendedConversationMemory(llm=llm, ai_prefix="AI Specialist", k=2,
                                                                              extra_variables=["patient_information", "collected_information"])
                                     )

    convo = conversation({'input': query,
                         'patient_information': patient_information,
                          'collected_information': collected_patient_information}
                        )
    response_as_dict = output_parser.parse(convo['response'])

    return convo, response_as_dict

def run_llm(query: str, input_variables: List[str], input_keys, llm_model="gpt-4"):

    llm = ChatOpenAI(temperature=0, model=llm_model, openai_api_key=os.environ['OPENAI_API_KEY'])

    prompt_template = query
    prompt = PromptTemplate(
        input_variables=input_variables, template=prompt_template
    )

    chain = LLMChain(llm=llm, prompt=prompt, verbose=True, )

    return chain.run(**input_keys)
