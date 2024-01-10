from langchain.document_loaders import TextLoader
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
import openai
from openai import OpenAI
import os
import json
from dotenv import load_dotenv

from analyze import *

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
openai.api_key = os.environ["OPENAI_API_KEY"]

#parse the ehr file given and run our ai functionality on it
#We'll take a PDF EHR and parse the data to get a text based summary with the primary medical and dietary restrictions
#to help our later steps focus on the necessary information and streamline the process.
#1. Parse texts
#2. Get summary of EHR
def parse_ehr(filepath):
    loader = None
    loader = PyPDFLoader(filepath)
    docs = loader.load()
    #print('Documents -> Data', documents[0])
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(docs)

    print("\n\n What are the texts? \n\n")
    print(texts)
    print("\n\n")
    embeddings = OpenAIEmbeddings()
    docsearch = Chroma.from_documents(texts, embeddings)

    prompt_template = """Use the following pieces of context to answer the question at the end. 
    {context}

    Question: {question}
    """
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    #chain_type_kwargs = {"prompt": PROMPT}

    qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=docsearch.as_retriever())
    #Add a confidence level field alongside each field you extract representing how confident you are in your result.
    query = '''
    You are given an Electronic Health Record which summarizes a patient's health and their conditions as per doctor diagnoses.
    You are tasked with finding all the conditions and medications the patient has/is on and listing those out comprehensively.

    1. Find all medical conditions diagnosed by the doctor.
    2. Find any medications that the patient is currently on.
    3. Find any allergies to certain drugs or medical restrictions that the patient has.
    4. Find any dietary restrictions that the patient may have.

    Give me a bulleted list/summary of all the medical conditions and the medications that
    the patient is on. Make this comprehensive because we will use it to assess what future
    medications that the patient can take.

    Rules:
    1. Include all the medical conditions and medications specified.
    2. Be comprehensive in the dosage, exceptions and all specifications regarding the conditions & medications.
    3. Don't make up any analysis or extrapolations--simply pull details from the doctor's diagnosis.
    '''
    
    result = qa.run(query)
    return result

#augment with some data
def magic(analysis, ehr, ingredients):
    loader = PyPDFLoader('ref1.pdf')
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    docsearch = Chroma.from_documents(texts, embeddings)
    qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=docsearch.as_retriever(search_kwargs={"k": 1}))
    query = f'''
    This is some medical contextual information. We have a patient who has the following health summary:
    {ehr}

    ...
    The drug they are planning to take is:

    {ingredients}

    ...
    The current analysis is:

    {analysis}

    Parse over the document that is provided to you and see if there are any relevant pieces of information to the above analysis and summarize those.

    '''
    res = qa.run(query)
    return res


def main(res, ingds):
    print('Summary\n')
    print(res)
    print('\n')
    ehr_formatted_summary, analysis = analyze(res, ingredients=ingds)

    print('Analysis \n\n\n')
    print(type(analysis))
    print('\n\n')
    print(analysis.content)

    result = magic(analysis=analysis.content, ehr=ehr_formatted_summary, ingredients=ingds)

    print('Result')
    print(result)
    print('\n\n')

    '''client = OpenAI(
        api_key = os.environ["OPENAI_API_KEY"]
    )
    resp = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[
                {"role": "system", "content": ''You are a smart being that takes context from third party services and improves existing medical analyses. 
                You do not delete information or make up facts but rather summarize and incorporate context into the existing analysis in a smart way.''},
                {"role": "user", "content": f''
                    Here's the medical analysis: {analysis}. 
                    Here's the third party contex which includes some generic information about allergies, conditions and alternative drugs: {result}.

                    IMPORTANT: Make sure to not delete any original analysis.
                    Give me a comprehensive updated analysis.
                 ''}
            ]
    )'''

    combiner = LLMChain(llm=ChatOpenAI(model='gpt-4-1106-preview', temperature=0), prompt=PromptTemplate.from_template(f'''
        You're given a medical analysis. Our goal is to take some additional third party context(a summary of relevant docs) and add it to enhance the
        comprehensiveness of the analysis.
                                                                                                                   
        Analysis:
        {analysis}.
        
        ...

        Third Party Context:

        {result}

        Rules:
        1. Do not remove any information from the original analysis.
        2. Incorporate the context into the original analysis wherever it best fits.

    '''))

    cont = combiner.predict()

    print('Resp\n\n')
    print(cont)
    print('\n')

    manager = LLMChain(llm=ChatOpenAI(model='gpt-4-1106-preview', temperature=0), prompt=PromptTemplate.from_template(
    f'''
    Here's an expert medical analysis generated by state of the art technology.
    Analysis:
    {cont}

    We want to take this formalized medical output and turn it into a humanistic output that a patient can easily parse.

    
    Rule #1: Be kind and empathetic in your response. Show concern and care with human-like emotional statements.
    Rule #2: Use specific details regarding the condition or allergy and how the drug interferes with that. List out potential advantages/disadvantages concisely.

    Procedure:
    You must clearly state if the patient should take the drug or not. 
        -If the drug is safe, clearly state so.
        -If the drug is not safe:
            a) Explain what condition or allergy the drug will amplify or trigger.
            b) Explain why/how or necessary context.
            c) Offer possible alternative drug or therapy if you know of any for sure.
            d) Direct them to consulting their doctor.
        -When you have no confidence regarding your diagnosis--then direct them to their doctor.

    IMPORTANT: Write this output as if you are a nurse talking to a patient directly. Make the output human oriented.
    
    Remember that they're probably taking this drug cause they're feeling
    some other symptoms so focus on drug interferences and if any adverse effects will be caused and if not--clear them to take the drug.
    '''))

    print('\n\nOutput\n')
    humanistic = manager.predict()



    return (ehr_formatted_summary, analysis.content, humanistic)
