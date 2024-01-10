from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.summarize import load_summarize_chain
from langchain.schema.document import Document
import streamlit as st
import os
import json


def get_llm(OPENAI_MODEL=None, max_tokens=1000):
    if not OPENAI_MODEL:
        OPENAI_MODEL = os.environ.get("OPENAI_MODEL")
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    llm = ChatOpenAI(
        temperature=0,
        model_name=OPENAI_MODEL,
        openai_api_key=OPENAI_API_KEY,
        max_tokens=max_tokens,
    )
    return llm


@st.cache_data(ttl=60 * 60 * 12, show_spinner=False)  # Cache data for 12 hours
def get_openAPI_response(text, task, OPENAI_MODEL=None, max_tokens=1000, llm=None):
    messages = [HumanMessage(content=text)]
    llm = get_llm(OPENAI_MODEL=OPENAI_MODEL, max_tokens=max_tokens)
    response = llm.invoke(messages, config={"run_name": task})
    response = str(response.content)
    return response


@st.cache_data(ttl=60 * 60 * 12, show_spinner=False)  # Cache data for 12 hours
def summarize_bill(text, task="Summarize", chain_type="stuff"):
    docs = [Document(page_content=text)]
    llm = get_llm()
    prompt_template = """## Summarise the given document:
## Doucment Text:
{text}

## Document Summary:"""
    prompt = PromptTemplate.from_template(prompt_template)
    chain = load_summarize_chain(llm, prompt=prompt, chain_type=chain_type)
    result = chain.invoke(docs, config={"run_name": task})
    return result["output_text"]


@st.cache_data(ttl=60 * 60 * 12, show_spinner=False)  # Cache data for 12 hours
def get_recommended_question(document_summary, key_values=[]):
    key_values = format_dict_as_string(key_values)
    prompt = PromptTemplate(
        input_variables=["document_summary", "key_values"],
        template="""## You are given a document summary and some of the key value data from the document. 
## Document Summary: 
{document_summary} 

## Document Data: 
{key_values}

## Generate a list of 10 recommended questions on the given document. Format it as markdown text.
""",
    )
    llm = get_llm()
    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.invoke(
        input={
            "document_summary": document_summary,
            "key_values": key_values,
        },
        config={"run_name": "RecommenedQBill"},
    )
    response = response["text"]
    return response


def get_bill_metadata(document_content, key_values=[]):
    # Assuming format_dict_as_string is a function that formats key_values into a string
    key_values_str = format_dict_as_string(key_values)
    prompt = PromptTemplate(
        input_variables=["document_content", "key_values"],
        template="""## You are given a document summary and some of the key value data from the document. 
## Document Content: 
{document_content} 

## Document Data: 
{key_values}

## Based on the document summary and data provided, please output the following in JSON format:
- recommended_questions: A list of 10 recommended questions about the document.
- keywords: A list of keywords or key phrases that represent the main topics or themes of the document.
- named_entities: A list of named entities from the document, with each named entity being a list of the extracted name and its type (PERSON,Organization,Location,Date,Time,etc) (e.g., [["Steve Jobs", "PERSON"], ["India", "COUNTRY"]]).
- document_type: Identify and categorize the type of document (e.g., "bill", "invoice", "ID card", "invoice", "contract", "report", "legal document" etc.).
- language: Determine the language in which the document is written
- currency: Determine the currency used in the document. Output the currency code, such as USD, EUR, GBP, INR etc. Consider variations in formatting and placement of currency symbols, as well as potential textual references to currencies. 

## JSON Output:
""",
    )
    # Get the language model response
    llm = get_llm(max_tokens=2000)
    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.invoke(
        input={
            "document_content": document_content,
            "key_values": key_values_str,
        },
        config={"run_name": "BillMetadata"},
    )

    # Parse the response, assuming the model returns a properly formatted JSON string
    response_data = json.loads(response["text"])

    # Output the combined dictionary
    return response_data


@st.cache_data(ttl=60 * 60 * 12, show_spinner=False)  # Cache data for 12 hours
def get_billbot_response(question, document_summary, key_values=[]):
    key_values = format_dict_as_string(key_values)
    prompt = PromptTemplate(
        input_variables=["question", "document_summary", "key_values"],
        template="""## You are given a document summary and some of the key value data from the document. 
## Document Summary:
{document_summary} 

## Document Data:
{key_values}

## You have to answer the given user question stricty from the document. 
Do not assume any values or answer on your own. 
If you don't know the answer respond with "I don't know the answer"

## Question: {question}
""",
    )
    llm = get_llm()
    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.invoke(
        input={
            "question": question,
            "document_summary": document_summary,
            "key_values": key_values,
        },
        config={"run_name": "BillBot"},
    )
    response = response["text"]
    return response

@st.cache_data(ttl=60 * 60 * 12, show_spinner=False)  # Cache data for 12 hours
def get_credit_card_output(bill_summary, key_value_pairs ):
    key_value_pairs = format_dict_as_string(key_value_pairs)
    output_structure = {"CreditCard":{"BankName":"","CreditCardType":"","NameOnCard":"","CardNo":"","CreditLimit":"","Statement":{"StatementDate":"","PaymentDueDate":"","TotalAmountDue":"","MinimumAmountDue":"","FinanceCharges":"",}}}
    # "Transactions":[{"TransactionDate":"","TransactionDescription":"","TransactionType":"","TransactionAmount":"","TransactionCategory":{"TransactionHead":"","TransactionSubHead":"","Payee":""}}]
    prompt = PromptTemplate(
        input_variables=["bill_summary", "key_value_pairs", "output_structure"],
        template='''## Summary:
{bill_summary}

## Data extracted from Credit Card Statement as key values: 
{key_value_pairs}

## Output_JSON_Format:
{output_structure}

## Task: Structure the data into the provided Output_JSON_Format. 

## Output JSON:
''',
    )
    llm = get_llm(max_tokens=500)
    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.invoke(
        input={
            "bill_summary": bill_summary,
            "key_value_pairs": key_value_pairs,
            "output_structure": output_structure,
        },
        config={"run_name": "BillBot"},
    )
    response = response["text"]
    return response

def format_dict_as_string(input_dict):
    formatted_string = ""
    for key, value in input_dict.items():
        formatted_string += f"{key} : {value}\n"
    return formatted_string.strip()


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    import warnings
    from utils import get_or_generate_analyze_json,convert_keyvalues_to_dict,extract_text_from_pdf

    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        module="streamlit.runtime.caching.cache_data_api",
    )
    load_dotenv("./app/.env")

    pdf_path = "./app/data/LT E-BillPDF_231119_145147.pdf"
    json_data = get_or_generate_analyze_json(pdf_path)
    document_content = extract_text_from_pdf(json_data)
    key_values = convert_keyvalues_to_dict(json_data)
    get_bill_metadata(document_content, key_values)
