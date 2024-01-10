import streamlit as st
import pandas as pd
import os
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, create_tagging_chain, create_tagging_chain_pydantic
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.callbacks import get_openai_callback
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer
from sentence_transformers import CrossEncoder
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from sidebar import *
from tagging import *


st.set_page_config(page_title="Summarize and Tagging MA Bills", layout='wide')
st.title('Summarize Bills')

sbar()


template = """"
Can you please explain what the following MA bill means to a regular citizen without specialized knowledge? 

Please provide a one paragraph summary in 4 sentences. Please be direct and concise for the busy reader.

Note that the bill refers to specific existing sections of the Mass General Laws, so take into account what you know about the pre-existing language and meaning of those sections.

Summarize the bill that reads as follows:\n{context}\n\n

If this information exists, use the Massachusetts General Laws:\n{laws}\n
"""

# model to test hallucination
model = CrossEncoder('vectara/hallucination_evaluation_model')

# load the dataset
df = pd.read_csv("demoapp/12billswithmgl.csv")


def find_bills(bill_number, bill_title):
    """input:
    args: bill_number: (str), Use the number of the bill to find its title and content
    """
    bill = df[df['BillNumber'] == bill_number]['DocumentText']

    try:
         # Locate the index of the bill
        idx = bill.index.tolist()[0]
        # Locate the content and bill title of bill based on idx
        content = df['DocumentText'].iloc[idx]
        #bill_title = df['Title'].iloc[idx]
        bill_number = df['BillNumber'].iloc[idx]
        # laws
        law = df['combined_MGL'].iloc[idx]

        return content, bill_title, bill_number, law
    
    except Exception as e:
        content = "blank"
        st.error("Cannot find such bill from the source")
    

bills_to_select = {
    '#H3121': 'An Act relative to the open meeting law',
    '#S2064': 'An Act extending the public records law to the Governor and the Legislature',
    '#H711': 'An Act providing a local option for ranked choice voting in municipal elections',
    '#S1979': 'An Act establishing a jail and prison construction moratorium',
    '#H489': 'An Act providing affordable and accessible high-quality early education and care to promote child development and well-being and support the economy in the Commonwealth',
    '#S2014': 'An Act relative to collective bargaining rights for legislative employees',
    '#S301': 'An Act providing affordable and accessible high quality early education and care to promote child development and well-being and support the economy in the Commonwealth',
    '#H3069': 'An Act relative to collective bargaining rights for legislative employees',
    '#S433': 'An Act providing a local option for ranked choice voting in municipal elections',
    '#H400': 'An Act relative to vehicle recalls',
    '#H538': 'An Act to Improve access, opportunity, and capacity in Massachusetts vocational-technical education',
    '#S257': 'An Act to end discriminatory outcomes in vocational school admissions'
}

# Displaying the selectbox
selectbox_options = [f"{number}: {title}" for number, title in bills_to_select.items()]
option = st.selectbox(
    'Select a Bill',
    selectbox_options
)

# Extracting the bill number from the selected option
selected_num = option.split(":")[0][1:]
selected_title = option.split(":")[1]

bill_content, bill_title, bill_number, masslaw = find_bills(selected_num, selected_title)


def generate_categories(text):
    """
    generate tags and categories
    parameters:
        text: (string)
    """
    try:
        API_KEY = st.session_state["OPENAI_API_KEY"]
    except Exception as e:
         return st.error("Invalid [OpenAI API key](https://beta.openai.com/account/api-keys) or not found")
    
    # LLM
    category_prompt = """According to this list of category {category}.

        classify this bill {context} into a closest relevant category.

        Do not output a category outside from the list
    """

    prompt = PromptTemplate(template=category_prompt, input_variables=["context", "category"])

    
    llm = LLMChain(
            llm = ChatOpenAI(openai_api_key=API_KEY, temperature=0, model='gpt-4'), prompt=prompt)
        
    response = llm.predict(context = text, category = category_for_bill) # grab from tagging.py
    return response


def generate_tags(category, context):
    """Function to generate tags using Retrieval Augmented Generation
    """

    try:
        API_KEY = st.session_state["OPENAI_API_KEY"]
        os.environ['OPENAI_API_KEY'] = API_KEY
    except Exception as e:
         return st.error("Invalid [OpenAI API key](https://beta.openai.com/account/api-keys) or not found")
    
    loader = TextLoader("demoapp/category.txt").load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    documents = text_splitter.split_documents(loader)
    vectorstore = Chroma.from_documents(documents, OpenAIEmbeddings())
    retriever = vectorstore.as_retriever()

    # LLM
    template = """You are a trustworthy assistant for question-answering tasks.
    Use the following pieces of retrieved context to answer the question.
    Question: {question}
    Context: {context}
    Answer:
    
    """

    prompt = PromptTemplate.from_template(template)
    llm = ChatOpenAI(openai_api_key=API_KEY, temperature=0, model='gpt-4')

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    query = f"""Output top 3 tags from the category {category} that is relevant to the context {context}
    """
        
    response = rag_chain.invoke(query)
    return response


def generate_response(text, law_text):
    """Function to generate response"""
    try:
        API_KEY = st.session_state['OPENAI_API_KEY']
    except Exception as e:
        return st.error("Invalid [OpenAI API key](https://beta.openai.com/account/api-keys) or not found")
    
    prompt = PromptTemplate(input_variables=["context", "laws"], template=template)

    # Instantiate LLM model
    with get_openai_callback() as cb:
        llm = LLMChain(
            llm = ChatOpenAI(openai_api_key=API_KEY,
                     temperature=0.01, model="gpt-3.5-turbo-1106"), prompt=prompt)
        
        response = llm.predict(context=text, laws=law_text)
        return response, cb.total_tokens, cb.prompt_tokens, cb.completion_tokens, cb.total_cost
    

# Function to update or append to CSV
def update_csv(bill_num, title, summarized_bill, category, tag, csv_file_path):
    try:
        df = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        # If the file does not exist, create a new DataFrame
        df = pd.DataFrame(columns=["Bill Number", "Bill Title", "Summarized Bill", "Category", "Tags"])
    
    mask = df["Bill Number"] == bill_num
    if mask.any():
        df.loc[mask, "Bill Title"] = title
        df.loc[mask, "Summarized Bill"] = summarized_bill
        df.loc[mask, "Category"] = category
        df.loc[mask, "Tags"] = tag
    else:
        new_bill = pd.DataFrame([[bill_num, title, summarized_bill, category, tag]], columns=["Bill Number", "Bill Title", "Summarized Bill", "Category", "Tags"])
        df = pd.concat([df, new_bill], ignore_index=True)
    
    df.to_csv(csv_file_path, index=False)
    return df

csv_file_path = "demoapp/generated_bills.csv"


answer_container = st.container()
with answer_container:
    submit_button = st.button(label='Summarize')
    col1, col2, col3 = st.columns([1.5, 1.5, 1])

    if submit_button:
        with st.spinner("Working hard..."):
            
                response, response_tokens, prompt_tokens, completion_tokens, response_cost = generate_response(bill_content, masslaw)
                category_response = generate_categories(bill_content)
                tag_response = generate_tags(category_response, bill_content)
                
                with col1:
                    st.subheader(f"Original Bill: #{bill_number}")
                    st.write(bill_title)
                    st.write(bill_content)

                with col2:
                    st.subheader("Generated Text")
                    st.write(response)
                    st.write("###")
                    st.write("Category:", category_response)
                    st.write(tag_response)
                    
                    update_csv(bill_number, bill_title, response, category_response, tag_response, csv_file_path)
                    st.download_button(
                            label="Download Text",
                            data=pd.read_csv("demoapp/generated_bills.csv").to_csv(index=False).encode('utf-8'),
                            file_name='Bills_Summarization.csv',
                            mime='text/csv',)
                    
                with col3:
                    st.subheader("Evaluation Metrics")
                    # rouge score addition
                    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
                    rouge_scores = scorer.score(bill_content, response)
                    st.write(f"ROUGE-1 Score: {rouge_scores['rouge1'].fmeasure:.2f}")
                    st.write(f"ROUGE-2 Score: {rouge_scores['rouge2'].fmeasure:.2f}")
                    st.write(f"ROUGE-L Score: {rouge_scores['rougeL'].fmeasure:.2f}")
                    
                    # calc cosine similarity
                    vectorizer = TfidfVectorizer()
                    tfidf_matrix = vectorizer.fit_transform([bill_content, response])
                    cosine_sim = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])
                    st.write(f"Cosine Similarity Score: {cosine_sim[0][0]:.2f}")

                    # test hallucination
                    scores = model.predict([
                        [bill_content, response]
                    ])
                    score_result = float(scores[0])
                    st.write(f"Factual Consistency Score: {round(score_result, 2)}")
                    
                    st.write("###")
                    st.subheader("Token Usage")
                    st.write(f"Response Tokens: {response_tokens}")
                    st.write(f"Prompt Response: {prompt_tokens}")
                    st.write(f"Response Complete:{completion_tokens}")
                    st.write(f"Response Cost: $ {response_cost}")
                    
                    