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


template = """"You are a summarizer model that summarizes legal bills and legislation. Please include the bill's main purpose, relevant key points and any amendements. 
The summaries must be easy to understand and accurate based on the provided bill. I want you to summarize the legal bill and legislation. 
Use the title {title} to guide your summary. Summarize the bill that reads as follows:\n{context}\n\nSummary: An Act [bill title]. This bill [key information].
"""

# model to test hallucination
model = CrossEncoder('vectara/hallucination_evaluation_model')

# load the dataset
df = pd.read_csv("demoapp/all_bills.csv")

# Creating search bar 
search_number = st.text_input("Search by Bill Number")
search_title = st.text_input("Search by Bill Title")

# Initial empty DataFrame
filtered_df = df

# Filtering based on inputs
if search_number:
    filtered_df = df[df['BillNumber'].str.contains(search_number, case=False, na=False)]
if search_title:
    filtered_df = df[df['Title'].str.contains(search_title, case=False, na=False)]

if not filtered_df.empty:
    # Creating selectbox options safely
    selectbox_options = [f"Bill #{num}: {filtered_df[filtered_df['BillNumber'] == num]['Title'].iloc[0]}" 
                         for num in filtered_df['BillNumber'] if not filtered_df[filtered_df['BillNumber'] == num].empty]

    option = st.selectbox(
        'Select a Bill',
        selectbox_options
    )

    # Extracting the bill number, title, and content from the selected option
    bill_number = option.split(":")[0][6:]
    bill_title = option.split(":")[1]
    bill_content = filtered_df[filtered_df['BillNumber'] == bill_number]['DocumentText'].iloc[0]
    
else:
    if search_number or search_title:
        st.write("No bills found matching the search criteria.")


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


def generate_response(text, title):
    """Function to generate response"""
    try:
        API_KEY = st.session_state['OPENAI_API_KEY']
    except Exception as e:
        return st.error("Invalid [OpenAI API key](https://beta.openai.com/account/api-keys) or not found")
    
    prompt = PromptTemplate(input_variables=["context", "title"], template=template)

    # Instantiate LLM model
    with get_openai_callback() as cb:
        llm = LLMChain(
            llm = ChatOpenAI(openai_api_key=API_KEY,
                     temperature=0.01, model="gpt-3.5-turbo-1106"), prompt=prompt)
        
        response = llm.predict(context=text, title=title)
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
    # col1, col2, col3 = st.columns(3, gap='medium')
    col1, col2, col3 = st.columns([1.5, 1.5, 1])

    if submit_button:
        with st.spinner("Working hard..."):
            
                response, response_tokens, prompt_tokens, completion_tokens, response_cost = generate_response(bill_content, bill_title)
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
                    
                    
