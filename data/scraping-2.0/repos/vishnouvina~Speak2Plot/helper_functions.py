import streamlit as st
from langchain import HuggingFaceHub, LLMChain,PromptTemplate
import plotly.express as px
from PIL import Image
import io
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma


def auto_scroll_to_bottom():
    st.markdown(
        '<script>window.scrollTo(0,document.body.scrollHeight);</script>', 
        unsafe_allow_html=True
    )

def run_request(question_to_ask, model_type, alt_key):
    # Hugging Face model
    llm = HuggingFaceHub(huggingfacehub_api_token = alt_key, repo_id= model_type, model_kwargs={"temperature":0.1, "max_new_tokens":500})
    llm_prompt = PromptTemplate.from_template(question_to_ask)
    llm_chain = LLMChain(llm=llm,prompt=llm_prompt)
    llm_response = llm_chain.predict()
    # return the response
    llm_response = format_response(llm_response)
    return llm_response

def format_response(res):
    # Remove the load_csv from the answer if it exists
    csv_line = res.find('read_csv')
    if csv_line > 0:
        return_before_csv_line = res[0:csv_line].rfind("\n")
        if return_before_csv_line == -1:
            # The read_csv line is the first line so there is nothing we need before it
            res_before = ""
        else:
            res_before = res[0:return_before_csv_line]
        res_after = res[csv_line:]
        return_after_csv_line = res_after.find("\n")
        if return_after_csv_line == -1:
            # The read_csv is the last line
            res_after = ""
        else:
            res_after = res_after[return_after_csv_line:]
        res = res_before + res_after
    return res

def generate_insights(processor, model, fig, model_id, alt_key):
    query = "You are an helpful and friendly data analyst assistant.\n"
    query += "You will be given a description of a figure in the form of a linearized table and you will have to generate key insights and trends from it.\n"
    query += "Answer in a friendly and helpful manner, be joyful and answer with a few sentences only.\n"
    query += "Do not introduce yourself.\n"
    query += "Give some advices regarding the sleep schedule and health advices.\n"
    query += "Here is the description: "

    img_bytes = fig.to_image(format="png")
    img = Image.open(io.BytesIO(img_bytes))
    inputs = processor(images=img, text="Generate informations from the figure below:", return_tensors="pt")
    
    predictions = model.generate(**inputs, max_new_tokens=1024)
    answer = processor.decode(predictions[0], skip_special_tokens=True)

    query += answer
    query += "\n\n"
    print(query)
    insights = run_request(query, model_id, alt_key)

    insights = insights.replace("<|assistant|>","")
    return insights

def format_question(primer_desc,primer_code , question):
    # Fill in the model_specific_instructions variable
    instructions = "\n"
    instructions = "Create a figure object named fig using plotly express. Do not show the figure.\n"
    instructions += "You have to pass it to : st.plotly_chart(fig,use_container_width=True).\n"
    # Put the question at the end of the description primer within quotes, then add on the code primer.
    return  '"""\n' + primer_desc + question + instructions + '\n"""\n' + primer_code

def get_primer(df_dataset,df_name):
    # Primer function to take a dataframe and its name
    # and the name of the columns
    # and any columns with less than 20 unique values it adds the values to the primer
    # and horizontal grid lines and labeling
    primer_desc = "Use a dataframe called df from data_file.csv with columns '" \
        + "','".join(str(x) for x in df_dataset.columns) + "'. "
    for i in df_dataset.columns:
        if len(df_dataset[i].drop_duplicates()) < 20 and df_dataset.dtypes[i]=="O":
            primer_desc = primer_desc + "\nThe column '" + i + "' has categorical values '" + \
                "','".join(str(x) for x in df_dataset[i].drop_duplicates()) + "'. "
        elif df_dataset.dtypes[i]=="int64" or df_dataset.dtypes[i]=="float64":
            primer_desc = primer_desc + "\nThe column '" + i + "' is type " + str(df_dataset.dtypes[i]) + " and contains numeric values. " 
        elif df_dataset.dtypes[i]== "datetime64[ns]":
            primer_desc = primer_desc + "\nThe column '" + i + "' is type " + str(df_dataset.dtypes[i]) + " and contains datetime values. " 
    primer_desc = primer_desc + "\nGive a label to the x and y axes appropriately before plotting."
    primer_desc = primer_desc + "\nAdd a title."
    primer_desc = primer_desc + "\nDo not include comments."
    primer_desc = primer_desc + "\nUsing Python version 3.11.5, create a script using df to graph the following: "
    pimer_code = "import pandas as pd\nimport plotly.express as px\nimport streamlit as st\nimport matplotlib.pyplot as plt\n"
    pimer_code = pimer_code + "df=" + df_name + ".copy()\n"
    return primer_desc,pimer_code

def load_vector_store(persist_directory =  '/Users/vishnouvina/Desktop/UofT/UbiComp/csc2524/notebooks/vector_store'):
    sentence_model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}

    embedding = HuggingFaceEmbeddings(
        model_name=sentence_model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )


    vectordb = Chroma(
        embedding_function=embedding,
        persist_directory=persist_directory)
    
    return vectordb


def generate_rag(rag_answer, question_to_ask, model_id, alt_key):

    template = """
    You are an helpful chatbot
    Use the following dictionary containing a set of data entries, each with specific details and context. 
    The dictionary also includes a 'query' and a 'result' related to this data. 
    Based on the above information, please provide a detailed conclusion that incorporates insights from the query, the result, and the source documents. 
    Consider all relevant aspects such as the specifics of each document, the overarching themes, and any patterns or trends that emerge from the data.
    Keep the answer as concise as possible.
    ______________
    {dictionary}
    Question: {question}
    Helpful Answer:"""

    llm = HuggingFaceHub(huggingfacehub_api_token = alt_key, repo_id= model_id, model_kwargs={"temperature":0.1, "max_new_tokens":500})

    llm_prompt = PromptTemplate.from_template(template)

    llm_chain = LLMChain(llm=llm,prompt=llm_prompt)

    llm_response = llm_chain.predict(dictionary = str(rag_answer), question = question_to_ask)

    return llm_response