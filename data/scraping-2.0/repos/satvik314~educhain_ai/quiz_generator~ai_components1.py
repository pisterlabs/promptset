import os
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain ,create_extraction_chain,ConversationChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
import plotly.graph_objects as go
from langchain.chains.question_answering import load_qa_chain
import pickle
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

llm = ChatOpenAI(model_name='gpt-4', temperature=0.0)
memory = ConversationBufferMemory()


def respond_to_query(query):
    llm_agent = ConversationChain(
        llm=llm,
        memory=memory,
    )
    return llm_agent.run(query)
def sort_objects(obj_list):

    question = []
    options = []
    correct = []

    for obj in obj_list:

        if 'question' in obj :
            question.append(obj['question'])
        for i in range(3):
            list=[]
            if 'option1' in obj :
                list.append(obj['option1'])
            if 'option2' in obj :
                list.append(obj['option2'])
            if 'option3' in obj :
                list.append(obj['option3'])
        options.append(list)
        if 'correct answer' in obj :
            correct.append(obj['correct answer'])

    return [question,options,correct]

def create_ques_ans(number_of_qn,board,classe, subject , lesson , topic,standard):
    if standard is "Basic":
        level="Remembering, Understanding"
    if standard is "Intermediate":
        level="Applying, Analyzing"
    if standard is "Advanced":
        level="Evaluating or complex numerical"
    
    # template =f"""Prepare {number_of_qn} multiple choice questions on {board} board {classe} ,{subject} subject , {lesson} on {topic}
    # in {level} levels of blooms taxonomy. generate a python list which contains {number_of_qn} sublists . In each python sublist ,
    # first element should be the question. Second , third and fourth elements should be the only 3 options , 
    # and fifth element should be the complete correct option to the question exactly as in options .avoid unnecesary text connotations,
    # extra whitespaces and also avoid newlines anywhere , terminate the lists and strings correctly"""
    template=template =f"""Create {number_of_qn} multiple choice questions on {lesson} on {topic} with 3 options
    in {level} levels of blooms taxonomy. """
    # generate a python list which contains {number_of_qn} sublists . In each python sublist ,
    # first element should be the question. Second , third and fourth elements should be the only 3 options , 
    # and fifth element should be the complete correct option to the question exactly as in options .avoid unnecesary text connotations,
    # extra whitespaces and also avoid newlines anywhere , terminate the lists and strings correctly"""
    
    llm = ChatOpenAI(model = "gpt-4")

    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    with open(os.getcwd()+'/Vector_DB/CBSE-9th-Motion.pkl','rb') as f:
        chunks=pickle.load(f)
    db = Chroma.from_texts(chunks,embedding=embeddings)
    qa_chain = load_qa_chain(llm, chain_type="stuff",verbose=True)
    matching_docs = db.similarity_search(template)
    answer =  qa_chain.run(input_documents=matching_docs, question=template)

    schema = {
    "properties" : {
        "question" : {"type" : "string"},
        "option1" : {"type" : "string"},
        "option2" : {"type" : "string"},
        "option3" : {"type" : "string"},
        "correct answer" : {"type" : "string"}
    },
    "required" : ["question", "options","correct_answer"]
    }
    
    llm2=ChatOpenAI(model="gpt-3.5-turbo-0613")
    chain = create_extraction_chain(schema, llm2)
    response = chain.run(answer)
     
    return sort_objects(response) 
def report(list,score,total):
    
    # Create the linear graph (line plot) with labeled sections
    fig = go.Figure()

# Create the linear graph (line plot) with labeled sections
    fig.add_trace(go.Scatter(x=[0, total], y=[0, 100], mode='lines', line=dict(color='gray', dash='dash'), name='Diagonal Line'))
    fig.add_trace(go.Scatter(x=[0.4*total, 0.4*total], y=[0, 120], mode='lines', line=dict(color='red', dash='dash'), name='40% Line'))
    fig.add_trace(go.Scatter(x=[0.75*total, 0.75*total], y=[0, 120], mode='lines', line=dict(color='blue', dash='dash'), name='75% Line'))
 
    point_x = score  # Replace this with the number of correct answers you want to plot
    point_y = (score / total) * 100  # Calculate the percentage for the given point
    fig.add_trace(go.Scatter(x=[point_x], y=[point_y], mode='markers', marker=dict(color='green', size=10), name='Plotted Point'))
    fig.add_annotation(text=f'(you are here at {point_y:.2f}%)', x=point_x, y=point_y, showarrow=True, arrowhead=2)

    fig.update_layout(
        xaxis=dict(range=[0, total]),
        yaxis=dict(range=[0, 150]),
        xaxis_title='Number of Correct Answers',
        yaxis_title='Percentage of Correct Responses',
        title='Correct Answers vs. Percentage of Correct Responses',
        showlegend=True
    )

    fig.add_shape(
        type="line",
        x0=point_x,
        x1=point_x,
        y0=0,
        y1=point_y,
        line=dict(color="gray", dash="dash"),
    )

    # Label the sections
    fig.add_annotation(text='need to go through the lesson again', x=0.15*total, y=130, showarrow=False, font=dict(color='red'))
    fig.add_annotation(text='revise concepts of wrong answers', x=0.5*total, y=125, showarrow=False, font=dict(color='blue'))
    fig.add_annotation(text='well done , look at solutions once', x=0.95*total, y=130, showarrow=False, font=dict(color='green'))

    # Display the plot in Streamlit app using st.plotly_chart()
    st.plotly_chart(fig)


    template =f"""U are provided with a list of questions {{question}} and list of coreesponding answers{list[1]} marked .
    Suggest if any reading or clairty is required in concepts. dont write anythign unnecesary"""
    prompt = PromptTemplate.from_template(template)
    gpt4_model = ChatOpenAI(model="gpt-4",temperature=0.8)
    quizzer = LLMChain(prompt = prompt, llm = gpt4_model)
    a=quizzer.run(question=list[0])
    return a
