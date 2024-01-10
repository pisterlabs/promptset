import streamlit as st
import numpy as np
import pandas as pd
import faiss

from sklearn.feature_extraction.text import TfidfVectorizer
from streamlit_chat import message

from langchain.chains import ConversationChain
from langchain.llms import OpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
import os
import google.generativeai as genai
from langchain.schema import HumanMessage, SystemMessage
import networkx as nx
import plotly.graph_objects as go
# Load data
data = pd.read_csv(r'coursera_courses.csv')

# TF-IDF vectorization
course_corpus = data['course_title']
vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 3), min_df=5)
X = vectorizer.fit_transform(course_corpus)

# Convert sparse matrix to numpy array
X_array = np.float32(X.toarray())

# Create Faiss index
index = faiss.IndexFlatL2(X_array.shape[1])
index.add(X_array)

#for LLM gemini

os.environ['GOOGLE_API_KEY'] = "AIzaSyCvXu33gltO3ZEL5WRjqSyrl4ANgDeO84o"
genai.configure(api_key = os.environ['GOOGLE_API_KEY'])
llm = ChatGoogleGenerativeAI(model="gemini-pro")
prompt = ChatPromptTemplate(
    messages=[
        # SystemMessagePromptTemplate.from_template(
        #     "you are a course recommender, you ask the candidates a few questions to get his personal interests, end goals and current skill level and provide him with a curated list of courses alongside mentioning its difficulty level as beginner, intermediate and advanced."
        # ),
        # The `variable_name` here is what must align with memory
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("Imagine you are a course recommender, ask the candidates questions one by one to get his personal interests, end goals and current skill level and provide him with a curated list of courses alongside mentioning its difficulty level as beginner, intermediate and advanced. do not ask multiple questions at once. wait for the user to answer each question one by one and after 4-5 question provide him with a list of courses. suggest atleat one course from knowvationlearnings.in"),
    ]
)

# Notice that we `return_messages=True` to fit into the MessagesPlaceholder
# Notice that `"chat_history"` aligns with the MessagesPlaceholder name
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
conversation = LLMChain(llm=llm, prompt=prompt, verbose=True, memory=memory)

def create_and_plot_graph(recommendations):
    # Create a directed graph
    G = nx.DiGraph()

    # Define courses for different levels
    # beginner_courses = recommendations['Beginner']

    # intermediate_courses = recommendations['Intermediate']

    # advanced_courses = recommendations['Advanced']

    beginner_courses = [
        ("Data Science from Johns Hopkins University", "Fractal Data Science from Fractal Analytics"),
    ("Data Science from Johns Hopkins University", "What is Data Science? from IBM"),
    ("IBM Data Science from IBM", "SQL for Data Science from University of California, Davis"),
    
    
    ("IBM Data Science from IBM", "Data Science Math Skills from Duke University"),
    ("Tools for Data Science from IBM", "Practical Data Science with MATLAB from MathWorks")# Add more beginner courses as needed
    ]

    intermediate_courses = [
        ("Data Science with Databricks for Data Analysts from Databricks", "Genomic Data Science from Johns Hopkins University"),
        ("IBM Data Science from IBM", "Introduction to Data Science from IBM"),
    ("IBM Data Science from IBM", "Tools for Data Science from IBM"),
    ("IBM Data Science from IBM", "Applied Data Science from IBM")# Add more intermediate courses as needed
    ]

    advanced_courses = [
        ("Genomic Data Science from Johns Hopkins University", "Foundations of Data Science from Google"),
        ("IBM Data Science from IBM", "Executive Data Science from Johns Hopkins University"),
    ("IBM Data Science from IBM", "Data Science Methodology from IBM")# Add more advanced courses as needed
    ]
    # Add nodes and edges to the graph
    G.add_edges_from(beginner_courses)
    G.add_edges_from(intermediate_courses)
    G.add_edges_from(advanced_courses)

    # Positioning nodes in the graph
    pos = nx.spring_layout(G, seed=42)

    # Create a Plotly figure
    fig = go.Figure()

    # Add nodes to the figure
    for node in G.nodes:
        x, y = pos[node]
        fig.add_trace(go.Scatter(
            x=[x],
            y=[y],
            mode="markers",
            marker=dict(size=16, color="skyblue"),
            text=node,
            hoverinfo="text",
            name=node,
        ))

    # Add edges to the figure
    for edge in G.edges:
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        fig.add_trace(go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            mode="lines",
            line=dict(color="gray", width=0.5),
            hoverinfo="none",
        ))

    # Customize plot appearance
    fig.update_layout(
        title_text="Data Science Courses Flow Diagram",
        title_x=0.5,
        showlegend=False,
        hovermode="closest",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    )
    
    return fig
# Function to recommend courses by difficulty
def recommend_courses_by_difficulty(title):
    search_text = [title]
    search_text_vector = vectorizer.transform(search_text)
    search_text_vector_array = np.float32(search_text_vector.toarray())
    distances, indices = index.search(search_text_vector_array, 15)

    recommendations = {'Beginner': [], 'Intermediate': [], 'Advanced': []}

    for i in range(15):
        course_title = data['course_title'][indices[0][i]]
        organization = data['course_organization'][indices[0][i]]
        difficulty = data['course_difficulty'][indices[0][i]]
        link = data['course_url'][indices[0][i]]

        recommendation = f"[{course_title}]({link}) from {organization}\nDifficulty: {difficulty}"

        # Add the recommendation to the corresponding difficulty level
        recommendations[difficulty].append(recommendation)

        # Break once we have 3 recommendations for each difficulty level
        if all(len(recommendations[level]) >= 3 for level in ['Beginner', 'Intermediate', 'Advanced']):
            break

    return recommendations

def langchain_conversation(user_input):
    # st.write("in lgconv")
    messages = []

    # Append user input to messages
    # messages.append(HumanMessage(content=user_input))

    # Ask a new user prompt
    # prompt = HumanMessage(content="Can you tell me about the LLMChain in LangChain?")
    # messages.append(prompt)
    # st.write("gening resp")
    try:
      response = conversation({"question": user_input})
      # st.write(response["chat_history"][1])
      return response

    except Exception as e:
    # Catching any other exceptions
      print("An error occurred:", e)

# Streamlit App
def get_text():
    input_text = st.text_input("You: ", "Hello, how are you?", key="input")
    return input_text
# user_input = get_text()

st.set_page_config(page_title="Course Compass", page_icon=":robot:")
st.header("Course Compass - Gemini AI + vector DB")
st.sidebar.title("Select Mode")

# Sidebar radio buttons for mode selection
selected_mode = st.sidebar.radio("Select Mode", [ "AI Course Strategizer", "AI Course Counsellor"])

def talk():
  st.subheader("AI Counsellor Conversation:")
  uploaded_file = "/content/mona_talking.mp4"
  if uploaded_file is not None:
      # video_bytes = uploaded_file.read()
      st.video(uploaded_file, format='video/mp4')

  if st.button("Start Conversation"):
    st.subheader("Hold space bar while you speak..")
    st.subheader("Running ASR(Automatic Speech Recog) on your voice")
    st.subheader("Parsing the input and passing to the LLM model")
    st.subheader("LLM Model Output spoken back by AI counsellor")
    st.subheader("OUTPUT: Curated learning path recommendation and Roadmap")

def chat():
  # user_input = st.text_input("search courses based on keywords:", "")

    # Add logic to ask questions based on user's interests and generate recommendations
  # if st.button("Ask AI"):
  #       # Add logic to generate AI responses and recommendations based on user input
  #       st.subheader("AI's Response:")
  #       st.text("AI's response goes here.")

# Course recommendation feature with search bar
  image = open('/content/diag_hackverse.jpg', 'rb').read()
  st.image(image, caption='The backend Architecture')

  st.subheader("Course Recommendation from data:")
  user_input = st.text_input("Search for a course:", "data science")
  if st.button("Search using FAISS vector Search"):
      # Add logic to display course recommendations based on the search query
      recommended_courses = recommend_courses_by_difficulty(user_input)

      for difficulty, recommendations in recommended_courses.items():
          st.subheader(f"{difficulty} Courses:")
          for recommendation in recommendations:
              st.markdown(recommendation, unsafe_allow_html=True)
      
      # if st.button("show roadmap"):
      st.title("Data Science Courses Flow Diagram")
      st.plotly_chart(create_and_plot_graph(recommended_courses))  
        
  if "generated" not in st.session_state:
      
      st.session_state["generated"] = []

  if "past" not in st.session_state:
      st.session_state["past"] = []

  user_input2 = st.text_input("ask gemini:", "i wanna be a data scientist")

  if user_input2 and st.button("Ask Gemini AI!"):
      # output = chain.run(input=user_input)
      output = langchain_conversation(user_input2)
      st.session_state.past.append(user_input2)
      st.session_state.generated.append(output["chat_history"][1].content)
      # st.write(output["chat_history"][1].content)
  if st.session_state["generated"]:

      for i in range(len(st.session_state["generated"]) - 1, -1, -1):
          message(st.session_state["generated"][i], key=str(i))
          message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
# Display latest curated blogs
# st.subheader("Latest Curated Blogs:")
# Add logic to display the latest curated blogs based on the backend model

# Marquee text
# st.markdown("<p style='color:red; font-size:20px;'>Selective courses curated just for you from 6000 courses</p>", unsafe_allow_html=True)

# Display recommendations based on the selected mode
if selected_mode == "AI Course Counsellor":
    # Display AI avatar and conversation
    talk()
    # Add logic to display AI avatar conversation and recommendations based on the conversation

elif selected_mode == "AI Course Strategizer":
    # Input box for user to enter text
    chat()








