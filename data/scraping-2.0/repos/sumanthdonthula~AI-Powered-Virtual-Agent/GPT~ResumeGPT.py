
import os

from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import GPT2Tokenizer
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import streamlit as st



# Specify the file path of the PDF in Google Drive
file_path = "/content/drive/MyDrive/Colab Notebooks/Sumanth Langchain Resume.pdf"

text="""Sumanth Donthula
Mail Id:sdonthula@hawk.iit.edu, donthula.sumanth@gmail.com
Phone:+1 312 752 6725
City: Chicago
State: Illinois
Country: United States

03.29.2023
Dear Hiring Manager,
As an experienced software engineer with a strong background in machine learning and artificial intelligence, I am confident that my skills and expertise align well with the requirements of this role.
Throughout my academic and personal projects, I have honed my skills in software solutions and programming languages, and I am confident in my ability to work with data sets and conduct data-driven analysis. With extensive experience in Machine learning, I have worked on numerous projects that required the development of algorithms and models to uncover hidden patterns and relationships in large datasets.
In my prior experience at Tredence Analytics, I have worked with enormous datasets and utilized various statistical approaches & built Machine Learning Models to extract insights and make inferences. I have also cooperated with cross-functional teams to develop solutions and report findings to stakeholders. These experiences not only provided me with technical abilities, but also taught me the value of good communication and teamwork.
In addition, I have experience working with Python and Machine Learning Libraries, and I am eager to learn new software and tools as needed. I am also a skilled communicator and have experience collaborating cross-functionally with engineering, product management, and customer success teams.
As an intern, I am eager to contribute to your company's goals and learn from experienced professionals. I am excited about the opportunity to work in a small, agile environment where I can gain exposure to various aspects of the business and contribute to scalable data driven solutions. I am confident that I can add value to your team and would be honored to be part of your company's success.
Thank you for considering my application. I look forward to the opportunity to further discuss my qualifications with you.
Sincerely,
Sumanth Donthula


Sumanth Donthula
donthula.sumanth@gmail.com| +1 312 752 6725
SUMMARY:
Conducted Research and built a computer vision-based YOLO model with an average confidence of 90% for objects. Ensured Data Quality, performed EDA & Feature Engineering, and designed a classification model which predicted Breast Cancer with a precision of 0.84 and recall of 0.89. Designed Unified Data Models to enable Data Flow and Business Intelligence using cloud solutions like Snowflake, Azure Data Lake, Azure Data Factory, and Databricks, the architecture reduced time to value by 70%. Performed innovative proof-of-concept for an offshore retailer and built a data pricing pipeline that prorates discounts and reduces manual efforts by 90%. Handled & led our team to work on multiple projects simultaneously and ensured delivery on time. GPA: 4/4.
PROFESSIONAL EXPERIENCE:
Data Analyst June 2021- July 2022
Tredence Analytics, Bangalore, India
 Owned end to end data infrastructure by ingesting sales data and enabling data base systems, which are plugged into BI products to give insights for business growth in the market.
 Developed & Deployed, multiple SQL objects like tables, views & Stored Procedures in Snowflake and automated the data pipelines using scheduled triggers in Azure Data Factory.  Performed Standard Integration Testing, debugging and evaluated performance metrics, and resolved bugs to ensure the quality and reliability of Data pipelines.
 Organized Knowledge Transfer Sessions to mentor & communicate with juniors at the company.
Program Analyst Trainee (Internship) February 2021- June 2021
Cognizant Technologies PVT.Ltd, Chennai, India
 Worked on projects involving AWS, Java, Python, DBMS and SQL.
 Developed a Server less web application with AWS applications.
PROJECTS:

PROJECTS:
Breast Cancer Detection Using Classification Models: Implemented feature selection and built predictive models like SVM, Logistic regression and KNN Models for a small dataset for class segmentation. Performed optimization techniques for different Models and a model with better performance is selected comparing quantitative error metrics.

NBA League Visualization: Processed and prepared the raw datasets using Pandas from multiple MS Excel sources. Transformed the data into Azure SQL tables and created star schema using Azure Data Factory pipelines. Used this star schema as the source to Power BI and created multiple reports.

Real Time object detection YOLO Model and COCO Dataset: Implemented software product of YOLO model trained on COCO (Common Objects in Context) dataset created by Microsoft team for live object recognition using neural nets and OpenCV module.
Breast Cancer Detection Using Classification Models: Implemented feature selection and built classification models like SVM, Logistic regression and KNN Models for a small dataset for class segmentation. Performed optimization techniques and tuning parameters for predictive models and best model is selected based on performance.
EDUCATION:
Masters in Artificial Intelligence Graduation Date: May 2024
Illinois Institute of Technology, Chicago
GPA: 4/4
TECHNICAL SKILLS:
Programming Languages: Python, Java, SQL, Matlab, R
Databases: MySQL, Snowflake, Postgres, SQL server Libraries: Pandas, NumPy, Matplotlib, Plotly, Seaborn, Scikit-Learn, SciPy, Keras, TensorFlow, PyTorch
Others : Statistics, Machine Learning Algorithms, Compute Vision, Deep Learning, Data Structures & Algorithms
CERTIFICATIONS & ACHIEVEMENTS:
1. Microsoft Certified: AI-900 & AZ-900
2. Got “Quick Off Block” Award at Tredence for best customer service and collaboration with stakeholders
INTERESTS:
Natural Language Processing (NLP), Neural Networks, Artificial Intelligence, Computer Vision, Computer Science, Data Science, Recommendation Systems, Problem Solving

Sumanth Donthula
donthula.sumanth@gmail.com| +1 312 752 6725
SUMMARY:
Designed Unified Data Models to enable Data Flow and Business Intelligence using cloud solutions like Snowflake, Azure Data Lake, Azure Data Factory, and Databricks, which reduced time to value by 70%. Performed POC for an offshore retailer and built a data pricing pipeline that prorates discounts and reduces manual efforts by 90%. Handled & led our team to work on multiple projects simultaneously and ensured delivery on time. Built a computer vision-based YOLO model with an average confidence of 90% for objects. Ensured Data Quality, performed EDA & Feature Selection, and designed a classification model which predicted Breast Cancer with a precision of 0.84 and recall of 0.89. GPA: 4/4.
PROFESSIONAL EXPERIENCE:
Data Analyst June 2021- July 2022
Tredence Analytics, Bangalore, India  Owned data streamlining by ingesting data and enabling data base systems, which are plugged to BI tools for business analytics.
 Developed multiple SQL objects like tables, views & Stored Procedures in Snowflake and automated the data pipelines using scheduled triggers in ADF.
 Handled Deployment of SQL and ADF objects from development to Quality and Production via secured releases.  Performed Testing, evaluated performance metrics, and resolved issues to ensure the quality and reliability of Data pipelines.
 Organized Knowledge Transfer Sessions to mentor & communicate with freshers at the company.
Program Analyst Trainee (Internship) February 2021- June 2021
Cognizant Technologies PVT.Ltd, Chennai, India
 Worked on projects involving AWS, Java, Python, DBMS and SQL.
 Developed a Server less web application with AWS services.
PROJECTS:
Breast Cancer Detection Using Classification Models: Implemented feature selection and built classification models like SVM, Logistic regression and KNN Models for a small dataset for class segmentation. Performed optimization techniques and tuning parameters for different models and best model is selected based on performance.
NBA League Visualization: Processed and prepared the raw datasets using Pandas from multiple MS Excel sources. Transformed the data into Azure SQL tables and created star schema using Azure Data Factory pipelines. Built Innovative reports for visualizing analytics through Power BI reports.
EDUCATION:
Masters in Artificial Intelligence Graduation Date: May 2024
Illinois Institute of Technology, Chicago
GPA: 4/4
TECHNICAL SKILLS:
Software Languages: Python, Java, SQL, Matlab, R
Databases: MySQL, Snowflake, Postgres, SQL server Libraries: Pandas, NumPy, Matplotlib, Plotly, Seaborn, Scikit-Learn, SciPy, Keras, TensorFlow, PyTorch Others : AWS, Azure, Git, Power BI, Data Mining , Statistics, Data Modeling, Agile methodology
CERTIFICATIONS & ACHIEVEMENTS:
1. Microsoft Certified: AI-900 & AZ-900
2. Got “Quick Off Block” Award in Tredence Analytics for constant efforts and being collaborative with stakeholders
INTERESTS:
Machine Learning Algorithms, Cloud Computing, Deep Learning, Natural Language Processing, Neural Networks, Artificial intelligence (AI), Predictive Modeling, Data Science, Programming


Masters:
Name Birth Date Student Type
Donthula Sumanth 20-APR Continuing

Curriculum Information

Current Program : Master

Program College Major and

Master College of Computing Department

Artificial Intelligence,
Computer Science

Academic Standing
Good Standing

Academic Transcript

Department: College of Computing

Course:Artificial Intelligence New Student

Sumanth took CS 430: Introduction to Algorithms, CS 480: Introduction to Artificial Intelligence, CS 571: Data Preparation and Analysis, CS 401: Introduction to Advance Studies I, CS 595: thics of Data and Algorithms, and MATH 564: Applied Statistics.


Bachelors:
College:SRM Institute of Science and Technology
Branch: Electronics and Communication Engineering
Courses:Electronic Devices and Circuits, Network Theory, Digital Electronics, Signals and Systems, Analog Electronics, Electromagnetic Theory, Communication Systems, Microprocessors and Microcontrollers, Control Systems, Digital Signal Processing, VLSI Design, Antenna and Wave Propagation, Optical Communication, Embedded Systems, Wireless Communication, Satellite Communication, Microwave Engineering, Computer Networks, Data Communication and Networking, Mobile Communication, Information Theory and Coding, Robotics and Automation.

Github Repository:https://github.com/sumanthdonthula
Developer Portfolio:https://sumanthdonthula.github.io/Portfolio/


Here's a schedule that reflects Sumanths availability:

Monday to Sunday:

Available from 10:00 AM to 9:00 PM
Please note that this schedule applies to all days of the week, including weekends.

Sumanth Has Youtube Channel:

Channel work:https://youtube.com/@donthulasumanth5415

List of his Videos

Here are the YouTube video names from the channel "Donthula Sumanth":

1. Demistifying Software: The Evolution of Programming Algorithms, Data Structures, and Version Control
2. Demystifying Memory: Unveiling the Evolution and Advancements in Memory Technology
3. Demystifying IC Boards: The Power of PCBs, Moore's Law, and Photolithography in Digital Circuits
4. Demystifying Computer Computation: How Your Computer Thinks #CPU #Computer #LearnComputers #Tech
5. Atomic Habits: Tiny Changes, Remarkable Results by James ClearJames Clear • Quick Book 📚 Review •
6. We shall climb again 🌄 #trekking #travelling #travelvlog #nature #peace
7. Pubg Mobile M24 Headshots. Op bhai 🔥#bgmi #bgmivideos #bgmistatus #pubgmobile
8. Vijay Sethupathi Mass Attitude Whatsapp status 🔥
9. Sad Violin Music - Soothing Music🎻 for remembering your Memories #Music #Violin #Musiclover #Bgm
10.Dekho me cigarette🚬 Nahi pita |Scam 1992|Best dialogue Harshad Mehta story |
11. The success story of Spacex in 1 min |Elon musk's journey into Space|Worlds Richest Man
12. Face Recognition in Python Under 20 mins
13. Handwritten Digit Recognition on MNIST dataset |Machine Learning|
14. How to search by Image in Chrome in Android device
15. Batasaari-Sri Sri poem |MAHAPRASTHANAM| Mahakavi Sri Sri
16. Brief Summary Of Rich Dad Poor Dad
17. Gantalu - Sri Sri poem
18. Arjunreddy Saaho BGM mix #Arjunreddy #saaho
19. Nucleya #Deadly Newyear DROPS in #International Festival 2018 #Nucleya #concert #bollywood
20. I poem |MAHAPRASTANAM(Sri sri)|
21. Mahaprastanam poem |Sri sri ( Srirangam srinvasarao)|
22. AIB : Udd Gaye by RITVIZ [Nucleya drop] | NucleyaNeAaagLagaDiya
23. Nucleya #baahubali drop |Baahubali| #nucleya #prabhas #baahubali #ssrajamouli
24. vijay devarakonda mindblowing speech |World famous lover| #worldfamouslover

Please note that these are the video names available in the provided information."""

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')


def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))

# Step 4: Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size = 512,
    chunk_overlap  = 24,
    length_function = count_tokens,
)

chunks = text_splitter.create_documents([text])


os.environ["OPENAI_API_KEY"] = "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"
# Get embedding model
embeddings = OpenAIEmbeddings()

# Create vector database
db = FAISS.from_documents(chunks, embeddings)

# Create QA chain to integrate similarity search with user queries (answer query from knowledge base)
chain = load_qa_chain(OpenAI(temperature=0), chain_type="stuff")

st.set_page_config(
    page_title="Sumanth's Virtual Agent",
    page_icon="👋",
)

st.markdown("<h1 style='font-size:24px;'>👋 Hey! How's the day going? Nice to meet you. I am Sumanth's virtual agent. Grab a coffee ☕ and have a laid-back conversation about Sumanth's experience, projects, and curriculum.</h1>", unsafe_allow_html=True)


query = st.text_input("Please enter your query:")

# Perform actions based on the query
if query:
    # Check similarity search is working
    docs = db.similarity_search(query)

    # Get response from the QA chain
    response = chain.run(input_documents=docs, question=query)
    response="😎"+response

    st.write(response)

