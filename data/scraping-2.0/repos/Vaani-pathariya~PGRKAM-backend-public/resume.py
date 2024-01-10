import openai
import difflib

# Set your OpenAI API key
openai.api_key 

# Function to generate answers to questions
def generate_answer(question):
    prompt = f"Question: {question}\nAnswer:"
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=150,
        temperature=0.7,
        stop=None
    )
    answer = response['choices'][0]['text'].strip()
    return answer

# Function to check if the user's answer is correct
def is_answer_correct(user_answer, correct_answer):
    # Use difflib to get a similarity ratio
    similarity_ratio = difflib.SequenceMatcher(None, user_answer.lower(), correct_answer.lower()).ratio()

    # You can adjust the threshold based on your needs
    similarity_threshold = 0.7

    return similarity_ratio >= similarity_threshold

# Example usage
question_to_ask = """Skills Summary
•Languages : Python,C ,C++, JavaScript, SQL, HTML,CSS
•Frameworks : Scikit learn,TensorFlow, LangChain, Node.JS, Express,JS
•Development and Visualization Tools : Tableau, Microsoft Power BI, Streamlit, Postman
•Databases : MongoDB, SQLite, MySQL
•Version Control and Collaboration : Git, Github
•Soft Skills : Leadership, Event Management, Writing, Public Speaking, Time Management,Decision Making, Technical
Communication, Presentation skills
Projects
•PDF Chat App using LangChain (LangChain, Streamlit, OpenAI) : Creating an innovative PDF interaction system
enabling chatbot-style engagement with uploaded documents. Users can interact with the content as they would with a
chatbot, obtaining parsed results derived from the PDF content. Tech: LangChain, Streamlit, OpenAI API (October ’23 and
ongoing)
•Analytics Dashboard for Business Insights (Tableau) : Designed and developed an analytics dashboard for AtliQ
Hardware focused on providing comprehensive business insights. Leveraged Tableau to create an interactive and intuitive
platform for visualising and analysing data, enabling informed decision-making and actionable insights.Tech: SQL,
Tableau,MySQL (November ’23)
•LLM Chabot for Finance Decisions through URLs (Web scraping, OpenAI API, LangChain) : Working on an
LLM Chatbot tailored for finance decision-making using web scraping, and LangChain. The chatbot processes financial
information extracted from URLs and provides information in a conversational manner. Tech: LangChain, Streamlit,
Python(Ongoing)
•SEHAT- A software for telemedicine kiosk for rural India connecting to E-Sanjeevni App : ( MachineLearning,Sensors,Web development)Constructed 4 ML models- Heart disease prediction, Cataract detection, Skin disease
detection and Diabetes detection to detect diseases via sensors(spo2 level, heartbeat, blood sugar level) and image data
integrated with flask backend. Tech: Scikit-learn, Python(August’23)
•Heart disease prediction (Web Development, Machine Learning) : Implemented a heart disease prediction system.
The project aims to forecast the likelihood of heart disease by employing various machine learning algorithms and data
analysis techniques. Tech: Python, Scikit-learn (September ’23)
•Diabetes prediction(Web Development, Machine Learning) : Implemented a diabetes prediction system for women.
The project aims to forecast the likelihood of diabetes by employing various machine learning algorithms and data analysis
techniques. Tech: Python, Scikit-learn (September 23)
Honors and Awards
•TATA Imagination Challenge, 2023 Semi-Finalist - Oct,2023
•National Second Runner’s Up at Trident Hacks and 1st Runner’s Up at North Region Category - June 2023
•Participated in Google Solution Challenge,2023 - March 2023
•Qualified JEE Mains,2021
•Gold medallist in International English Olympiad 2017 and 2019
Volunteer Experience
•Programmer at Google Developer Student Clubs JSSATEN NOIDA, India
Conduct online and offline technical workshops. Created LLM bot for GDSC JSSATEN . March 2022 - Present
•Alpha Microsoft Student Learn Ambassador online
Conducting online workshops on ML and Github.Participation in Microsoft Learn challenges. March 2023 - Present
•Volunteer at Google Developers Group Gurugram Gurugram, India
Organizing events like Devfest and Google i/o extended September,23 - Present
Devanshi Bahuguna Email: devanshibahuguna007@gmail.com
LinkedIn:devanshi-bahuguna
Github: github.com/devanshibahuguna
Education
•JSS Academy of Technical Education, NOIDA NOIDA, India
Bachelor of Technology - Computer Science and Engineering December 2021 - June 2025
Courses: Operating Systems, Data Structures, Analysis Of Algorithms, Artificial Intelligence, Machine Learning, Networking, Databases
•Somerville School, NOIDA NOIDA,India
12th- 96.5 percentage(CBSE) April 2006- March 2020
Skills Summary
•Languages : Python,C ,C++, JavaScript, SQL, HTML,CSS
•Frameworks : Scikit learn,TensorFlow, LangChain, Node.JS, Express,JS
•Development and Visualization Tools : Tableau, Microsoft Power BI, Streamlit, Postman
•Databases : MongoDB, SQLite, MySQL
•Version Control and Collaboration : Git, Github
•Soft Skills : Leadership, Event Management, Writing, Public Speaking, Time Management,Decision Making, Technical
Communication, Presentation skills
Projects
•PDF Chat App using LangChain (LangChain, Streamlit, OpenAI) : Creating an innovative PDF interaction system
enabling chatbot-style engagement with uploaded documents. Users can interact with the content as they would with a
chatbot, obtaining parsed results derived from the PDF content. Tech: LangChain, Streamlit, OpenAI API (October ’23 and
ongoing)
•Analytics Dashboard for Business Insights (Tableau) : Designed and developed an analytics dashboard for AtliQ
Hardware focused on providing comprehensive business insights. Leveraged Tableau to create an interactive and intuitive
platform for visualising and analysing data, enabling informed decision-making and actionable insights.Tech: SQL,
Tableau,MySQL (November ’23)
•LLM Chabot for Finance Decisions through URLs (Web scraping, OpenAI API, LangChain) : Working on an
LLM Chatbot tailored for finance decision-making using web scraping, and LangChain. The chatbot processes financial
information extracted from URLs and provides information in a conversational manner. Tech: LangChain, Streamlit,
Python(Ongoing)
•SEHAT- A software for telemedicine kiosk for rural India connecting to E-Sanjeevni App : ( MachineLearning,Sensors,Web development)Constructed 4 ML models- Heart disease prediction, Cataract detection, Skin disease
detection and Diabetes detection to detect diseases via sensors(spo2 level, heartbeat, blood sugar level) and image data
integrated with flask backend. Tech: Scikit-learn, Python(August’23)
•Heart disease prediction (Web Development, Machine Learning) : Implemented a heart disease prediction system.
The project aims to forecast the likelihood of heart disease by employing various machine learning algorithms and data
analysis techniques. Tech: Python, Scikit-learn (September ’23)
•Diabetes prediction(Web Development, Machine Learning) : Implemented a diabetes prediction system for women.
The project aims to forecast the likelihood of diabetes by employing various machine learning algorithms and data analysis
techniques. Tech: Python, Scikit-learn (September 23)
Honors and Awards
•TATA Imagination Challenge, 2023 Semi-Finalist - Oct,2023
•National Second Runner’s Up at Trident Hacks and 1st Runner’s Up at North Region Category - June 2023
•Participated in Google Solution Challenge,2023 - March 2023
•Qualified JEE Mains,2021
•Gold medallist in International English Olympiad 2017 and 2019
Volunteer Experience
•Programmer at Google Developer Student Clubs JSSATEN NOIDA, India
Conduct online and offline technical workshops. Created LLM bot for GDSC JSSATEN . March 2022 - Present
•Alpha Microsoft Student Learn Ambassador online
Conducting online workshops on ML and Github.Participation in Microsoft Learn challenges. March 2023 - Present
•Volunteer at Google Developers Group Gurugram Gurugram, India
Organizing events like Devfest and Google i/o extended September,23 - Present
What is the name? , give the answer in json format
"""
correct_answer = generate_answer(question_to_ask)

user_answer = input(f"Question: {question_to_ask}\nEnter your answer: ")

if is_answer_correct(user_answer, correct_answer):
    print("Correct!")
else:
    print(f"Incorrect. The correct answer is: {correct_answer}")
