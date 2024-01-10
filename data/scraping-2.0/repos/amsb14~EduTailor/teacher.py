import csv
import streamlit as st
import pandas as pd
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from openai.error import AuthenticationError

def generate_prompt(data, student_id, subject):
    # Filter data for the specific student
    student_data = data[data['Student ID'] == student_id]

    # Get student details
    name = student_data['Full Name'].values[0]
    gender = student_data['Gender'].values[0]
    grade = student_data['Grade'].values[0]
    score = student_data[f'{subject} Score'].values[0]
    conceptual_score = student_data['Conceptual Understanding Score'].values[0]
    participation_level = student_data['Participation Level'].values[0]
    absences = student_data['Number of Absences'].values[0]
    
    # Determine pronouns based on gender
    if gender == 'Female':
        pronoun1, pronoun2, pronoun3 = 'her', 'she', 'her'
    else:
        pronoun1, pronoun2, pronoun3 = 'his', 'he', 'him'
    
    

    # Construct the prompt
    prompt = f"""
You are an AI-driven educational system, and your goal is to generate {{num_questions}} {subject} questions in {{question_format}} format for the {grade}th grade student {name}. Make sure you abide by educational best practices, consider {pronoun1} individual strengths, weaknesses, and participation tendencies, and provide opportunities for {pronoun3} to improve {pronoun1} understanding of key {subject} concepts. Based on the following profile data, generate a series of questions:

- Name: {name}
- Gender: {gender}
- Student ID: {student_id}
- Grade: {grade} (Consider grade-level appropriate complexity in the questions)
- {subject} Score: {score} (Use this to understand {pronoun1} current proficiency in {subject})
- Conceptual Understanding Score: {conceptual_score} (Use this to identify gaps in {pronoun1} conceptual understanding)
- Participation Level: {participation_level} (Use this to understand {pronoun1} classroom engagement level. A lower score may mean {pronoun2} is less confident or has less exposure)
- Number of Absences: {absences} (Consider topics that {pronoun2} may have missed due to absences)


Remember to generate questions that are appropriate for {name}'s grade level, engage {name}'s active participation, target areas where {pronoun2} might improve {pronoun1} conceptual understanding, and cover key topics {pronoun2} may have missed due to absences.

Questions must have a unified format. For example, start with letter Q -> question number -> semicolon -> the question. 

IMPORTANT NOTES:
You SHOULD be aware of the format for the question type requested in this prompt. Nevertheless, For Multiple-Choice Questions (MCQ), provide four choices labelled as A), B), C), and D). Essay questions should have a clear prompt, specific instructions, a focus on critical thinking, adequate context, and a limit on response length. Short answer questions should have clear, concise questions that directly address the topic without requiring extensive explanations. They should focus on key concepts and points. Fill-in-the-blank questions should have clear instructions, contextualize blanks, varied question types, effective distractors, avoid ambiguity, test comprehension and application, balance difficulty, review and revise.
    """
    
    return prompt


# load_dotenv()
# OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Initialize the OpenAI language model
# global llm
# llm = OpenAI(temperature=0.7, model="text-davinci-003")


def check_student_id(llm):
    student_id = st.text_input("Enter the Student ID")
    subject = st.selectbox("Select the Subject", ["Math", "Science", "English"])
    question_format = st.selectbox("Select the Question Format", ["MCQ", "True/False", "Fill in the Blanks", "Short Answer", "Essay Questions"])
    num_questions = st.selectbox("Select the Number of Questions", list(range(1, 11)))

    if st.button("Check Student ID and Generate Questions"):
        # Check if student ID input is empty
        if not student_id.strip():
            st.warning("Please enter a student ID.")
            return False

        # Check if student ID input is a non-numeric string
        try:
            student_id = int(student_id)
        except ValueError:
            st.warning("Invalid student ID. Please enter a numeric student ID.")
            return False

        df = pd.read_csv('student_data.csv')
        if student_id in df['Student ID'].values:
            st.session_state.student_id = student_id
            st.session_state.subject = subject
            st.session_state.question_format = question_format
            st.session_state.num_questions = num_questions
            
            # Get student data for summary
            student_data = df[df['Student ID'] == student_id]
            name = student_data['Full Name'].values[0]
            gender = student_data['Gender'].values[0]
            grade = student_data['Grade'].values[0]
            score = student_data[f'{subject} Score'].values[0]
            conceptual_score = student_data['Conceptual Understanding Score'].values[0]
            participation_level = student_data['Participation Level'].values[0]
            absences = student_data['Number of Absences'].values[0]
            
            # Create and display student summary
            st.markdown(f"""
            **Student Summary:**
            - **Full Name:** {name}
            - **Gender:** {gender}
            - **Grade:** {grade}
            - **{subject} Score:** {score}
            - **Conceptual Understanding Score:** {conceptual_score}
            - **Participation Level:** {participation_level}
            - **Number of Absences:** {absences}
            """)
            
            st.success("The entered student ID exists in the database. Generating questions now.")
            
            try:
                # llm = OpenAI(openai_api_key=openai_api_key, temperature=0.7, model="text-davinci-003")
                llm
                generate_questions()
                
            except AuthenticationError:
                e = RuntimeError('There is an error processing your request right now.')
                st.error(e) 
            
            
        else:
            st.warning("The entered student ID doesn't exist in the database. Please check the ID and try again.")
            return False
    return False



def generate_questions():
    # Get student data
    data = pd.read_csv('student_data.csv')
    student_id = int(st.session_state.student_id)
    subject = st.session_state.subject
    prompt_template = generate_prompt(data, student_id, subject)

    # Create prompt
    prompt = PromptTemplate(
        input_variables=["question_format", "num_questions"],
        template=prompt_template
    )

    # Format prompt
    formatted_prompt = prompt.format(
        question_format=st.session_state.question_format, 
        num_questions=st.session_state.num_questions
    )

    # Generate questions
    response = llm(formatted_prompt)
    
    # Post-process the output to modify the formatting
    response = response.replace("A)", "\nA)").replace("B)", "\nB)").replace("C)", "\nC)").replace("D)", "\nD)")
    response = response.replace(";", "")
    
    # save the generated questions to a file
    with open('questions.txt', 'w') as f:
        for question in response.split('\n'):
            f.write("%s\n" % question)
            
    st.success("The generation of questions for the chosen student has been completed successfully!")
    
    
def display_student_answers():
    # Try to read the CSV file
    try:
        # Read the student answers CSV
        df = pd.read_csv('student_answers.csv')
        # Display the dataframe
        st.dataframe(df)
    except:
        st.write("No student answers available.")   


def read_csv_file(filename):
    data_string = ""

    with open(filename, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            student_id = row['Student ID']
            question = row['Question']
            answer = row['Answer']
            data_string += f"""
{question}\n
Student Answer: {answer}
\n
"""

    return data_string, student_id

def generate_feedback_prmopt(answered_questions, student_id):
    
    
    data = pd.read_csv('student_data.csv')
    student_data = data[data['Student ID'] == int(student_id)]

    # Get student details
    name = student_data['Full Name'].values[0]
    gender = student_data['Gender'].values[0]
    grade = student_data['Grade'].values[0]
    conceptual_score = student_data['Conceptual Understanding Score'].values[0]
    participation_level = student_data['Participation Level'].values[0]
    absences = student_data['Number of Absences'].values[0]


    feedback_engine = PromptTemplate(
        input_variables=["questions_answers"],
        template=f"""
As a teacher, your goal is to provide corrective feedback to the student's answers. Students were given a customized quiz based on their previous performance during the class.

Student Performance Summary:

- Name: {name}
- Gender: {gender}
- Student ID: {student_id}
- Grade: {grade} 
- Conceptual Understanding Score: {conceptual_score}
- Participation Level: {participation_level} 
- Number of Absences: {absences} 


The student has provided answers to two questions. Your task is to correct the student's responses based on the given questions:

###
{{questions_answers}}
###

The response must return the number of correct answers out of the total number of questions. For example:

"
The number of correct answers: [num correct answers] out of [total number of questions] Questions.


Q1: [Question]

Student Answer: [Student Answer]

Correct Answer: [Correct Answer]


[Feedback], for example, -> 
###
Overall, [student name], you scored 0 out of 2 on this quiz. It's important to improve your understanding of these concepts. Your response to the first question regarding the process of plants making food using sunlight was incorrect. The correct answer is "photosynthesis," where plants use sunlight to convert carbon dioxide and water into glucose and oxygen. Additionally, your response to the second question was correct; wood is indeed an example of a solid material. Keep up the good work on your solid material knowledge.

To improve your overall performance, I recommend studying the topic of photosynthesis in plants to better understand the process. It's also essential to review different states of matter and examples of solids to enhance your conceptual understanding. With more practice and effort, you can achieve better results. Keep up the hard work!
###
"


DO NOT include extra information, verbosity, notes etc.

        """
    )
    
    feedback = feedback_engine.format(questions_answers=answered_questions)
    return feedback

def questions_to_dict(input_str):
    lines = input_str.split('\n')
    questions = {}
    current_question = ''
    
    # Check if the last line starts with 'Please note', and if so, remove it
    if lines[-1].strip().startswith('Please') or lines[-1].strip().startswith('Note:'):
        lines = lines[:-1]


        
    for line in lines:
        stripped_line = line.strip()
        
        
        # if stripped_line.startswith("Correct Answer:"):
        #     pass
        
        if stripped_line.startswith('Q'):
            current_question = stripped_line
            questions[current_question] = []
            
        
        elif stripped_line and current_question:
            questions[current_question].append(stripped_line.split(') ')[1])

    return questions


def display_questions_and_collect_answers(questions_choices):
    
    st.markdown(""" ### Welcome, student""")
    
    student_id = st.session_state.student_id # assuming the student ID is stored in session state
    student_answers = {}

    # Loop through the dictionary
    for i, (question, choices) in enumerate(questions_choices.items(), 1):
        # Print the question
        st.write(question)

        # Loop through the choices
        for index, choice in enumerate(choices, 1):
            # Print the choice
            st.write(f"\t{index}. {choice}")

        # Add a line break for readability
        st.write("\n")

        # Collect the student's answer along with the student id
        student_answers[i] = [str(student_id), question, st.text_input(f"Enter Answer for question {i}")]
    
    st.write("\n")

    # When the student clicks the 'Submit Answers' button
    if st.button('Submit Answers'):
        # Save the student's answers to a dataframe
        df = pd.DataFrame.from_dict(student_answers, orient='index', columns=['Student ID', 'Question', 'Answer'])
        # Save the dataframe to a CSV file without the index
        df.to_csv('student_answers.csv', index=False)

        st.write(student_answers)

    return student_answers


def teacher_actions(openai_api_key):
    global llm
    llm = OpenAI(openai_api_key=openai_api_key, 
                 temperature=0.7, 
                 model="text-davinci-003")

    
    tab1, tab2, tab3 = st.tabs(["Questions Generation", "View Students' Details", "View Answers"])
    
    with tab1:
        
        st.markdown(""" ### Welcome, teacher""")
        if "student_id" not in st.session_state:
            st.session_state.student_id = ""
        if "subject" not in st.session_state:
            st.session_state.subject = ""
        if "question_format" not in st.session_state:
            st.session_state.question_format = ""
        if "num_questions" not in st.session_state:
            st.session_state.num_questions = 0
        if st.session_state.student_id and st.session_state.subject and st.session_state.question_format and st.session_state.num_questions:
            generate_questions()
        else:
            check_student_id(llm)
            
            
    with tab2:
        st.markdown("""
    ### Welcome to the Student Details Viewer! 
    This tab displays all the information about our wonderful students in an organized manner. Feel free to browse through to get an understanding of the diverse pool of talent in our class.

    **Quick Tip:** Want to create personalized questions for a student? No problem! Just find their ID in the table below and paste it in the 'Generate Questions' tab. Voila! Tailored questions for individual student needs, in no time. Isn't that exciting?
""")
        data = pd.read_csv('student_data.csv', dtype={'Student ID': str})
        st.dataframe(data)
        
    with tab3:
        st.markdown("""
            ### Review Student Responses
        
            Here you can view each student's answers to the generated questions.
        
            The table below pairs each question with the respective student's ID and their submitted answer. This can assist you in understanding student comprehension and thought processes.
        
            Additionally, viewing this tab initiates the automatic generation of personalized feedback for each student, aiding in individual progress tracking.
        
            Dive in and explore!
        """)

        display_student_answers()
        

        data, id_ = read_csv_file("student_answers.csv")

        prompt = generate_feedback_prmopt(data, id_)
        
        st.markdown("### Student Feedback")
        
        feedback = llm(prompt)
        st.write(feedback)


if __name__ == "__main__":
    teacher_actions()
