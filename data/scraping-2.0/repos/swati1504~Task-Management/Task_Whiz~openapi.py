import openai
import re 

class OpenAPI:
    def __init__(self, api_key):
        self.api_key = api_key
        openai.api_key = self.api_key

    def decompose_task(self, task_description):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages = [
              {
              "role": "system",
              "content": "You are an advanced AI trained to divide complex tasks into subtasks."
              },
              {
              "role": "user",
              "content": "Task: Develop a New Social Media Platform. Divide this into subtasks."
              },
              {
              "role": "assistant",
              "content": "Frontend Developer: Design a user-friendly interface. Backend Developer: Build scalable server architecture. R&D: Research innovative social interaction features. HR: Plan recruitment for expanding development team."
              },
              {
              "role": "user",
              "content": "Task: Create an E-commerce Website. How do you divide this task?"
              },
              {
              "role": "assistant",
              "content": "Frontend Developer: Design an engaging and easy-to-navigate product display. Backend Developer: Implement secure payment processing systems. Backend Developer: Develop an efficient order management system."
              },
              {
              "role": "user",
              "content": "Task: Launch a Company Intranet. Divide this into subtasks."
              },
              {
              "role": "assistant",
              "content": "Frontend Developer: Create a dashboard with personalized access. Backend Developer: Set up a secure internal communication network. HR: Conduct training for employees on intranet usage."
              },
              {
              "role": "user",
              "content": "Task: Introduce a Fitness Tracking App. How do you divide this task?"
              },
              {
              "role": "assistant",
              "content": "Frontend Developer: Design a user-friendly interface with interactive charts. Backend Developer: Develop algorithms for activity tracking and data analysis. R&D: Research advanced health monitoring technologies."
              },
              {
              "role": "user",
              "content": "Task: Organize a Virtual Reality Game Development. Divide this into subtasks."
              },
              {
              "role": "assistant",
              "content": "Frontend Developer: Design immersive and interactive game environments. Backend Developer: Program robust game mechanics. R&D: Experiment with cutting-edge VR technology. HR: Recruit skilled VR developers and designers."
              },
              {
              "role": "user",
              "content": f"Now, divide the task of '{task_description}' into subtasks."
              }
            ],
            max_tokens=500,
            temperature=0
        )
        return self.extract_tasks(response.choices[0].message.content)

    @staticmethod
    def extract_tasks(chatbot_output):
        pattern = r"(Frontend Developer|Backend Developer|HR|R&D): ([^.]*)"
        matches = re.findall(pattern, chatbot_output)

        tasks_dict = {}
        for match in matches:
            task_type, task_description = match
            tasks_dict[task_description.strip()] = task_type

        return tasks_dict

    def generate_employee_review(self, data, employee_name):
        employee_data = self.get_employee_data(data, employee_name)
        context = self.prepare_context(employee_data)

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages = [
                {"role": "system", "content": "You are an advanced AI trained to provide insightful reviews of employee performance based on given data."},
                {"role": "user", "content": '''Employee Review:
            Name: Ram Prasad
            ID: 1
            Gender: Male
            Age: 34
            Marital Status: Married
            Role: Backend Developer
            Salary: 2,60,000
            Position: Senior
            Absences: 2
            Projects Completed: 42
            Mean Monthly Hours: 143
            Years in the Company: 5
            Joining Year: 2015
            Current Employment Rating: 9
            Moral: Very High
            Stress & Burnout Score: 3
            Ongoing Project Count: 3
            Projects Within Deadline: 22
            Project Start Date: 5/2/2024
            Project Description: Design an efficient algorithm for data processing and visualization
            Project Difficulty: High
            Project Deadline: 15/3/2024
            Manager ID: 4
            Completion Time: 37'''},
                {"role": "assistant", "content": '''Ram Prasad, a Senior Backend Developer, has been a notable asset to our team for five years. His track record of completing 42 projects, including 22 within the deadline, highlights his efficiency and dedication. Specializing in high-difficulty tasks, such as designing algorithms for data processing, he demonstrates both technical expertise and innovative problem-solving skills.

            Despite a high workload, as indicated by his 143 mean monthly hours, Ram maintains a 'Very High' morale and manages a balance between work and well-being, evidenced by his resilience in high-stress environments. His long tenure since 2015 and an impressive employment rating of 9 reflect his consistent performance and value to the team.

            While managing stress more effectively could be a focus area, Ram's overall contribution, marked by a strong work ethic and a positive attitude, is highly commendable. His growth and success are integral to our team's progress.'''},
                {"role": "user", "content": "Employee Review:\nName: Sara Jenkins\nRole: Project Manager\nYears in the company: 3\nProjects Completed: 30\nProject Success Rate: 90%\nMean Monthly Hours: 130\nJoining Year: 2017\nCurrent Employment Rating: 8.5\nStress Level: Moderate"},
                {"role": "assistant", "content": "Sara Jenkins, a Project Manager, has shown remarkable leadership over three years. Her management of 30 projects, with a 90% success rate, showcases her project coordination skills. Balancing 130 monthly hours, she demonstrates a work-life balance, maintaining moderate stress. Her tenure since 2017 and a rating of 8.5 reflect her strong alignment with our goals. Sara's potential for further growth in strategic planning is exciting."},
                {"role": "user", "content": "Employee Review:\nName: Alex Martinez\nRole: IT Specialist\nYears in the company: 4\nProjects Completed: 50\nProject On-Time Completion: 80%\nMean Monthly Hours: 120\nJoining Year: 2016\nCurrent Employment Rating: 8\nStress Level: Low"},
                {"role": "assistant", "content": "Alex Martinez, an IT Specialist for four years, has successfully completed 50 projects, 80% on time, showing his technical prowess and timeliness. Working 120 hours monthly, he balances efficiency and personal time. Since joining in 2016 with a rating of 8, Alex has consistently contributed to our tech advancements, while maintaining low stress levels. His approach to project management and time efficiency is highly effective."},
                {"role": "user", "content": "Employee Review:\nName: Emily Chen\nRole: Marketing Analyst\nYears in the company: 2\nProjects Completed: 20\nProject Success Rate: 85%\nMean Monthly Hours: 135\nJoining Year: 2018\nCurrent Employment Rating: 7.5\nStress Level: High"},
                {"role": "assistant", "content": "Emily Chen, a Marketing Analyst for two years, has completed 20 projects with an 85% success rate, demonstrating her analytical skills and creativity. Working 135 hours monthly, she shows dedication, though her high stress level suggests a need for better work-life balance. Since 2018, with a rating of 7.5, Emily has been a growing asset. Her ability to manage stress while maintaining productivity will be key to her future success."},
                {"role": "user", "content": "Employee Review:\nName: Michael Brown\nRole: Sales Executive\nYears in the company: 6\nProjects Completed: 60\nProject Success Rate: 95%\nMean Monthly Hours: 150\nJoining Year: 2014\nCurrent Employment Rating: 9.2\nStress Level: Medium"},
                {"role": "assistant", "content": "Michael Brown, a Sales Executive for six years, has impressively completed 60 projects with a 95% success rate. His dedication is evident from 150 monthly hours, balancing substantial workload with medium stress levels. Since joining in 2014 with a high rating of 9.2, Michael has been a key player in driving sales growth. His ability to handle stress and maintain high performance is commendable and crucial for his role."},
                {"role": "user", "content": context}
            ],
            max_tokens=500,
            temperature=0
        )
        return response.choices[0].message.content

    @staticmethod
    def get_employee_data(df, employee_name):
        return df[df['Employee Name'] == employee_name].to_dict('records')[0]

    @staticmethod
    def prepare_context(employee_data):
        context = (
            f"Employee Review:\n"
            f"Name: {employee_data['Employee Name']}\n"
            f"ID: {employee_data['ID']}\n"
            f"Gender: {employee_data['Gender']}\n"
            f"Age: {employee_data['Age']}\n"
            f"Marital Status: {employee_data['Married']}\n"
            f"Role: {employee_data['Role']}\n"
            f"Salary: {employee_data['Salary']}\n"
            f"Position: {employee_data['Position']}\n"
            f"Absences: {employee_data['Absences']}\n"
            f"Projects Completed: {employee_data['Projects_Completed']}\n"
            f"Mean Monthly Hours: {employee_data['Mean Monthly Hours']}\n"
            f"Years in the Company: {employee_data['Years in the company']}\n"
            f"Joining Year: {employee_data['Joining_Year']}\n"
            f"Current Employment Rating: {employee_data['Current_Employ_Rating']}\n"
            f"Moral: {employee_data['Moral']}\n"
            f"Stress & Burnout Score: {employee_data['Stress & Burnout Score']}\n"
            f"Ongoing Project Count: {employee_data['Ongoing_Project_Count']}\n"
            f"Projects Within Deadline: {employee_data['Projects_Within_Deadline']}\n"
            f"Project Start Date: {employee_data['Project_Start_Date']}\n"
            f"Project Description: {employee_data['Project_Description']}\n"
            f"Project Difficulty: {employee_data['Project_Difficulty']}\n"
            f"Project Deadline: {employee_data['Project_Deadline']}\n"
            f"Manager ID: {employee_data['Manager_ID']}\n"
            f"Completion Time: {employee_data['Completion_Time']}\n"
        )
        return context
    
        # Email Generation
    def generate_email(self, input_text):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-1106",
            messages=[
                {"role": "system", "content": "You are a highly creative assistant that writes emails based on the given talking points."},
            # {"role": "user", "content": "leave for 5 days, urgent family emergency, dates: 10th to 15th Jan'2024"}
                {"role": "user", "content": input_text}
            ],
            max_tokens = 700,
            temperature = 0
        )
        return response.choices[0].message.content

    # Text Summarisation
    def summarise(self, input_text):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-1106",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that provides a concise summary of the text."},
                {"role": "user", "content": input_text}
            ],
            max_tokens = 200,
            temperature = 0
        )
        return response.choices[0].message.content

    # Improve writing
    def improve(self, original_text):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-1106",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that improves the text. Make it professional and give it a formal tone."},
                {"role": "user", "content": original_text}
            ],
            max_tokens = 500,
            temperature = 0
        )
        return response.choices[0].message.content