import os
import openai

with open('/home/silas/foundation-files/sk', 'r') as file:
    secret_key = file.read().strip()

os.environ["OPENAI_API_KEY"] = secret_key

openai.api_key = os.getenv("OPENAI_API_KEY")

# DEBUG
skipping = True

# get job post from user
def getJobPost():
    if skipping:
        return '''Job Opening Summary
 The DSS analyst will play a key role in delivering essential data through providing high-level reporting and analytics for key quality improvement, patient safety and service initiatives. This position will provide ad hoc reporting, maintain standard reporting and forecast key business processes using various systems, databases and tools, including MS Office Suite, MS SQL Server, MS Visual Studio and SAP Business Objects.

 This position will work with physicians and data science teams to manage and analyze data pertaining to the needs of assigned projects. The incumbent will communicate and explain complex data and information to leaders at all levels of the organization.

 This role reports to the manager of DSS (reporting) and interfaces with a wide customer base from unit managers to executive leadership, providing insights on both business and clinical operations. Builds and maintains positive relationships with clients while utilizing industry and subject-matter best practices. Translates data for reports, which requires a keen understanding of the health care business, operational processes and the ability to perform complex analyses using data resources and technical tools.

 Job Opening Qualifications
 Minimum Education and Experience Requirements:

 Minimum Education:

 Bachelor's degree in STEM (science, technology, engineering and math).
 Business analytics or a related field required.


 Minimum Job Experience:

 Two years of health care data analysis experience with in-depth knowledge of health care operations required. An advanced degree may substitute as health care data analysis experience on a year-for-year basis.
 Job-Related Knowledge, Skills and Abilities:

 Strong problem-solving, quantitative and analytical abilities.
 Experience with large relational databases and data warehousing, including the ability to query database using SQL or similar language.
 Mastery of business intelligence report writing and visualization tools, such as Epic Analytics, Business Objects (WebI), Power BI, Tableau or similar report writing and visualization tools.
 Advanced Excel spreadsheet skills, including complex functions, formulas and formatting.
 Proven ability to work with and track large amounts of data (millions of records) with accuracy.
 Excellent communication, collaboration and delegation skills.
 Demonstrated ability to communicate information in an easy-to-understand format.


 Motor Vehicle Operator Designation:

 Employees in the position will operate vehicles for an assigned business purpose as a "non-frequent driver."


 Licensure/Certification/Registration::

 To be completed within six months of hire: clinical data model train track (proficiency).
 #LI-90

 Shift hours: 8 a.m. - 5 p.m., Monday-Friday
 '''
    else:
        print('Hi there, I\'m your Versifier, your AI resume assistant. To get started please provide me with the job post for the job you want followed by a the word \"endpost\"' )
        paragraph = []
        while True:
            line = input()
            if line.strip() != 'endpost':
                paragraph.append(line)
            else:
                break
        return '\n'.join(paragraph)

# get skill words from job post
def get_skill_words():
    if not skipping:
        my_message = getJobPost() + "\n Above is a job posting, please extract any relevant skills that you think the employer might be looking for. In particular look for repeat skill words. Each skill word should be a single item in a comma separated list. Look for repeated words but try not to repeat items in your final list. Do not provide anything beyond this list of skills."
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user","content": my_message}]
            )
        reply = response["choices"][0]["message"]["content"]
        return [word.strip() for word in reply.split(',')]

# get skills from user
def get_user_skills():
    if skipping:
        data = '''data analysis
reporting
analytics
forecasting
MS Office Suite
MS Visual Studio
data science
communication
complex data
business operations
relationship building
operational processes
problem-solving
quantitative abilities
analytical abilities
business intelligence
report writing
visualization
Excel
accuracy
collaboration
delegation
communication skills
python
jupyter notebooks
python libraries including numpy pandas and sklearn
R
C++
github workflows
creative
organized
enthusiastic
team-player
'''
        return ", ".join(data.split('\n'))
    else:
        user_skills = []
        rec_skills = [word.strip() for word in reply.split(',')]

        print("Thanks! I've gone through and picked out skills from that job posting I think the employer might be looking for. Let's go over them. Remember to be generous with yourself as many skill words can be vague, but don't be afraid to say you don't have a skill if you don’t have it. Be honest with me, and I’ll make sure to give you the best resume possible")
        for skill in rec_skills:
            print(f"Would you say that you have the skill \"{skill}\"? (y/n)")
            line = input()
            if line.strip().lower() == 'y':
                user_skills.append(skill)

        ready = False
        while not ready:
            print("Ready for the next question? (y/n)")
            line = input()
            if line.strip().lower() == 'y':
                ready = True

        print("Please list any technical skills you have separated by commas. Things like a coding language or software certification")
        line = input()
        user_skills.extend([word.strip() for word in line.split(',')])

        print("Please list any soft skills you have separated by commas. Things like creative, organized, enthusiastic, or team-player")
        line = input()
        user_skills.extend([word.strip() for word in line.split(',')])

        for skill in user_skills:
            if skill == "":
                user_skills.remove(skill)

        return ", ".join(user_skills)
    
questionare = """* Personal info:
    * Name: Silas Raye
    * Email: [silas.raye@gmail.com](mailto:silas.raye@gmail.com)
    * Phone: 352-231-0747
* Education:
    * What is your highest level of education?
        * college
    * What degree(s) have you earned and from which institution(s)?
        * bachelor's degree from the university of florida
    * What was your major or field of study?
        * Data science
    * Did you receive any academic honors or awards?
        * Yes, I made the dean's list, and I received an AI certificate
* Experience:
    * Can you describe any significant projects you have worked on? What was your role?
        * I coded a small video game demo for a puzzle game called Boom + Bust
        * I wrote a paper on using sentiment analysis and news headlines to predict the movements of the dow jones
        * I helped code a resume creation program
    * Did any of these projects have a measurable impact or result in any achievements?
        * The goal was to see if I could code a short but fun video game. The game was pretty good
        * The program was able to get a simulation accuracy of 140% return on investment over one year
        * The program was a success and now has a small user base
    * Did you collaborate with a team or work independently on these projects?
        * Worked independently
        * Worked independently
        * Worked in a small team
    * What is your employment history? Please provide the job titles, companies, and dates of employment.
        * Satchel's Pizza 2018 - present
    * What did you do at your job? What did a typical day look like?
        * Working at satchels pizza taught me dedication and hard work. Over the 3 years I've worked there I have worked as in the following positions:
            * Graphic designer
            * Waiter
            * Cashier
            * Host
            * Soda maker
    * Did you achieve any notable accomplishments or receive any recognition during your employment?
        * I was promoted three times
    * Were there any specific technologies or tools you utilized in your previous roles?
        * Restaurant POS systems
        * Adobe illustrator
        * Adobe photoshop
    * Have you pursued any additional certifications, training, or professional development courses? If yes, please provide details.
        * I'm certified in the adobe and microsoft suites
        * I'm also very proficient in the google suite
    * Have you attended any relevant workshops, conferences, or industry events?
        * I attended a job fair at the university of florida
"""

order = '''Personal info
Education
Technical Skills
Experience
'''

my_message = "Given the following resume information...\n" + questionare + "\nMake a resume using as many of the following skill words as possible through out the resume...\n" + get_user_skills() + "\nThe resume should be in the this order...\n" + order
response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user","content": my_message}]
            )
reply = response["choices"][0]["message"]["content"]
print(reply)