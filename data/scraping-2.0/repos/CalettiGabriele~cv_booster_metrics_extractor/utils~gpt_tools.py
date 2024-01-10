import openai

def get_promopt(prompts, prompt):
    for key, value in prompts.items():
        if key == prompt:
            return value["prompt"], value["model"], value["temperature"]

def gpt_request(prompt, context, model='gpt-3.5-turbo', temperature=0, *args):
        return {
            "infos": ["Gabriele Caletti", "Genoa, Italy", "+39 320 309 0795", "calettigabriele@gmail.com"],
            "education": ["University of Genoa", "Expected Graduation: October 2023 International MSc Data Science and Artificial Intelligence in English", "Grade Point Average: 29,2/30", "University of Genoa", "Graduation: July 2021 BSc Computer Science"],
            "jobs": ["Freelance Full Stack Developer 2021-2023", "In my most recent position, I spearheaded website development and online advertising campaigns for the renowned music band, The Bustermoon. Through my efforts, their online presence experienced a significant boost, leading to a remarkable doubling of their merchandise sales."],
            "skills": ["Problem Solving", "Communication", "Critical Thinking", "Time Management", "Decision Making", "PROGRAMMING LANGUAGES: Python - Proficient, C++ - Intermediate, C - Intermediate, Java - Intermediate, JavaScript - Basic, Go - Basic", "OTHER SKILLS: MySQL, MongoDB, HTML, CSS, Git, GitHub, Jupiter NoteBook, Latex, Docker, Excel, Tableau"],
            "other": ["LANGUAGES: Italian - Native, English", "RELEVANT PROJECTS: Falcon - Python scikit-learn like library to solve approximate kernel ridge regression for large scale computing, Shor’s Algorithm Implementation - RSA protocol killer quantum algorithm, in qiskit.", "RELEVANT COURSES: Data Visualization - Throughout the course, I gained hands-on experience in designing interactive graphs while simultaneously acquiring knowledge on selecting the most suitable graph type to effectively convey specific information, Advanced Machine Learning - I learned how to build machine learning models, useful to support decisions making. I realized from scratch many regression and classification models, High Performance Computing - During the course, I utilized MPI, OpenMP, and CUDA to significantly minimize the time required for executing complex computational tasks. Through hands-on experience, I mastered the implementation of these cutting-edge technologies to optimize performance.", "LINKS: Linkedin Profile, MSc Thesis, BSc Thesis, MSc Data Science, passionate about AI looking for a role in tech"]
            }