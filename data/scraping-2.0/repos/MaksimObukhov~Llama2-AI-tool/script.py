import os
from datetime import timedelta

import pandas as pd
from langchain.llms import Clarifai
from langchain import PromptTemplate, LLMChain
from langchain.chains import SimpleSequentialChain, SequentialChain

EMP_PATH = "data_hackaton_5.xlsx"

import re


def remove_bracket_content(input_string):
    pattern = r'\[.*?\]'
    result = re.sub(pattern, '', input_string).strip()
    result = re.sub(r'\s+', ' ', result)
    return result
    # return input_string


def skills_retrieve(path):
    employees = pd.read_excel(path)
    skills = list(employees['Skills'])
    skills = set(','.join(skills).split(sep=","))
    return skills


def match_employees(pat, model_id, name, overview, duration, goals, skillset, desired_outcomes):
    clarifai_llm = Clarifai(pat=pat, user_id='meta', app_id='Llama-2', model_id=model_id)
    # clarifai_llm_70 = Clarifai(pat=pat, user_id='meta', app_id='Llama-2', model_id='llama2-70b-chat')
    # clarifai_llm_13 = Clarifai(pat=pat, user_id='meta', app_id='Llama-2', model_id='llama2-13b-chat')

    template = """<<SYSTEM>> You are a project manager and professional in team building. You can build a team and 
    match employees for a specific project the most efficient way. Also you are able to decompose the project into 
    specific sequential steps. Your answer should not include any annotations and abstract. Tell only facts. Behave 
    by the principle "less is more". <</SYSTEM>>

    [OUTPUT STRUCTURE] [EXAMPLE] Summary: This IT-project focuses on building a personalized content recommendation 
    engine for an online streaming platform using Machine Learning and Data Science. The goal is to enhance user 
    engagement, retention, and overall experience by delivering tailored content suggestions. The ultimate aim is to 
    improve viewer satisfaction and drive business success through optimized content recommendations.

    Sequential steps: Data Collection: Gather user interaction data from the online streaming platform, including 
    viewing history, ratings, searches, and clicks. Feature Engineering: Extract relevant features from the collected 
    data to create detailed user profiles, incorporating factors like genre preferences, viewing frequency, 
    and watch history. ML Model Development: Build and train advanced Machine Learning models such as collaborative 
    filtering and content-based filtering to generate accurate content recommendations. Algorithm Evaluation: Assess 
    the performance of various recommendation algorithms, refining their effectiveness through metrics like 
    precision, recall, and user engagement rates. Integration with Platform: Seamlessly integrate the recommendation 
    engine into the streaming platform's interface to provide real-time, personalized content suggestions. A/B 
    Testing: Conduct controlled experiments to measure the impact of the recommendation engine on user engagement and 
    retention. Performance Monitoring: Implement mechanisms to continuously monitor and evaluate the recommendation 
    engine's performance, making adjustments as necessary. Documentation and Delivery: Provide comprehensive 
    documentation, including model code, integration guidelines, and insights from A/B testing. [/EXAMPLE] [/OUTPUT 
    STRUCTURE]

    [TASK] Based on the [PROJECT CONTEXT] below summarize the project into a comprehensive description covering all 
    the categories below and decompose the project into sequential steps by the [OUTPUT STRUCTURE]. [PROJECT CONTEXT] 
    always consists of [Project Name], [Project Overview], [Project Duration], [Project Goals], [Team skill set], 
    [Desired Outcome]. The report must follow the [OUTPUT STRUCTURE]. [/TASK]

    [PROJECT CONTEXT]
    [Project Name]
    {name_input}

    [Project Overview]
    {overview_input}

    [Project Duration]
    {duration_input}

    [Project Goals]
    {goals_input}

    [Team skill set]
    {skillset_input}

    [Desired Outcome]
    {outcome_input}
    [/PROJECT CONTEXT]
    """
    prompt = PromptTemplate(template=template, input_variables=['name_input', 'overview_input', 'duration_input',
                                                                'goals_input', 'skillset_input', 'outcome_input'])
    llm_chain_1 = LLMChain(prompt=prompt, llm=clarifai_llm)

    template_2 = """<<SYSTEM>> You are a project manager and professional in team building. You can build a team and 
    match employees for a specific project the most efficient way. Also you are a professional in understanding what 
    people's roles and skills are required based on the [PROJECT SUMMARY AND SPECIFIC SEQUENTIAL STEPS]. Your answer 
    should not include any annotations and abstract besides the output structure in the following example. <</SYSTEM>>

    [OUTPUT STRUCTURE]
    [STRUCTURE FORMAT]
    [Role]
    [Technical skills]
    [Soft Skills]
    [Tools]
    [/Role]

    [/STRUCTURE FORMAT]
    [EXAMPLE]
    Team Roles, Skills, and Tools for the Project:

    Project Manager
    Technical Skills: Project management methodologies, communication, risk management.
    Soft Skills: Leadership, communication, problem-solving.
    Tools: Project management software (e.g., Jira, Trello), communication tools (e.g., Slack, Microsoft Teams).

    Data Scientist
    Technical Skills: Machine Learning, data analysis, feature engineering, algorithm development.
    Soft Skills: Critical thinking, attention to detail, analytical mindset.
    Tools: Python, Jupyter Notebook, pandas, scikit-learn, TensorFlow or PyTorch.

    Machine Learning Engineer
    Technical Skills: Machine Learning algorithms, model training, evaluation, optimization.
    Soft Skills: Collaboration, teamwork, problem-solving.
    Tools: Python, scikit-learn, TensorFlow or PyTorch, model evaluation metrics.
    [/EXAMPLE]
    [/OUTPUT STRUCTURE]

    [TASK] Define key roles for the following project, for each role define necessary skills, both technical and 
    soft-skills, tools. Write down a list that includes specific role, role’s skills and tools. [/TASK]

    [PROJECT SUMMARY AND SPECIFIC SEQUENTIAL STEPS]
    {agent_1_output}
    [/PROJECT SUMMARY AND SPECIFIC SEQUENTIAL STEPS]
    """
    prompt_2 = PromptTemplate(template=template_2, input_variables=["agent_1_output"])

    llm_chain_2 = LLMChain(prompt=prompt_2, llm=clarifai_llm)
    inputs = {'name_input': name, 'overview_input': overview, 'duration_input': duration, 'goals_input': goals,
              'skillset_input': skillset, 'outcome_input': desired_outcomes}
    output_1 = llm_chain_1.run(inputs)
    print("1 complete")

    output_2 = llm_chain_2.run(output_1)

    print("2 complete")
    # AGENT 3
    employees = pd.read_excel(EMP_PATH)

    template_3 = """<<SYSTEM>> You are a project manager's assistant. Now you are collaborating with the project 
    manager to build a team of professionals for the project. You must use your skills and knowledge to define the 
    main capabilities of employees in the current company. You are able to make a comprehensive summary of each 
    employee based on his working and educational background. You always follow [OUTPUT EXAMPLE] below. <</SYSTEM>>

    [OUTPUT EXAMPLE]
    [Employee Summary] A highly qualified professional with a PhD in Computer Science, 
    specializing in machine learning and statistical analysis. Possesses extensive experience as a Senior Data Scientist. 
    Notable for contributions to various projects including Customer Segmentation and Sales Forecasting. Holds a 
    significant portfolio on GitHub showcasing work in the field. Proficient in utilizing Python and R for data analysis 
    and machine learning purposes.

    [Education]
    PhD in Computer Science

    [Experience]
    Senior Data Scientist, Specialization in machine learning and statistical analysis

    [In-Company Projects]
    Sales Optimization Platform, Market Basket Analysis

    [Skills and Tools]
    Machine Learning, Statistical Analysis, Python, R
    [/OUTPUT EXAMPLE]

    [TASK] Based on the provided [DESCRIPTION] summarise employee portfolio, so that it fully describes the person from 
    the professional point of view. [DESCRIPTION] consists of [Overall employee description] and [Skills]. Tell only 
    facts. Behave by the principle "less is more". Do not make any conclusions, assessments nor expectations by yourself.
    The output must follow the [OUTPUT EXAMPLE].
    [/TASK]

    [DESCRIPTION] 
    [Overall employee description]
    {description} 

    [Skills] 
    {skills}
    [/DESCRIPTION]
    """
    prompt_3 = PromptTemplate(template=template_3, input_variables=["description", "skills"])
    llm_chain_3 = LLMChain(prompt=prompt_3, llm=clarifai_llm)

    emp_summary = {}

    for i in range(employees.shape[0]):
        name, description, skills = employees.iloc[i, 0], employees.iloc[i, 1], employees.iloc[i, 2]
        inputs = {"description": description, "skills": skills}
        emp_summary[name] = llm_chain_3.run(inputs)

    print("3 complete")

    # AGENT 4
    template_4 = """<<SYSTEM>> You are the project manager’s assistant. You know everything about a new project, 
    required roles and skills, also you have summarized information about each employee. Also you are a professional 
    in understanding what person is the best fit for a specific role. You can evaluate a person from a professional 
    point of view. You are able and empowered to critically evaluate each [EMPLOYEE SUMMARY] based on [PROJECT 
    SUMMARY AND SPECIFIC SEQUENTIAL STEPS] and [ROLES DEFINED BY THE PROJECT MANAGER]. Your main function is to 
    provide a comprehensive reasoning if the specific employee matches with the project needs, don't give any 
    assessments or conclusions. Tell only facts. Behave by the principle "less is more". <</SYSTEM>>

    [OUTPUT STRUCTURE] 
    [STRUCTURE FORMAT]
    [Role]
    [Pros]
    [Cons]
    [/Role]

    [/STRUCTURE FORMAT] [EXAMPLE] Data Scientist Pros: PhD in Computer Science, specialization in machine learning 
    and statistical analysis, extensive experience as a Senior Data Scientist, significant GitHub portfolio 
    showcasing relevant work, proficiency in Python and R. Cons: No notable cons based on the provided information.

    DevOps Engineer 
    Pros: None. 
    Cons: No specific DevOps skills mentioned in the provided summary. 
    [/EXAMPLE] 
    [/OUTPUT STRUCTURE] 

    [TASK] Provide a report that includes pros and cons of a person's participation in the project for each role, 
    according to the project needs. Good fit means that a person fits at least one role. Do not draw any conclusions 
    and notes, you have all required data. The report must follow the [OUTPUT STRUCTURE]. [/TASK]

    [PROJECT SUMMARY AND SPECIFIC SEQUENTIAL STEPS]
    {output_1}
    [/PROJECT SUMMARY AND SPECIFIC SEQUENTIAL STEPS]

    [ROLES DEFINED BY THE PROJECT MANAGER]
    {output_2}
    [/ROLES DEFINED BY THE PROJECT MANAGER]

    [EMPLOYEE SUMMARY]
    {output_3}
    [/EMPLOYEE SUMMARY]
    """
    prompt_4 = PromptTemplate(template=template_4, input_variables=["output_1", "output_2", "output_3"])
    llm_chain_4 = LLMChain(prompt=prompt_4, llm=clarifai_llm)
    emp_reasoning = {}
    for key in emp_summary:
        output_3 = emp_summary[key]
        inputs = {"output_1": output_1, "output_2": output_2, "output_3": output_3}
        emp_reasoning[key] = llm_chain_4.run(inputs)
    print("4 complete")

    # AGENT 5
    template_5 = """<<SYSTEM>> You are a decision-making system, responsible for assigning employees to the project. 
    You can make a final conclusion about a person matching a specified role in the project based on [REPORT]. The 
    output should include roles the employee fits the most and justification. You are able and empowered to 
    critically evaluate each employee. Tell only facts. Behave by the principle "less is more". <</SYSTEM>>

    [OUTPUT STRUCTURE]
    [STRUCTURE FORMAT]
    Plain text

    [/STRUCTURE FORMAT] [POSITIVE EXAMPLE] Based on the provided information in the report, the most suitable role for the given 
    employee appears to be Machine Learning Engineer. This conclusion is based on the employee's extensive experience as 
    a Senior Data Scientist, specialization in machine learning and statistical analysis, and proficiency in Python, 
    a key language for machine learning. The employee's notable contributions to projects involving machine learning 
    further support their suitability for this role. No specific cons are mentioned based on the provided information.[/POSITIVE EXAMPLE]
    
    [NEGATIVE EXAMPLE]
    Considering the information presented in the report, the employee doesn't seem to be a suitable candidate for any 
    of the required project roles. While the employee holds a position as a Senior Data Scientist, 
    their specialization in machine learning and statistical analysis, as well as their proficiency in Python, 
    are not adequately aligned with the specific roles needed for the project. Furthermore, their contributions to 
    machine learning projects might not sufficiently qualify them for the roles at hand. It's important to carefully 
    evaluate whether the employee's skills and experience truly match the project's requirements before considering 
    them for a role. [/NEGATIVE EXAMPLE] [/OUTPUT STRUCTURE]

    [TASK] Decide which role is the most suitable for the given employee based on [REPORT] below. If the given employee does 
    not match none of the mentioned roles, clearly describe that the person is not suitable for the project. Use only provided 
    information, keep your answer short and clear. The report must follow the [OUTPUT STRUCTURE].[/TASK]

    [REPORT]
    {output_4}
    [/REPORT]
    """
    prompt_5 = PromptTemplate(template=template_5, input_variables=["output_4"])
    llm_chain_5 = LLMChain(prompt=prompt_5, llm=clarifai_llm)
    emp_evaluation = {}
    for key in emp_reasoning:
        output_4 = emp_summary[key]
        inputs = {"output_4": output_4}
        emp_evaluation[key] = remove_bracket_content(llm_chain_5.run(inputs))
    print("5 complete")

    summary_pd = pd.DataFrame([emp_summary]).transpose()
    reasoning_pd = pd.DataFrame([emp_reasoning]).transpose()
    evaluation_pd = pd.DataFrame([emp_evaluation]).transpose()

    summary_pd.columns = ['summary']
    reasoning_pd.columns = ['reasoning']
    evaluation_pd.columns = ['evaluation']

    return summary_pd, reasoning_pd, evaluation_pd


def sentiment(pat, evaluation):
    ######################################################################################################
    # In this section, we set the user authentication, user and app ID, model details, and the URL of
    # the text we want as an input. Change these strings to run your own example.
    ######################################################################################################

    # Specify the correct user_id/app_id pairings
    # Since you're making inferences outside your app's scope
    USER_ID = 'erfan'
    APP_ID = 'text-classification'
    # Change these to whatever model and text URL you want to use
    MODEL_ID = 'sentiment-analysis-distilbert-english'
    MODEL_VERSION_ID = 'c0b09e606db94d9bae7eb40c626192fc'
    TEXT = evaluation

    ############################################################################
    # YOU DO NOT NEED TO CHANGE ANYTHING BELOW THIS LINE TO RUN THIS EXAMPLE
    ############################################################################

    from clarifai_grpc.channel.clarifai_channel import ClarifaiChannel
    from clarifai_grpc.grpc.api import resources_pb2, service_pb2, service_pb2_grpc
    from clarifai_grpc.grpc.api.status import status_code_pb2

    channel = ClarifaiChannel.get_grpc_channel()
    stub = service_pb2_grpc.V2Stub(channel)

    metadata = (('authorization', 'Key ' + pat),)

    userDataObject = resources_pb2.UserAppIDSet(user_id=USER_ID, app_id=APP_ID)

    post_model_outputs_response = stub.PostModelOutputs(
        service_pb2.PostModelOutputsRequest(
            user_app_id=userDataObject,
            # The userDataObject is created in the overview and is required when using a PAT
            model_id=MODEL_ID,
            version_id=MODEL_VERSION_ID,  # This is optional. Defaults to the latest model version
            inputs=[
                resources_pb2.Input(
                    data=resources_pb2.Data(
                        text=resources_pb2.Text(
                            raw=TEXT
                        )
                    )
                )
            ]
        ),
        metadata=metadata
    )
    if post_model_outputs_response.status.code != status_code_pb2.SUCCESS:
        print(post_model_outputs_response.status)
        raise Exception("Post model outputs failed, status: " + post_model_outputs_response.status.description)

    # Since we have one input, one output will exist here
    output = post_model_outputs_response.outputs[0]

    ret = {}
    for concept in output.data.concepts:
        ret[concept.name] = concept.value
    # Uncomment this line to print the full Response JSON
    # print(output)
    return ret


# %%
# pr_name = 'Personalized Content Recommendation Engine for Online Streaming Platform'
# pr_overview = ('In this project description, we present the details of an IT-company project focused on leveraging '
#                'Machine Learning (ML) and Data Science to develop an innovative recommendation engine for an online '
#                'streaming platform. The project aims to enhance user experience, increase engagement, and optimize '
#                'content recommendations based on individual preferences.')
# pr_duration = '8 months'
# pr_goals = '''The primary objective of this IT-company project is to design, build, and deploy a sophisticated
# recommendation engine that utilizes Machine Learning and Data Science techniques to: Enhance User Engagement: Provide
# users with tailored content recommendations that match their viewing history, preferences, and behaviors. Improve
# Content Discovery: Facilitate the discovery of new content by suggesting relevant movies, TV shows, and genres that
# align with users interests. Optimize Viewing Experience: Increase user satisfaction by reducing the time spent
# searching for content and improving the relevance of recommendations. Boost Platform Retention: Encourage users to
# spend more time on the platform by consistently delivering compelling and personalized content. Drive Business Value:
# Translate improved user engagement into higher viewer retention rates, increased subscriptions, and enhanced brand
# loyalty.'''
# pr_skillset = '''Skills: Machine Learning, Data Science, Software Development, UI/UX Design, Project Management,
# Data Analysis, Backend Development Tools: Machine Learning Libraries (e.g., TensorFlow, PyTorch, scikit-learn),
# Data Analysis Tools (e.g., Pandas, NumPy), Programming Languages (e.g., Python, Java), Version Control (e.g., Git),
# Project Management Software, UI/UX Design Tools (e.g., Adobe XD, Sketch), Backend Frameworks (e.g., Django, Flask)'''
# pr_outcome = '''The successful completion of this project will result in a cutting-edge recommendation engine
# integrated into the online streaming platform. The engine will deliver accurate and personalized content suggestions
# to users, ultimately enhancing their viewing experience, increasing engagement, and contributing to the platform's
# business success.'''
#
# summary, reasoning, evaluation = match_employees(model_id="llama2-70b-chat", pat='f3e3f98c5c0e4e2998390ae4ef11e18f',
#                                                  name=pr_name,
#                                                  overview=pr_overview, duration=pr_duration, skillset=pr_skillset,
#                                                  goals=pr_goals, desired_outcomes=pr_outcome)

# %%
# positive = []
# value = []
# for eval in evaluation['evaluation']:
#     try:
#         prob = sentiment(pat='f3e3f98c5c0e4e2998390ae4ef11e18f', evaluation=str(eval))['POSITIVE']
#         value.append(prob)
#         positive.append(prob > 0.5)
#     except:
#         value.append(0)
#         positive.append(False)
# evaluation['value'] = value
# evaluation['positive'] = positive
# evaluation = evaluation.sort_values(by='value', ascending=False)
