import os, uuid
import json
import streamlit as st
import pandas as pd
from loguru import logger

from ai_analysis.prompts.screening import screening_instructions, cv_analysis_instructions_screening
from ai_analysis.prompts.tech import tech_instructions, cv_analysis_instructions_tech
from ai_analysis.prompts.default import job_analysis_default_instructions, cv_analysis_default_instructions
from ai_analysis.openai_tools import openai_response_with_validation
from utils.st_sessionstate import st_getenv, st_apikey
from data.mockdata import JOB_DESC, CV


def analyze_job(job_desc, instructions=job_analysis_default_instructions):
    """
    This function initializes chatbot
    """
    instructions = f"TASK:\n{instructions}\n\nJOB DESCRIPTION:\n{job_desc}"
    prompt_task = f"Here is the detailed analysis of the main skills we are looking for:\n"
    messages = [{"role": "system", "content": instructions},
                {"role": "assistant", "content": prompt_task}, ]
    return messages


def analyze_cv(cv, job=None, instructions=cv_analysis_default_instructions):
    """
    This function initializes chatbot
    """
    job = "" if job is None else job
    instructions = f"TASK:\n{instructions}\n\nCV:\n{cv}\n\nJOB DESCRIPTION:\n{job}"
    prompt_task = f"Here is the detailed analysis of the candidate's CV:\n"
    messages = [{"role": "system", "content": instructions},
                {"role": "assistant", "content": prompt_task}, ]
    return messages


def create_interview_plan(job_data, cv_data, instructions=screening_instructions):
    """
    This function creates interview plan
    :return: None
    """
    cv_dict = json.loads(cv_data)
    job_dict = json.loads(job_data)
    if instructions == tech_instructions:
        try:
            role = job_dict['position_title']
            seniority = job_dict['seniority_level']
            tasks = job_dict['tasks']
            skills = job_dict['must_have_skills']
            instructions = instructions\
                .replace('{role}', role)\
                .replace('{seniority}', seniority)\
                .replace('{tasks}', f"{tasks}")\
                .replace('{skills}', f"{skills}")
        except Exception as e:
            logger.error(e)

    instructions = instructions.replace('{job}', job_data).replace('{cv}', cv_data)
    instructions = f"TASK:\n{instructions}\n"
    prompt_task = f"Here is the detailed plan of the screening interview:\n"
    messages = [{"role": "system", "content": instructions},
                {"role": "assistant", "content": prompt_task}, ]
    return messages


def download_results(results_id):
    data_to_download = {}
    for k, v in st.session_state['user_template'].items():
        # print(f"\033[92m{k}\033[0m", v)
        if k in ['job_desc', 'cv', 'plan_id']:
            data_to_download[k] = v
        else:
            data_to_download[k] = json.loads(v)
    st.download_button("Download JSON with results",
                       json.dumps(data_to_download, indent=4),
                       f"interview_plan_{results_id}.json")


def st_json(json_data, container_name="JSON", download_button=False):
    try:
        json_data = json.loads(json_data)

        with st.expander(container_name, expanded=False):
            st.json(json_data)

            if download_button:
                st.download_button("Download JSON",
                                   json.dumps(json_data),
                                   f"{container_name}.json")
    except Exception as e:
        logger.error(e)
        with st.expander("Error"):
            st.markdown(f":red[{e}]")
            st.write(json_data)


def save_plan_and_create_id(json_data, plan_type="screening"):
    """
    This function saves interview plan to the database and creates an ID
    :param json_data: JSON data
    :return: None
    """
    db_data = []
    id = uuid.uuid4()
    for k, v in json_data.items():
        for idx, question in enumerate(v):
            db_data.append({
                "id": f"{id}-{k}-{idx}",
                "plan_id": f"{id}",
                "question": question,
                "topic": k,
                "plan_type": plan_type,
                "comment": ""})
    if os.path.exists("./db/plans.csv"):
        db = pd.read_csv("./db/plans.csv")
        # db = db.append(db_data, ignore_index=True)
        # concatenate:
        db = pd.concat([db, pd.DataFrame(data=db_data, columns=db_data[0].keys())], ignore_index=True)
    else:
        db = pd.DataFrame(data=db_data, columns=db_data[0].keys())
    db.to_csv("./db/plans.csv", index=False)
    return str(id)


def main(admin=None):
    """
    This function is a main program function
    :return: None
    """
    st_apikey()
    api_key = st_getenv("api_key")
    st.session_state['user_template'] = {}
    with st.form("Job Description"):
        interview_type = st.selectbox("Interview Type", ["Screening", "Technical"])
        if interview_type == "Screening":
            instructions = screening_instructions
            cv_instructions = cv_analysis_instructions_screening
        elif interview_type == "Technical":
            instructions = tech_instructions
            cv_instructions = cv_analysis_instructions_tech

        col1, col2 = st.columns(2)
        with col1:
            st.info("Paste a Job Description below")
            job_desc = st.text_area("Job", label_visibility="collapsed", value=JOB_DESC, height=600)
        with col2:
            st.info("Paste a CV below")
            cv = st.text_area("CV", label_visibility="collapsed", value=CV, height=600)

        submitted = st.form_submit_button("Update")

    if submitted:
        st.markdown("#### ☑️ Job Description and CV submitted")
        st.markdown("---")
        st.markdown("#### 🔬 Analyzing Job Description and CV...")

        with st.spinner("Analyzing job description..."):
            messages = analyze_job(job_desc)
            job_analysis = openai_response_with_validation(messages, api_key)
            st.session_state['user_template']['job_desc'] = job_desc
            st.session_state['user_template']['job_analysis'] = job_analysis
            logger.info("✅ Job Description analyzed")
            st.markdown("* ✅ Job Description analyzed")

        with st.spinner("Analyzing CV..."):
            messages = analyze_cv(cv, job_desc, cv_instructions)
            cv_analysis = openai_response_with_validation(messages, api_key)
            st.session_state['user_template']['cv'] = cv
            st.session_state['user_template']['cv_analysis'] = cv_analysis
            logger.info("✅ CV analyzed")
            st.markdown("* ✅ CV analyzed")

        with st.spinner("Creating interview plan..."):
            messages = create_interview_plan(job_analysis, cv_analysis, instructions)
            plan = openai_response_with_validation(messages, api_key)
            st.session_state['user_template']['plan'] = plan
            logger.info("✅ Interview plan created")
            st.markdown("* ✅ Interview plan created")

            # Save plan to the database
            try:
                plan = json.loads(plan)
                plan_id = save_plan_and_create_id(plan)
                st.session_state['user_template']['plan_id'] = plan_id
                logger.info("✅ Interview plan saved to the database")
                st.markdown("* ✅ Interview plan saved to the database")
            except Exception as e:
                logger.error(e)

    if "user_template" in st.session_state:
        st.markdown("---")
        st.markdown("#### 📝 Interview Plan")
        try:
            # Load saved data
            job_analysis = st.session_state['user_template']['job_analysis']
            cv_analysis = st.session_state['user_template']['cv_analysis']
            plan = st.session_state['user_template']['plan']
            plan_id = st.session_state['user_template']['plan_id']

            # Display saved data
            st_json(job_analysis, "Job Analysis")
            st_json(cv_analysis, "CV Analysis")
            st_json(plan, "Interview Plan")

            plan = json.loads(plan)
            st.markdown("---")
            st.markdown(f"#### :blue[**Interview Plan:**]")
            for k, v in plan.items():
                st.markdown(f"💡 **{k}:**")
                for idx, question in enumerate(v):
                    if interview_type == "Technical":
                        st.markdown(f"**Q {idx}:** {question['Q']}")
                    elif interview_type == "Screening":
                        st.markdown(f"**Q {idx}:** {question}")

            st.markdown(f"#### 📝 :blue[**Interview Plan ID:**] **{plan_id}**")
            download_results(plan_id)

        except Exception as e:
            st.info("Please submit Job Description and CV to create an interview plan")
            logger.error(e)
            # with st.expander("Error"):
            #     st.write(e)


if __name__ == "__main__":
    main(admin=False)
