# streamlit_app.py

import os
import tempfile
from datetime import datetime

import pypandoc
from langchain.callbacks import get_openai_callback
from llm_utils.database import get_db_connection
from llm_utils.streamlit_common import apply_uab_font, hide_streamlit_branding
from llm_utils.text_format import convert_markdown_docx

import IRB_Assistant.prompts as irb_assistant_prompts
import IRB_Assistant_config.app_config as irb_assistant_app_config
import IRB_Assistant_config.boilerplate as IRB_boilerplate
import IRB_Assistant_config.config as irb_Assistant_config
import streamlit as st
from IRB_Assistant.generate import (
    get_irb_assistant_response,
    get_variable_assistant_response,
)


def on_accept_click():
    st.session_state["accepted_terms"] = True


def show_irb_page(template_location=irb_Assistant_config.TEMPLATE):
    # openai
    os.environ["OPENAI_API_KEY"] = irb_assistant_app_config.OPENAI_API_KEY

    # page metadata
    st.set_page_config(
        page_title="IRB Assistant",
        page_icon="⚖️",
    )
    # hide streamlit branding
    hide_streamlit_branding()

    # apply uab font
    apply_uab_font()

    st.title("⚖️ IRB Assistant 🤖")
    st.markdown(
        """
    **Get help filling out the department protocol form for IRB submission.**

    Brought to you by the Anesthesiology Research Support, Informatics, and Data Science teams.

    _Not approved for use with PHI._

    All submissions are recorded for potential review by departmental and health system personnel.

    ---
    """
    )

    if st.session_state.setdefault("accepted_terms", False) is False:
        st.write(
            """# Purpose
This tool uses AI to create a ***starting point*** for your department protocol form.

It should make your work faster, but it can't do *all* the work for you.

# Keep in Mind
- ***You*** are in charge of the content in the documents this tool creates. Always check and fix your drafts.
- The tool makes a basic version of a protocol from the information you provide. Some details may be left out, and others may not fit your work.
- Remember, skilled professionals will be checking your work. If you submit without changes, it ***will*** be rejected.
- ***Don't forget***, this tool doesn't have a "save" button. Your work is saved only in the word document you can download after filling out the form. If you leave or refresh the page, your work will be lost.
                 """
        )
        if st.button(label="I understand."):
            on_accept_click()
            st.experimental_rerun()
    else:
        # page content
        st.write("## IRB Application Information")
        st.write("### Research question (ideally in PICOS format)")
        research_question = st.text_area(
            "PICOS Format: Patient, problem, or population (P), Investigated condition (I), Comparison condition (C), Outcome (O), Study type (S)",
            "Retrospectively, In adult patients undergoing surgery, how does the use of regional anesthesia techniques compare to general anesthesia in terms of postoperative pain management?",
        )

        st.write("### Relevant literature/citations")
        references = st.text_area(
            "Include any references. A protocol form submission requires at least ***8*** references from the literature. If you have not yet investigated the literature, consider using the Literature & Novelty generative AI tool.",
            "None",
        )

        st.write("### Inclusion criteria (i.e., patient population)")
        inclusion_criteria = st.text_area(
            "A list of specific conditions or characteristics that qualify potential participants from taking part in the study.",
            "Patients who underwent surgery and received either regional or general anesthesia.",
        )

        exclusion_criteria_options = [
            "<18 years (or bodyweight <50kg if age unknown)",
            "prisoners",
            "pregnant women",
            "non-English speaking",
            "patients enrolled in concurrent ongoing interventional trial",
            "students of UAB",
            "employees of UAB",
            "targeting specific populations",
        ]

        st.write("### Exclusion criteria")

        # Using a default_value parameter set to all the options,
        # the multiselect will be pre-filled with all options checked.
        exclusion_criteria_multiselect = st.multiselect(
            "Applicable conditions or characteristics that disqualify potential participants from taking part in the study. We've already entered the usual ones. **Most researchers will *not* need to change what's in this box.** ",
            exclusion_criteria_options,
            default=exclusion_criteria_options,
        )

        exclusion_criteria_other = st.text_input(
            "If there are other conditions or characteristics that disqualify potential participants not listed above, please specify them here."
        )

        # Then you can create a single variable combining the selected options and the typed input
        exclusion_criteria = exclusion_criteria_multiselect
        if exclusion_criteria_other:
            exclusion_criteria.append(exclusion_criteria_other)

        # Convert the list into a string, with each element separated by a comma
        exclusion_criteria = ", ".join(exclusion_criteria)

        study_design_options = [
            "Retrospective research",
            "Quality improvement/assurance",
            "Cross-sectional study or survey",
            "Prospective – observational only",
            "Prospective – interventional",
            "Clinical trial",
            "Sponsor initiated study",
            "Other",
        ]

        st.write("### Study design")
        study_design = st.radio("Please select the design of the study", study_design_options)

        # TODO Move study-specific notes to config
        # Display instructions based on selected option
        if study_design == "Quality improvement/assurance":
            study_specific_note = """
    **Note: Not for Generalized Knowledge – the results cannot be used for research purposes or peer-reviewed publications, etc.**
    In your final draft, be sure to address how you will:

    1. Identify the process or outcome to be improved.
    2. Establish the performance standards or targets.
    3. Collect and analyze data on the current performance. (Can it be de-identified?)
    4. Implement an intervention designed to improve performance.
    5. Monitor and measure the impact of the intervention on performance.
    6. Make adjustments and refinements as necessary.
            """

        elif study_design == "Retrospective research":
            study_specific_note = """
    **Review of Data already collected.**    
    In your final draft, be sure to address how you will:

    1. Define the research question.
    3. Define the variables and outcomes to be extracted. (Can they be de-identified?)
    2. Identify the existing data set(s) to be used - this can include both a "treatment" group and a "comparison/cohort" group.  How will the two groups be matched? - age? ASA rating? sex? other? - this will lead to the next step of defining variables.  Will the two groups be the same size or will there be a 2:1 matching, 3:1?
    4. Extract the data.
    5. Clean and prepare the data for analysis.
    6. Conduct the data analysis.
            """

        elif study_design == "Cross-sectional study or survey":
            study_specific_note = """
    In your final draft, be sure to address how you will:

    1. Identify the target population and sampling method - this can include both a "treatment" group and a "comparison/cohort" group.  How will the two groups be matched? - age? ASA rating? sex? other? - this will lead to the next step of defining variables.  Will the two groups be the same size or will there be a 2:1 matching, 3:1?
    2. Develop the survey instrument.
    3. Acquire informed consent. 
    4. Administer the survey.
    5. Collect the completed surveys.
    6. Clean and prepare the data for analysis. (Can it be de-identified?)
    7. Analyze the data.
            """

        elif study_design == "Prospective – observational only":
            study_specific_note = """
    Using future data routinely collected for non-research purposes.
    In your final draft, be sure to address how you will:

    1. Define the research question and objectives.
    2. Define the variables and outcomes to be collected.
    3. Identify the source of the routine data.
    4. Acquire informed consent. 
    5. Collect the data at the predetermined time points.
    6. Clean and prepare the data for analysis. (Can it be de-identified?)
    7. Analyze the data.
            """

        elif study_design == "Prospective – interventional":
            study_specific_note = """
    In your final draft, be sure to address how you will:

    1. Define the research question and objectives.
    2. Develop the intervention.
    3. Identify the target population and sampling method.
    4. Acquire informed consent. 
    5. If drugs or devices involved, will there need to be an IND or IDE; what is the source of the drugs/devices?
    6. Will patients incur any costs related to the research?
    7. Conduct baseline assessments.
    8. Implement the intervention.
    9. Conduct follow-up assessments at predetermined time points.
    10. Analyze the data. (Can it be de-identified?)
            """

        elif study_design == "Clinical trial":
            study_specific_note = """
    Be sure to discuss if your study is randomized, non-randomized, blinded, unblinded.
    In your final draft, be sure to address how you will:

    1. Define the research question and objectives.
    2. Develop the intervention.
    3. Identify the target population and sampling method. (Have you determined how many patients may be available/eligible for study?)
    4. Acquire informed consent. 
    5. Randomize participants to the intervention or control group (If applicable, what is the randomization method and when is it performed?).
    6. If drugs or devices involved, will there need to be an IND or IDE; what is the source of the drugs/devices?
    7. Will patients incur any costs related to the research? Who holds the "key" to double-blinded information?
    8. Conduct baseline assessments.
    9. Administer the intervention.
    10. Conduct follow-up assessments at predetermined time points.
    11. Analyze the data.
            """

        elif study_design == "Sponsor initiated study":
            study_specific_note = """
    Industry sponsored or multi-site study where UAB is a participating site.

    *If this is an industry-initiated study, you may attach the sponsor's protocol instead of recreating it in the protocol form.*
    Otherwise, in your final draft, be sure to address how you will:

    1. Collaborate with the sponsor to understand the study protocol.
    2. Train staff on the study protocol.
    3. Identify the target population and sampling method.
    4. Acquire informed consent. 
    4. Conduct baseline assessments.
    5. Administer the intervention or conduct the study procedures.
    6. Collect data according to the study protocol.
    7. Transfer data to the sponsor or central data management site.
            """

        # Check if the 'Other' option is selected, if so, prompt the user to input the study design
        if study_design == "Other":
            study_design = st.text_input("Please specify the study design here")
            study_specific_note = "For an 'Other' study design, in your final draft, be sure to clearly articulate how you will execute the study and the key steps involved."

        st.write(study_specific_note)

        st.write(
            "If you have already decided these points, include those details in the optional box at the bottom of this form."
        )

        st.write("### Time window of study")
        time_window = st.text_input(
            "The expected date range the study will take place over OR the historical date range for which data will be extracted",
            "2016-2020",
        )

        st.write("### Other details")
        other_details = st.text_area(
            "Are there any other aspects of your study that you have already decided? Examples: number of patients involved/enrolled; research support or IT resources; specifics of outcomes, endpoints, or logistics.",
            "None",
        )

        if st.button("Submit"):
            st.write(
                """While you wait on your draft, we have some important reminders.   
                    1. This tool provides a draft for you to edit.                
                    2. Depending on the type of study you selected, there are specific elements you will need to expand and fill in.     
                    3. Generative AIs have a stochastic (random) nature. Sometimes the output you get might be nonsense. Sometimes the output may look right but be wrong.     
                    4. Always use your best judgment.     
                    
                    """
            )
            with st.spinner("Drafting. This may take a few minutes..."):
                submit_time = datetime.now()
                with get_openai_callback() as response_meta:
                    result = get_irb_assistant_response(
                        hypothesis=research_question,
                        inclusion=inclusion_criteria,
                        time_window=time_window,
                        exclusion=exclusion_criteria,
                        design=study_design,
                        details=other_details,
                        chat=irb_Assistant_config.CHAT,
                        chat_prompt=irb_assistant_prompts.irb_chat_prompt,
                    )
                    generated_text = result.content
                    variable_list_response = get_variable_assistant_response(
                        generated_protocol=generated_text,
                        chat=irb_Assistant_config.CHAT,
                        chat_prompt=irb_assistant_prompts.variable_chat_prompt,
                    )
                response_time = datetime.now()
                output_text = ""
                output_text = (
                    IRB_boilerplate.CONTRACT
                    + study_specific_note
                    + "\n\n- [ ] **Unique to YOUR study:** Confirm the following list of variables to be collected or extracted is ***exhaustive***:\n\n"
                    + variable_list_response.content
                    + "\n\n------------------------------------------------------------------------------------------------------------------------\n\n"
                    + IRB_boilerplate.FRONT_MATTER
                    + "\n\n"
                    + "**Study Type:**"
                    + study_design
                    + "\n\n"
                    + generated_text
                    + "\n\n# References\n\n"
                    + references
                    + "\n\n"
                    + IRB_boilerplate.BACK_MATTER
                )

                docx_data = convert_markdown_docx(output_text, template_location)

            if docx_data:
                st.balloons()
                st.write("Note that once you hit download, this form will reset.")

                st.download_button(
                    label="Download Draft",
                    data=docx_data,
                    file_name="DRAFT_protocol_form.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",  # correct MIME type for docx
                )

            try:
                with get_db_connection(
                    db_server=irb_assistant_app_config.DB_SERVER,
                    db_name=irb_assistant_app_config.DB_NAME,
                    db_user=irb_assistant_app_config.DB_USER,
                    db_password=irb_assistant_app_config.DB_PASSWORD,
                ) as conn:
                    # tempting to move this into llm_utils, but the query will be unique to each app.
                    cursor = conn.cursor()
                    query = """
                            INSERT INTO [dbo].[irb_assistant] (
                                research_question, 
                                inclusion_criteria, 
                                time_window, 
                                exclusion_criteria, 
                                study_design, 
                                other_details, 
                                model_response, 
                                input_time, 
                                response_time,
                                total_cost
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """

                    cursor.execute(
                        query,
                        (
                            research_question,
                            inclusion_criteria,
                            time_window,
                            exclusion_criteria,
                            study_design,
                            other_details,
                            variable_list_response.content + "\n\n" + generated_text,
                            submit_time,
                            response_time,
                            response_meta.total_cost,
                        ),
                    )

                st.success(
                    "To comply with a Health System Information Security request, submissions are recorded for potential review."
                )
            except Exception as e:
                st.error(
                    "Something went wrong, and your submission was not recorded for review. Give the following message when asking for help."
                )
                st.error(e)


if __name__ == "__main__":
    show_irb_page()
