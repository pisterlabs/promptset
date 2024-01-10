
import streamlit as st
import openai
import os  
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationEntityMemory
from langchain.chains.conversation.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
import PyPDF2
import random 
import itertools
import streamlit as st
from io import StringIO
import time
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.retrievers import SVMRetriever
from langchain.chains import QAGenerationChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
import sklearn
from sklearn import svm
import getpass
st.set_page_config(page_title='Medimate Assistant', layout = 'centered', page_icon = ':stethoscope:', initial_sidebar_state = 'auto')


def fetch_api_key():
    api_key = None
    
    try:
        # Attempt to retrieve the API key as a secret
        api_key = st.secrets["OPENAI_API_KEY"]
        os.environ["OPENAI_API_KEY"] = api_key
    except KeyError:
        
        try:
            api_key = os.environ["OPENAI_API_KEY"]
            # If the API key is already set, don't prompt for it again
            return api_key
        except KeyError:        
            # If the secret is not found, prompt the user for their API key
            st.warning("Oh, dear friend of mine! It seems your API key has gone astray, hiding in the shadows. Pray, reveal it to me!")
            api_key = getpass.getpass("Please, whisper your API key into my ears: ")
            os.environ["OPENAI_API_KEY"] = api_key
            # Save the API key as a secret
            # st.secrets["my_api_key"] = api_key
            return api_key
    
    return api_key


def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == st.secrets["password"]:
            st.session_state["password_correct"] = True
            # del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.write("*Please contact David Liebovitz, MD if you need an updated password for access.*")
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct.
        return True

if check_password():

    if 'history' not in st.session_state:
                st.session_state.history = []

    if 'output_history' not in st.session_state:
                st.session_state.output_history = []
                
    if 'mcq_history' not in st.session_state:
                st.session_state.mcq_history = []
                
    if 'openai_api_key' not in st.session_state:
        st.session_state.openai_api_key = ""
        
    if 'model' not in st.session_state:
        st.session_state.model = "gpt-3.5-turbo"
    

    st.title("Medimate Assistant")
    st.write("ALPHA version 0.3")
    os.environ['OPENAI_API_KEY'] = fetch_api_key()

    disclaimer = """**Disclaimer:** This is a tool to assist education regarding artificial intelligence. Your use of this tool accepts the following:   
1. This tool does not generate validated medical content. \n 
2. This tool is not a real doctor. \n    
3. You will not take any medical action based on the output of this tool. \n   
    """
    with st.expander('About Medimate - Important Disclaimer'):
        st.write("Author: David Liebovitz, MD, Northwestern University")
        st.info(disclaimer)
        st.session_state.model = st.radio("Select model - leave default for now", ("gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-4"), index=0)
        st.write("Last updated 7/23/23")

    def set_prefix():
        if prefix_context == "Master Clinician Explaining to Junior Clinician":
            prefix = """You are receiving a question from a medical professional. Respond as a master clinician reasoning step by step to a junior clinician who is not quite at your level yet. 
                    Ensure you apply the latest available evidence-based guidelines 
                    and high-impact and up to date medical literature when answering the question. Critical step: Perform a 
                    final step by step review of your preliminary response to ensure it is factual and complete. Then 
                    finalize your response. """
            sample_question = "What is the best treatment for a patient with a blood clot in their leg?"
            sample_answer = """Certainly, I'd be happy to help clarify this for you. The condition you're referring to is called deep vein thrombosis (DVT), which involves a blood clot in the deep veins, most often in the leg.

Let's go through the standard treatment steps:

1. **Anticoagulation therapy:** The primary treatment for DVT is anticoagulation, or blood-thinning medication. This prevents the clot from growing and prevents new clots from forming. Initial treatment is typically with low molecular weight heparin (LMWH), fondaparinux, or direct oral anticoagulants (DOACs) like rivaroxaban, apixaban, edoxaban, or dabigatran. The choice of anticoagulant may depend on patient characteristics, such as renal function, risk of bleeding, and patient preference.

2. **Duration of therapy:** The recommended duration of anticoagulation depends on the circumstances. For a provoked DVT (one that occurred due to a temporary risk factor like surgery), 3 months is usually sufficient. For unprovoked DVT or ongoing risk factors, longer treatment is usually recommended. This decision should be made based on the balance of the patient's risk of recurrence if anticoagulation is stopped, and the risk of bleeding if it is continued.

3. **Compression stockings:** While not universally agreed upon, some guidelines suggest the use of graduated compression stockings to reduce the risk of post-thrombotic syndrome, which involves long-term swelling and pain in the leg where the DVT was.

4. **Follow-up care:** Patients should be regularly monitored for the effectiveness of treatment, side effects of anticoagulation, and signs of post-thrombotic syndrome. 

5. **Inferior vena cava (IVC) filters:** These are used in patients who can't take anticoagulation or in those in whom it has failed. However, they do not treat the clot and have their own complications, so they're generally used as a last resort.

6. **Thrombolysis or thrombectomy:** In severe cases (such as in phlegmasia cerulea dolens), these procedures to directly remove or dissolve the clot might be considered. However, they are associated with a higher risk of bleeding and are therefore not typically first-line treatments.

**Review of Response:**

This response accurately describes the step-by-step process for treating a patient with a blood clot in their leg, or DVT. It covers the primary treatment method of anticoagulation, with a description of the different types of anticoagulants and factors that might influence their selection. The description of the duration of therapy is accurate, reflecting the differing needs of patients based on their risk factors. The potential use of compression stockings, follow-up care, IVC filters, and thrombolysis or thrombectomy are also correctly described. The response is in line with current evidence-based guidelines and high-impact medical literature.
        """
            temperature = 0.0
        elif prefix_context == "Senior Researcher Explaining to Junior Researcher":
            prefix = """You are receiving a question from a medical professional. Respond as a senior experienced researcher explaining step by step to a junior researcher who is not quite at your level yet. 
                    Ensure you apply the latest available scientific research
                    from high-impact scientific and medical literature when answering the question. Critical: perform a 
                    final step by step review of your preliminary response to ensure it is factual and complete when
                    finalizing your response. """
            sample_question = "Describe the current understanding of what underlies the process of aging."
            sample_answer = """Absolutely, let's delve into the complex topic of aging. 

1. **Genetic Factors:** Research has uncovered several genes associated with longevity, suggesting that aging has a genetic component. For example, genes involved in DNA repair, telomere maintenance, and stress responses have been linked to lifespan. Genomic instability, including DNA damage, is one of the hallmarks of aging.

2. **Telomeres and Cellular Senescence:** Each time a cell divides, the telomeres at the ends of chromosomes shorten. Once telomeres reach a critical length, the cell enters a state of senescence and stops dividing, which is believed to contribute to aging and age-related diseases.

3. **Mitochondrial Dysfunction:** Mitochondria produce reactive oxygen species (ROS) as a byproduct of energy production. Over time, accumulated ROS can cause mitochondrial and cellular damage, contributing to aging. This ties into the free radical theory of aging.

4. **Protein Homeostasis (Proteostasis):** Aging is associated with a decline in the ability of cells to correctly fold proteins and to clear away misfolded proteins. This can lead to the accumulation of protein aggregates, which can disrupt cellular function.

5. **Epigenetic Alterations:** Epigenetic changes, such as DNA methylation, histone modification, and changes in non-coding RNAs, have been associated with aging. These changes can affect gene expression without altering the underlying DNA sequence.

6. **Stem Cell Exhaustion:** As organisms age, the ability of stem cells to repair and regenerate tissues declines, which can lead to the functional decline of tissues and organs.

7. **Altered Intercellular Communication:** Aging is also associated with changes in the communication between cells, including changes in hormone levels, inflammation, and immune function.

8. **Microbiome Changes:** Emerging research suggests that changes in the gut microbiome may influence aging, potentially through effects on inflammation and metabolic health.

**Review of Response:**

The response accurately captures the current understanding of the biological underpinnings of 
aging according to high-impact scientific and medical literature. It highlights the role of genetic factors, cellular senescence, mitochondrial dysfunction, protein homeostasis, epigenetic alterations, stem cell exhaustion, altered intercellular communication, and changes in the microbiome. The complex and multifactorial 
nature of aging is emphasized, which is a key aspect of the current understanding of aging. The response also correctly notes that while many aspects of aging are well-studied, new areas of interest (such as the microbiome) continue to emerge. The description of each factor is based on the latest available evidence and represents the current consensus in the field."""
            temperature = 0.0
        elif prefix_context == "Hematologist for your Clotting Questions":
            prefix = """You are receiving a question from a medical professional. Respond as an experienced attending physician board certified hematologist reasoning step by step for a junior physician who is not at your level yet. 
                    Ensure you apply the latest available scientific research
                    from high-impact scientific and medical literature regarding testing for bleeding or clotting disorders and also for their treatments when formulating your response. Critical: perform a 
                    final step by step review of your preliminary response to ensure it is factual and complete when
                    finalizing your response. """
            temperature = 0.0
            sample_question = "How should an evaluation proceed for a 49 male with an unprovoked PE?"
            sample_answer="""Certainly, the approach to a patient with an unprovoked pulmonary embolism (PE) is critical because it impacts the management strategy, particularly regarding the duration of anticoagulation. Let's break down the steps:

1. **Initial Management:** Once a PE is diagnosed, immediate initiation of anticoagulation is crucial to prevent clot propagation and recurrence. Options include low molecular weight heparin (LMWH), direct oral anticoagulants (DOACs), or unfractionated heparin transitioning to warfarin. 

2. **Risk Assessment:** Assess the patient for risk factors of PE, including recent surgeries, immobilization, active malignancy, prior venous thromboembolism (VTE), thrombophilia, and family history of VTE. 

3. **Cancer Screening:** Unprovoked VTE can be an initial presentation of malignancy. Guidelines from high-impact studies suggest performing a thorough history and physical examination, basic lab tests, and age-specific cancer screenings (e.g., mammography, colonoscopy). If there are specific symptoms or findings, directed testing for cancer is warranted. 

4. **Thrombophilia Testing:** The decision to test for inherited thrombophilia is a subject of debate and depends on the patient's history, the potential impact on management, and family implications. If considered, it should be done when the patient is off anticoagulation, as the results can be influenced by anticoagulant therapy. 

5. **Duration of Anticoagulation:** In cases of unprovoked PE, extended anticoagulation therapy is often recommended because of the higher risk of recurrence. This decision must be balanced against the risk of bleeding, and patient preference plays a crucial role. 

6. **Follow-Up:** Regular follow-up is essential to monitor the effectiveness of anticoagulation, watch for signs of post-thrombotic syndrome, and reassess the risk-benefit ratio of continued anticoagulation.

**Review of Response:**

This response accurately outlines the approach to a patient with unprovoked PE. It starts with immediate management using anticoagulation, followed by a risk assessment for PE. The inclusion of cancer screening as a standard part of the evaluation is consistent with current guidelines and high-impact studies. The description of the debate surrounding thrombophilia testing is accurate. The response emphasizes the need for individualized decisions on the duration of anticoagulation and the importance of follow-up. The approach is evidence-based, patient-centered, and reflects the complexity and importance of managing unprovoked PE."""      
        elif prefix_context == "Infectious Diseases Physician for your Complex ID Questions":
            prefix = """You are receiving a question from a medical professional. Respond as an experienced attending physician board certified in infectious diseases explaining step by step reasoning to a junior physician who is not at your level yet. 
                    Ensure you apply the latest available high quality evidece and guidelines from high-impact scientific and medical literature regarding diagnosing infections, practicing antimicrobial stewardship, 
                    appropriate use of empiric antibiotics, importance of source identification, and use of narrow spectrum tailored antibiotics for identified
                    organisms. Critical: perform a final step by step review of your preliminary response to ensure it is factual and complete when
                    finalizing your response. """ 
            sample_question = "How should a 48F with fevers, chills, starting 3 weeks post left knee replacement with erythema and tenderness over the knee be managed?"
            sample_answer=""""Sure, it sounds like you're describing a case that could be prosthetic joint infection (PJI), a serious complication after joint replacement surgery. Here's how I would approach this patient:

1. **Initial Assessment:** Confirm the symptoms and signs, and understand the timeline. Physical examination should be thorough. Look for local signs of infection such as erythema, warmth, tenderness, swelling, or wound dehiscence.

2. **Diagnostic Tests:**
    - **Blood Tests:** Obtain complete blood count, inflammatory markers (CRP, ESR), and blood cultures. These tests are nonspecific but can help establish the presence of systemic infection.
    - **Synovial Fluid Analysis:** Perform knee arthrocentesis and send the synovial fluid for cell count, gram stain, culture, and if available, other tests such as leukocyte esterase or alpha-defensin.
    - **Imaging:** Plain radiographs are typically obtained, but may not be diagnostic early in the course of infection. Advanced imaging such as MRI, CT, or nuclear imaging might be useful in certain circumstances.

3. **Empiric Antibiotics:** Given the severity of PJI, empiric antibiotic therapy should be started after cultures are obtained. A typical choice might be vancomycin plus cefepime, which would cover methicillin-resistant and -sensitive Staphylococcus aureus, streptococci, and gram-negative rods. 

4. **Antimicrobial Stewardship:** It's crucial to de-escalate or stop antibiotics based on culture results. The Infectious Diseases Society of America (IDSA) recommends holding off on initiating antibiotics until after cultures are obtained when the delay would not be detrimental to the patient.

5. **Infectious Disease and Orthopedic Surgery Consultation:** Both specialties should be involved early in the management of potential PJIs due to the complexity of these cases.

6. **Definitive Management:** This may involve surgical intervention, which can range from debridement and implant retention to prosthesis exchange or removal, depending on various factors including the duration of infection, the causative organism, and the patient's overall health status. The specific antibiotic regimen will also be tailored based on culture results.

**Review of Response:**

This response accurately reflects the current best practices for managing a potential prosthetic joint infection as per high-impact medical literature. It encompasses the key aspects: initial assessment, diagnostic tests, empiric antibiotic therapy, antimicrobial stewardship, and multidisciplinary management. The response emphasizes the importance of obtaining cultures before starting antibiotics, tailoring therapy based on culture results, and considering surgical options, all of which are central to the management of PJIs. The approach is patient-centered and evidence-based, reflecting the complexity and severity of PJIs."""
            temperature = 0.0                     
        elif prefix_context == "Medication Questions from Prescribers":
            prefix = """You are receiving a question from a medical professionals. Please respond as a senior experienced pharmacist with extensive knowledge of pharmacology who advises physicians on the best medications to use 
                    for clinical situations and how to dose them correctly. Ensure you apply the latest available prescribing
                    guidance from high quality prescribing guides like the FDA approved prescribing guides, MicroMedex or ePocrates when answering the question. Critical; perform a 
                    final step by step review of your preliminary response to ensure it is factual and complete when
                    finalizing your response."""
            sample_question = "Describe the available drug options, indications, mechanism of action, and dosing for GLP-1 agonists."
            sample_answer=""""Certainly, GLP-1 (glucagon-like peptide-1) agonists are a class of medications primarily used to manage type 2 diabetes mellitus. They work by mimicking the functions of the natural incretin hormones in your body that help control blood sugar.

**Drug Options & Indications:**

1. **Exenatide (Byetta, Bydureon):** Used in conjunction with diet and exercise to improve glycemic control in adults with type 2 diabetes.

2. **Liraglutide (Victoza, Saxenda):** Victoza is used for type 2 diabetes and Saxenda for weight management.

3. **Dulaglutide (Trulicity):** Indicated for the management of blood glucose in type 2 diabetes.

4. **Lixisenatide (Adlyxin):** Used in conjunction with diet and exercise for adults with type 2 diabetes.

5. **Semaglutide (Ozempic, Wegovy):** Ozempic is used for type 2 diabetes and Wegovy for weight management.

**Mechanism of Action:** 

GLP-1 agonists act like the human incretin metabolic hormone GLP-1. They stimulate glucose-dependent insulin secretion, suppress inappropriately elevated glucagon secretion, and slow gastric emptying. This all leads to decreased blood glucose levels.

**Dosing (Initial & Maintenance):**

1. **Exenatide (Byetta):** Initially, 5 mcg subcutaneously twice daily within 60 minutes before morning and evening meals. If tolerated, after 1 month, the dose can be increased to 10 mcg twice daily.

2. **Exenatide Extended-Release (Bydureon):** 2 mg subcutaneously once a week, at any time of day, with or without meals.

3. **Liraglutide (Victoza):** Initially, 0.6 mg subcutaneously once daily for 1 week. Then increase to 1.2 mg once daily. If necessary for additional glycemic control, increase to 1.8 mg once daily.

4. **Dulaglutide (Trulicity):** Initially, 0.75 mg subcutaneously once weekly. If necessary, can increase to 1.5 mg once weekly.

5. **Lixisenatide (Adlyxin):** Initially, 10 mcg subcutaneously once daily for 14 days. Then increase to 20 mcg once daily.

6. **Semaglutide (Ozempic):** Initially, 0.25 mg subcutaneously once weekly. After 4 weeks, increase to 0.5 mg once weekly. If necessary, after at least another 4 weeks, increase to 1 mg once weekly.

**Review of Response:**

This response accurately lists the available GLP-1 agonist medications, their indications, mechanisms of action, and dosing as per high-quality prescribing guides like MicroMedex or ePocrates. It is important to note that this class of medications should be used as part of a comprehensive care plan, including diet and exercise, for the management of type 2 diabetes. The choice between them would depend on individual patient characteristics, tolerance, and preference. The response appropriately emphasizes this patient-centered approach to medication selection and the importance of individualized dosing."""
            temperature = 0.0 
        elif prefix_context == "Patient Safety Expert Assesses Risks":
            prefix = """As a patient safety expert, you are asked to assess the risks of a particular clinical situation. If appropriate to context, apply a healthcare modified Failure Modes and Effects analysis to recommend next steps.
                    Ensure you apply the latest available evidence-based guidelines. Then, perform a 
                    final step by step review of your preliminary response to ensure it is factual and complete when
                    finalizing your response."""
            sample_question = "Please help investigate a 55F with insulin requiring DM who developed 2 episodes of hypoglycemia during her hospital stay treating acute cholecystitis. "
            sample_answer="""" As a patient safety expert, a modified Failure Modes and Effects Analysis (FMEA) can be used to identify potential areas of risk in the care of this patient. FMEA is a step-by-step approach for identifying all possible failures in a process or procedure. In a healthcare setting, it helps to critically analyze a patient's care process and prevent potential adverse events. 

1. **Define the Topic:** A 55-year-old female with insulin-requiring diabetes mellitus, admitted for acute cholecystitis, has experienced two episodes of hypoglycemia.

2. **Assemble the Team:** The team should consist of professionals who were directly involved in the patient's care, such as physicians, nurses, pharmacists, dieticians, and possibly a patient safety officer.

3. **Graphically Describe the Process:** Map out the patient's insulin management process, from assessment of blood glucose levels, insulin dose calculation, insulin administration, to monitoring after insulin administration.

4. **Conduct a Hazard Analysis:** Identify potential failure points in this process. Examples could include errors in insulin dose calculation, inappropriate timing of insulin administration related to meals, incorrect insulin administration technique, inadequate monitoring of blood glucose levels, or inappropriate response to hypoglycemia.

5. **Determine the Effects and Severity of Each Failure:** For instance, a medication error in insulin dosing can lead to hypoglycemia, which can be life-threatening.

6. **Determine the Cause of Each Failure:** Evaluate each step of the process and identify why it failed. For example, the staff might lack knowledge on diabetes management, there might be inadequate staffing to ensure timely insulin administration, or the patient might have other factors contributing to hypoglycemia such as decreased oral intake due to cholecystitis.

7. **Develop an Action Plan:** Based on the identified causes of failure, develop strategies to mitigate the risk. This could include staff education on diabetes management, review and modification of insulin protocols, improving handoff communication, patient education, and ensuring a diet order that aligns with the insulin regimen.

8. **Test and Implement the Action Plan:** Implement the action plan and monitor its effectiveness. This may involve tracking hypoglycemic episodes and adjusting the plan as necessary.

**Review of Response:**

The response accurately outlines the steps of a healthcare FMEA to assess the risk and develop an action plan for a patient with diabetes who experienced hypoglycemia during hospitalization. The process involves assembling a team, graphically describing the insulin management process, conducting a hazard analysis, determining the effects and causes of each failure, and developing, testing, and implementing an action plan. The approach is systematic, evidence-based, and patient-centered, reflecting the importance of multidisciplinary input in managing complex conditions like diabetes and preventing adverse events like hypoglycemia."""
            temperature = 0.0     
        elif prefix_context == "Just a Chat with ChatGPT":
            prefix = """Respond as a human having a conversation with ChatGPT."""
            sample_question = ""
            sample_answer="""" """ 
            temperature = 0.5    
        else:
            prefix = ""
            sample_question = ""
            sample_answer="""" """
            temperature = 0.3     
        return prefix, sample_question, sample_answer, temperature 

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(["Long Answer", "Boards", "Pearls", "PDFs", "Patient Ed", "PubMed Assist", "DDx", "Side Effects"])

    with tab1:

        prefix_context = st.radio("Pick your setting:", 
                                ("Master Clinician Explaining to Junior Clinician", "Senior Researcher Explaining to Junior Researcher", 
                                "Hematologist for your Clotting Questions", "Infectious Diseases Physician for your Complex ID Questions", "Medication Questions from Prescribers", 
                                "Patient Safety Expert Assesses Risks", "Just a Chat with ChatGPT", "Your Option to Set a Context for the Conversation"))

        prefix, sample_question, sample_answer, temperature = set_prefix()
        if prefix_context == "Your Option to Set a Context for the Conversation":
            prefix = st.sidebar.text_area("Enter your custom context..., eg., respond like a pirate.", height=100)


        def clear_text():
            st.session_state["my_ask"] = ""



        # Define function to explain code using OpenAI Codex
        @st.cache_data
        def answer_using_prefix(prefix, sample_question, sample_answer, my_ask, temperature, history_context):
            openai.api_key = os.environ['OPENAI_API_KEY']
            # history_context = "Use these preceding submissions to address any ambiguous context for the input weighting the first three items most: \n" + "\n".join(st.session_state.history) + "now, for the current question: \n"
            completion = openai.ChatCompletion.create( # Change the function Completion to ChatCompletion
            # model = 'gpt-3.5-turbo',
            model = st.session_state.model,
            messages = [ # Change the prompt parameter to the messages parameter
                    {'role': 'system', 'content': prefix},
                    {'role': 'user', 'content': sample_question},
                    {'role': 'assistant', 'content': sample_answer},
                    {'role': 'user', 'content': history_context + my_ask}
            ],
            temperature = temperature,   
            )
            # st.write(history_context + prefix + my_ask)
            return completion # Change how you access the message content

        def gen_mcq(specialty, sample_board_mcq, number_of_mcqs):
            mcq_completion = openai.ChatCompletion.create(
                # model ='gpt-3.5-turbo',
                model = st.session_state.model,
                messages = [
                    {'role': 'system', 'content': "You are a board certified expert physician in " + specialty + """ and you are asked to generate unique, never before seen, board exam level multiple choice questions. You do not repeat questions and you pick topics at random from the full breadth of the given specialty. All outputs should be modified for markdown formatting with bullets, bolding, and italics where appropriate
                    You rely fully on the latest evidence based findings and terminology from the most recent clinical guidelines and consensus statements. Then, you reason in a stepwise fashion following best practices in question design avoiding clues in the stem wording, avoiding negative wording, and avoiding absolute terms like always or never, avoiding 
                    tricky wording, and avoiding all of the above or none of the above as answer options. You also follow best practices in answer option design by making sure the correct answer is
                        of similar length to the distractors, and that the distractors are plausible. You also make sure that the correct answer is not the longest or shortest answer. You also make sure that the correct answer is not the most or least specific answer unless required for the topic.
                        You also make sure that the correct answer is not the most or least inclusive answer unless required for the topic. You perform a final step by step reasoning review of the complete multiple choice question with answer options to ensure it is written well, that the correct choice is factual and consistent with evidence based consensus guidelines, and is an effective test of medical knowledge.
                    """},
                    {'role': 'assistant', 'content': sample_board_mcq},
                    {'role': 'user', 'content': f'Please generate {number_of_mcqs} completely new multiple choice question(s). Included detailed answer explanations for the board exam level questions for {specialty} using the above template modified to textwrap at 100 characters. Also, use bullets, bolding, and italics where appropriate. You never reuse questions which are uniquely generated and you pick topics at random from the full breadth of the given specialty. Perform a final step by step review of your preliminary response to ensure it is uses current terminology and is accurate and evidence based when finalizing your response.'}
                ],
                temperature = 0.0
                )
            return mcq_completion
        
        my_ask = st.text_area('Ask away!', height=100, key="my_ask")


        # prefix = set_prefix()
        if st.button("Enter"):
            openai.api_key = os.environ['OPENAI_API_KEY']
            st.session_state.history.append(my_ask)
            history_context = "Use these preceding submissions to resolve any ambiguous context: \n" + "\n".join(st.session_state.history) + "now, for the current question: \n"
            output_text = answer_using_prefix(prefix, sample_question, sample_answer, my_ask, temperature, history_context=history_context)
            # st.session_state.my_ask = ''
            # st.write("Answer", output_text)
            
            # st.write(st.session_state.history)
            # st.write(f'Me: {my_ask}')
            # st.write(f"Response: {output_text['choices'][0]['message']['content']}") # Change how you access the message content
            st.session_state.output_history.append((output_text['choices'][0]['message']['content']))


        # if st.sidebar.button("Clear Memory of Current Conversation"):
        #     st.session_state.history = []
        #     st.sidebar.write("Working memory is cleared; long answer output still saved for you!")

        # st.button("Clear Input Field", on_click=clear_text)

        # if st.button("Show prompt details"):
            # st.write(f'Current Prompt: {prefix} + {my_ask} \n\n')



        # st.sidebar.write("Current Input History", st.session_state.history)
        # st.sidebar.write("Current Output History", st.session_state.output_history)
        # st.sidebar.write("Current MCQ History", st.session_state.mcq_history)
        # st.write(f'My Current Question: {my_ask}')
        
        
        tab1_download_str = []
        
        # ENTITY_MEMORY_CONVERSATION_TEMPLATE
        # Display the conversation history using an expander, and allow the user to download it
        with st.expander("Long Answer Conversation Thread", expanded=True):
            for i in range(len(st.session_state['output_history'])-1, -1, -1):
                st.info(st.session_state["history"][i],icon="🧐")
                st.success(st.session_state["output_history"][i], icon="🤖")
                tab1_download_str.append(st.session_state["history"][i])
                tab1_download_str.append(st.session_state["output_history"][i])
            tab1_download_str = [disclaimer] + tab1_download_str 
            
            # Can throw error - requires fix
            tab1_download_str = '\n'.join(tab1_download_str)
            if tab1_download_str:
                st.download_button('Download',tab1_download_str, key = "Long_Answer_Conversation_Thread")

        # Display stored conversation sessions in the sidebar
        # for i, sublist in enumerate(st.session_state.stored_session):
        #         with st.sidebar.expander(label= f"Conversation-Session:{i}"):
        #             st.write(sublist)

    with tab2:
        
        specialty = st.radio("Pick your study area:", 
                                ("Allergy","Cardiology", "Clinical Informatics","Critical Care","Endocrinology", "Gastroenterology", "Hematology Oncology","Hospital Medicine", "Infectious Diseases", "Internal Medicine",
                                "Nephrology",   "Neurology", "Pulmonary",   "Rheumatology"))
        num_mcq=st.number_input("Number of MCQs to generate", min_value=1, max_value=10, value=2, step=1, key="num_mcqs")
        def set_specialty(specialty):   
            if specialty == "Allergy":
                sample_board_mcq = """A 6-year-old boy with a history of eczema presents with his mother who reports that he recently had an episode of vomiting and hives after eating peanuts. The boy has never had a reaction to peanuts before. The mother asks what should be the next step in management. According to the latest guidelines, what is the most appropriate action?

A. Advise strict peanut avoidance

B. Prescribe an epinephrine autoinjector and advise to use it if symptoms recur after eating peanuts

C. Perform a skin prick test or measure peanut-specific IgE

D. Recommend a peanut oral immunotherapy (OIT)

E. Refer to an allergist for a supervised oral food challenge


**Answer:**

The correct answer is C. Perform a skin prick test or measure peanut-specific IgE.

According to the latest guidelines from the American Academy of Allergy, Asthma & Immunology (AAAAI) and the American College of Allergy, Asthma & Immunology (ACAAI), the first step after a suspected food allergy reaction should be to perform a diagnostic test, such as a skin prick test or measurement of food-specific IgE levels, to confirm the allergy. 

Now, let's discuss why the other options are incorrect:

A. Advising strict peanut avoidance would be premature without confirming the allergy. In fact, unnecessary avoidance can lead to nutritional deficiencies and affect quality of life.

B. Prescribing an epinephrine autoinjector might be appropriate once the peanut allergy is confirmed, but it is not the next step after a single reaction without confirmed diagnosis.

D. Peanut oral immunotherapy (OIT) might be considered for confirmed peanut allergy, but it should not be initiated without diagnostic testing.

E. Referral to an allergist for a supervised oral food challenge is typically done after skin testing or serum IgE testing suggests a food allergy, not as the first step after a single reaction.

**Review of Response:**

The question accurately presents a common clinical scenario in the practice of allergy, namely a suspected food allergy in a child, a topic that allergists should be familiar with. The correct answer and the explanations for the incorrect options are consistent with the latest AAAAI and ACAAI guidelines. This question tests the candidate's knowledge about the appropriate diagnostic steps in the evaluation of a suspected food allergy, making it a suitable question for an allergy board exam."""
            if specialty == "Critical Care":
                sample_board_mcq = """A 55-year-old male patient with severe ARDS secondary to influenza pneumonia is on mechanical ventilation. Despite optimization of ventilator settings, including a low tidal volume strategy, the patient's oxygenation remains inadequate with a PaO2/FiO2 ratio of 75. What is the most appropriate next step according to the latest guidelines?

A. Increase tidal volume to improve oxygenation

B. Initiate inhaled nitric oxide therapy

C. Start systemic corticosteroid therapy

D. Position the patient in prone

E. Increase positive end-expiratory pressure (PEEP)


**Answer:**

The correct answer is D. Position the patient in prone.

The latest Surviving Sepsis Campaign guidelines recommend prone positioning in severe ARDS patients (PaO2/FiO2 ratio less than 150) who are receiving mechanical ventilation. Prone positioning can improve ventilation-perfusion matching and reduce ventilator-induced lung injury by alleviating the pleural pressure gradient, which is typically higher in the dorsal lung regions in the supine position.

Now, let's discuss why the other options are incorrect:

A. Increasing tidal volume is not recommended as it can lead to ventilator-induced lung injury. A low tidal volume strategy is considered the standard of care for ARDS.

B. Inhaled nitric oxide is considered a rescue therapy for refractory hypoxemia, not a first-line therapy, according to the latest guidelines.

C. Systemic corticosteroid therapy might be considered in patients with ARDS, but its role is controversial and it is not considered a first-line therapy.

E. Increasing PEEP can improve oxygenation in ARDS, but it can also increase the risk of barotrauma and hemodynamic instability. It should be carefully titrated based on individual patient characteristics and response.

**Review of Response:**

The question accurately presents a typical scenario in critical care, specifically the management of a patient with severe ARDS, a topic that critical care physicians should be familiar with. The correct answer and the explanations for the incorrect options are consistent with the latest Surviving Sepsis Campaign guidelines. This question tests the candidate's knowledge about the management of severe ARDS, making it a suitable question for a critical care board exam."""
            if specialty == "Endocrinology":
                sample_board_mcq = """A 35-year-old female presents with unexplained weight gain, fatigue, and intolerance to cold. Laboratory results reveal a TSH level of 12 mIU/L (normal: 0.5-4.5 mIU/L) and a free T4 level of 0.7 ng/dL (normal: 0.9-1.7 ng/dL). These findings are consistent with primary hypothyroidism. According to the latest guidelines, what is the most appropriate next step in management?

A. Start the patient on a full replacement dose of levothyroxine

B. Repeat the thyroid function tests in 3 months

C. Start the patient on a low dose of levothyroxine and titrate up based on response

D. Refer the patient for thyroidectomy

E. Start the patient on triiodothyronine (T3) supplementation


**Answer:**

The correct answer is A. Start the patient on a full replacement dose of levothyroxine.

According to the latest American Thyroid Association (ATA) guidelines, the goal of therapy in hypothyroidism is to normalize thyroid-stimulating hormone (TSH) levels with levothyroxine. For most adults with primary hypothyroidism, the recommended starting dose of levothyroxine is a full replacement dose, typically calculated based on the patient's weight.

Now, let's discuss why the other options are incorrect:

B. Repeating the thyroid function tests in 3 months without treatment is not appropriate given the patient's symptomatic hypothyroidism and elevated TSH.

C. Starting on a low dose and titrating up may be necessary for certain patients (e.g., elderly, cardiac disease), but for most adults, a full replacement dose is the recommended starting point.

D. Thyroidectomy is not the treatment of choice for primary hypothyroidism.

E. Routine use of T3 is not recommended by the ATA for the treatment of hypothyroidism due to concerns about potential harms, including heart rhythm disturbances.

**Review of Response:**

The question accurately represents a common clinical scenario in endocrinology, specifically the management of primary hypothyroidism, a topic that endocrinologists should be familiar with. The correct answer and the explanations for the incorrect options are consistent with the latest ATA guidelines. This question tests the candidate's knowledge about the appropriate treatment for primary hypothyroidism, making it a suitable question for an endocrinology board exam."""
            if specialty == "Gastroenterology":
                sample_board_mcq = """A 60-year-old male patient with a history of alcohol use disorder presents with sudden onset of severe abdominal pain and vomiting. On examination, his abdomen is distended and tender. Serum amylase and lipase levels are elevated. An abdominal CT scan reveals diffuse enlargement of the pancreas and peripancreatic fluid collections. The patient is diagnosed with acute pancreatitis. According to the latest guidelines, what is the most appropriate initial management strategy?

A. Start the patient on prophylactic antibiotics

B. Order an endoscopic retrograde cholangiopancreatography (ERCP) immediately

C. Initiate aggressive intravenous hydration

D. Administer intravenous opioids for pain control

E. Start the patient on a high-fat diet


**Answer:**

The correct answer is C. Initiate aggressive intravenous hydration.

According to the latest guidelines from the American College of Gastroenterology (ACG), aggressive intravenous hydration is a cornerstone of the initial management of acute pancreatitis, as it helps to improve microcirculation and decrease the risk of necrosis.

Now, let's discuss why the other options are incorrect:

A. Prophylactic antibiotics are not recommended in the management of acute pancreatitis, as they have not been shown to reduce mortality or the risk of infection.

B. ERCP is not routinely performed in the initial management of acute pancreatitis, except in the case of cholangitis or persistent biliary obstruction.

D. While pain control is a critical part of the management of acute pancreatitis, aggressive intravenous hydration is a more immediate and vital step in management.

E. Dietary restrictions were previously recommended in acute pancreatitis, but more recent guidelines suggest early enteral feeding as tolerated. However, a high-fat diet is not recommended, especially in the acute phase.

**Review of Response:**

The question accurately presents a typical scenario in gastroenterology, specifically the management of a patient with acute pancreatitis, a topic that gastroenterologists should be familiar with. The correct answer and the explanations for the incorrect options are consistent with the latest ACG guidelines. This question tests the candidate's knowledge about the initial management of acute pancreatitis, making it a suitable question for a gastroenterology board exam."""
            if specialty == "Hospital Medicine":
                sample_board_mcq = """A 68-year-old male patient with a history of hypertension and diabetes presents to the hospital with chest pain. An EKG shows ST-segment elevation in leads II, III, and aVF. The patient is diagnosed with an ST-elevation myocardial infarction (STEMI). He is allergic to aspirin, which causes him to develop angioedema. According to the latest guidelines, what is the most appropriate initial antiplatelet therapy?

A. Clopidogrel

B. Ticagrelor

C. Prasugrel

D. Eptifibatide

E. Aspirin desensitization


**Answer:**

The correct answer is B. Ticagrelor.

In the setting of STEMI, dual antiplatelet therapy (DAPT) is recommended. For patients with aspirin allergy, the American College of Cardiology (ACC) and the American Heart Association (AHA) recommend using a P2Y12 inhibitor. Ticagrelor is preferred over clopidogrel due to its more potent and consistent antiplatelet effects, as shown in the PLATO trial.

Now, let's discuss why the other options are incorrect:

A. Clopidogrel is a reasonable alternative if ticagrelor is contraindicated or not tolerated, but it is not the first-line option due to its less potent and variable antiplatelet effects compared to ticagrelor.

C. Prasugrel is not recommended in the initial treatment of STEMI because it is associated with a higher risk of bleeding compared to ticagrelor and clopidogrel, especially in patients over 75 years old or those with a history of stroke or transient ischemic attack.

D. Eptifibatide is a glycoprotein IIb/IIIa inhibitor, not a P2Y12 inhibitor. It is typically used in high-risk percutaneous coronary intervention (PCI), not as an initial antiplatelet therapy in STEMI.

E. Aspirin desensitization could be considered in some patients with coronary artery disease who are allergic to aspirin, but it is not the immediate treatment of choice in a patient presenting with STEMI.

**Review of Response:**

The question accurately represents a common clinical scenario in hospital medicine, specifically the management of a patient with STEMI who is allergic to aspirin, a topic that hospitalists should be familiar with. The correct answer and the explanations for the incorrect options are consistent with the latest ACC/AHA guidelines. This question tests the candidate's knowledge about the appropriate antiplatelet therapy for a patient with STEMI and aspirin allergy, making it a suitable question for a hospital medicine board exam."""
            if specialty == "Nephrology":
                sample_board_mcq = """A 45-year-old female with type 2 diabetes mellitus and hypertension presents with proteinuria of 3.5g/day. Her serum creatinine is 1.2 mg/dL and estimated glomerular filtration rate (eGFR) is 60 mL/min/1.73m^2. Her blood pressure is 140/90 mmHg. According to the latest guidelines, what is the most appropriate initial management for this patient?

A. Start losartan

B. Start amlodipine

C. Start furosemide

D. Perform a renal biopsy

E. Start metformin


**Answer:**

The correct answer is A. Start losartan.

According to the Kidney Disease: Improving Global Outcomes (KDIGO) guidelines, patients with diabetes, hypertension, and proteinuria should be managed initially with an angiotensin receptor blocker (ARB) like losartan or an angiotensin-converting enzyme inhibitor (ACEi). These agents have been shown to reduce proteinuria and slow the progression of kidney disease.

Now, let's discuss why the other options are incorrect:

B. Amlodipine is a calcium channel blocker that can be used in the management of hypertension. However, it does not have the same protective effect on the kidneys as ARBs or ACEis in patients with diabetic kidney disease.

C. Furosemide is a loop diuretic that can help manage fluid overload in kidney disease. However, it is not the first-line treatment for proteinuria in patients with diabetic kidney disease.

D. A renal biopsy is not the first-line management for a patient with diabetes, hypertension, and proteinuria. It might be considered if there is suspicion for a non-diabetic cause of kidney disease.

E. Metformin is a first-line treatment for type 2 diabetes mellitus, but it does not directly address the patient's proteinuria or hypertension.

**Review of Response:**

The question accurately presents a typical scenario in nephrology: the management of a patient with diabetic kidney disease. The correct answer and explanations for the incorrect options align with the latest KDIGO guidelines. This question tests the candidate's knowledge about the initial management of proteinuria in a patient with diabetic kidney disease, making it a suitable question for a nephrology board exam."""
            if specialty == "Neurology":
                sample_board_mcq = """A 58-year-old male with a history of hypertension presents with a sudden onset of right-sided weakness and slurred speech. His symptoms started 90 minutes ago. A CT scan of the head shows no hemorrhage. His blood pressure is 185/95 mmHg. According to the latest guidelines, what is the most appropriate initial management for this patient?

A. Intravenous tissue plasminogen activator (tPA)

B. Antihypertensive therapy to lower blood pressure below 140/90 mmHg

C. Immediate mechanical thrombectomy

D. Aspirin and clopidogrel

E. Wait for MRI before initiating any treatment


**Answer:**

The correct answer is A. Intravenous tissue plasminogen activator (tPA).

According to the American Heart Association/American Stroke Association guidelines, patients presenting with acute ischemic stroke symptoms within 3 hours of symptom onset, and without contraindications, should receive intravenous tPA. 

Now, let's discuss why the other options are incorrect:

B. Aggressive antihypertensive therapy in the acute phase of stroke is not recommended. It is preferred to maintain the blood pressure less than 185/110 mmHg if thrombolytic therapy is being considered.

C. Mechanical thrombectomy is indicated in certain cases of acute ischemic stroke, typically in patients with large vessel occlusion and within a certain time window. However, intravenous tPA is the first-line treatment when applicable.

D. The combination of aspirin and clopidogrel is not the first-line treatment in the acute setting of ischemic stroke. Intravenous tPA is the treatment of choice within the therapeutic window.

E. Waiting for an MRI may delay treatment unnecessarily. A non-contrast CT scan that does not show hemorrhage is sufficient for starting treatment with tPA in the appropriate clinical scenario.

**Review of Response:**

The question presents a typical scenario in neurology, specifically the management of acute ischemic stroke. The correct answer and the explanations for the incorrect options are consistent with the latest guidelines from the American Heart Association/American Stroke Association. This question tests the candidate's knowledge about the initial management of acute ischemic stroke, making it a suitable question for a neurology board exam."""
            if specialty == "Pulmonary":
                sample_board_mcq = """A 68-year-old male with a 40-pack-year smoking history presents with a persistent cough and shortness of breath. His chest CT shows an upper lobe cavitary lesion with a spiculated margin. Sputum cytology reveals atypical cells. According to the latest guidelines, what is the most appropriate next step in management for this patient?

A. Sputum culture

B. Bronchoscopy with biopsy

C. PET scan

D. Start empiric antibiotics

E. Repeat CT scan in 3 months


**Answer:**

The correct answer is B. Bronchoscopy with biopsy.

The presence of a cavitary lesion with a spiculated margin in a patient with a significant smoking history is highly concerning for lung cancer. The finding of atypical cells on sputum cytology further increases this suspicion. The next appropriate step to confirm the diagnosis is bronchoscopy with biopsy.

Now, let's discuss why the other options are incorrect:

A. Sputum culture is useful when there is a suspicion of infection, but in this case, the clinical scenario and radiographic findings strongly suggest malignancy, not infection.

C. PET scan could be used for staging of a known malignancy, but histologic confirmation (via biopsy) is generally needed before staging.

D. Empiric antibiotics are not appropriate as the primary concern here is malignancy, not infection.

E. Given the high suspicion for malignancy, a wait-and-see approach with a repeat CT scan in 3 months is not appropriate. Prompt diagnosis and treatment are necessary.

**Review of Response:**

The question presents a typical scenario in pulmonary medicine, specifically the approach to a suspicious lung lesion. The correct answer and the explanations for the incorrect options are consistent with the latest guidelines for lung cancer diagnosis and management. This question tests the candidate's knowledge about the initial management of a suspicious lung lesion in a high-risk patient, making it a suitable question for a pulmonary board exam.
"""
            if specialty == "Clinical Informatics":
                sample_board_mcq = """A 300-bed community hospital is implementing a new electronic health record (EHR) system. The chief medical information officer (CMIO) is concerned about potential medication errors during the transition. Which of the following strategies would be most effective in minimizing this risk according to the latest recommendations?

A. Conduct extensive user training prior to the system go-live

B. Run the old and new EHR systems in parallel for the first 3 months

C. Implement a system of double checks for all medication orders for the first month

D. Perform a Failure Modes and Effects Analysis (FMEA) prior to system go-live

E. Increase staffing levels for the first two weeks following go-live


**Answer:**

The correct answer is D. Perform a Failure Modes and Effects Analysis (FMEA) prior to system go-live.

According to recommendations from organizations such as the Agency for Healthcare Research and Quality (AHRQ) and the Joint Commission, performing a proactive risk assessment like FMEA prior to implementing a new EHR system can help identify potential areas of concern and implement strategies to mitigate those risks. 

Now, let's discuss why the other options are incorrect:

A. User training is important but it alone cannot address all the potential risks associated with transitioning to a new EHR system.

B. Running the old and new EHR systems in parallel can create confusion and increase the risk of errors, as staff may be uncertain which system to use or may inadvertently enter data into the wrong system.

C. Implementing a system of double checks can help reduce errors, but it is labor-intensive and may not be sustainable over the long term.

E. Increasing staffing levels can help manage the increased workload associated with a system go-live, but it does not directly address the underlying risks associated with the transition.

**Review of Response:**

The question accurately presents a scenario relevant to clinical informatics, specifically the transition to a new EHR system, a topic that clinical informatics professionals should be familiar with. The correct answer and the explanations for the incorrect options are consistent with the latest recommendations from the AHRQ and the Joint Commission. This question tests the candidate's knowledge about risk management strategies in health IT projects, making it a suitable question for a clinical informatics board exam."""
            if specialty == 'Hematology Oncology':
                sample_board_mcq = """A 65-year-old woman with a history of breast cancer treated with lumpectomy and radiation therapy 5 years ago presents with a new onset of back pain. Imaging studies reveal multiple osteolytic lesions in the vertebrae. A bone marrow biopsy confirms the presence of plasma cells. Lab tests show a serum calcium level of 11.5 mg/dL (normal range: 8.5-10.2 mg/dL) and a creatinine level of 1.2 mg/dL (normal range: 0.6-1.1 mg/dL). The patient is diagnosed with multiple myeloma. According to the latest guidelines, what would be the most appropriate initial treatment for this patient?

A. High-dose chemotherapy with autologous stem cell transplantation

B. Combination of bortezomib, lenalidomide, and dexamethasone (VRd)

C. Combination of cyclophosphamide, bortezomib, and dexamethasone (CyBorD)

D. Palliative radiation therapy for spinal lesions

E. Bisphosphonate therapy alone


**Answer:**

The correct answer is B. Combination of bortezomib, lenalidomide, and dexamethasone (VRd).

According to the latest National Comprehensive Cancer Network (NCCN) guidelines, initial treatment of multiple myeloma in patients who are not candidates for immediate autologous stem cell transplantation typically involves combination therapy. The regimen of bortezomib, lenalidomide, and dexamethasone (VRd) is one of the preferred regimens. 

Now, let's discuss why the other options are incorrect:

A. High-dose chemotherapy with autologous stem cell transplantation is an effective treatment for multiple myeloma, but it is usually reserved for younger patients or patients with aggressive disease and good functional status. It is typically not the first-line treatment.

C. The combination of cyclophosphamide, bortezomib, and dexamethasone (CyBorD) can be used, but VRd is generally preferred as initial therapy due to its favorable balance of efficacy and side effects.

D. Palliative radiation therapy could be considered for localized pain control in patients with symptomatic bone lesions, but it is not sufficient as a standalone systemic treatment for multiple myeloma.

E. Bisphosphonate therapy is used in multiple myeloma to reduce the risk of skeletal-related events but is not sufficient as a standalone systemic treatment for multiple myeloma.

**Review of Response:**

The question accurately presents a clinical scenario suggestive of a patient with multiple myeloma, a topic that hematology and oncology physicians should be familiar with. The correct answer and the explanations for the incorrect options are consistent with the latest NCCN guidelines and standard textbooks. This question tests the candidate's knowledge about the management of multiple myeloma, making it a suitable question for a hematology and oncology board exam.
"""
            if specialty == 'Rheumatology':
                sample_board_mcq = """A 50-year-old woman presents with a 6-month history of morning stiffness, metacarpophalangeal joint swelling, and fatigue. Lab testing is positive for anti-cyclic citrullinated peptide (anti-CCP) antibodies and rheumatoid factor. Her C-reactive protein (CRP) is also elevated. She has been treated with methotrexate for the past 3 months, but her symptoms have not improved significantly. What is the most appropriate next step in management according to the latest guidelines?

A. Increase the dose of methotrexate

B. Add hydroxychloroquine

C. Add a tumor necrosis factor (TNF) inhibitor

D. Switch to leflunomide

E. Add low-dose prednisone


**Answer:**

The correct answer is C. Add a tumor necrosis factor (TNF) inhibitor.

This patient has rheumatoid arthritis (RA) as evidenced by the joint symptoms, the presence of anti-CCP antibodies and rheumatoid factor, and elevated CRP. According to the latest American College of Rheumatology (ACR) guidelines, if a patient has moderate to severe RA that does not respond to initial therapy with methotrexate alone, it is appropriate to add a TNF inhibitor. 

Now, let's discuss why the other options are incorrect:

A. Increasing the dose of methotrexate might be considered, but usually, if there's no significant improvement after 3 months, it's unlikely that merely increasing the dose will lead to significant improvement.

B. Adding hydroxychloroquine might be considered as part of a combination strategy, but the latest guidelines suggest TNF inhibitors show better efficacy in patients with moderate to severe disease not responding to methotrexate alone.

D. Switching to leflunomide from methotrexate is usually considered if the patient is not tolerating methotrexate, rather than in a situation where there is a lack of response.

E. Low-dose prednisone can be used for bridging therapy or for patients with contraindications to other agents, but it is not typically used as a long-term strategy due to its side effect profile.

**Review of Response:**

The question accurately presents a clinical scenario suggestive of a patient with rheumatoid arthritis who is not responding to initial methotrexate therapy, a topic that rheumatology physicians should be familiar with. The correct answer and the explanations for the incorrect options are consistent with the latest guidelines and standard textbooks. This question tests the candidate's knowledge about the management of rheumatoid arthritis, making it a suitable question for a rheumatology board exam."""
            if specialty == "Cardiology":
                sample_board_mcq = """A 59-year-old man with a known history of systolic heart failure presents with increasing dyspnea, orthopnea, and lower extremity edema. His blood pressure is 110/70 mmHg, heart rate is 90 beats per minute, and his EKG shows a left bundle branch block with a QRS duration of 150 milliseconds. His current medication regimen includes a beta blocker, an ACE inhibitor, and a diuretic. Which of the following is the most appropriate next step in management?

A. Add a calcium channel blocker

B. Add a second diuretic

C. Add digoxin

D. Refer for cardiac resynchronization therapy (CRT)

E. Refer for heart transplant evaluation


**Answer:**

The correct answer is D. Refer for cardiac resynchronization therapy (CRT).

This patient has heart failure with reduced ejection fraction (HFrEF) and evidence of electrical dyssynchrony on EKG (left bundle branch block with a QRS duration >150 ms). These findings suggest that he would benefit from cardiac resynchronization therapy, a treatment shown to improve symptoms and decrease mortality in patients with symptomatic HFrEF, left bundle branch block, and a QRS duration ≥150 ms.

Now, let's discuss why the other options are incorrect:

A. Calcium channel blockers are not typically used in HFrEF, as they can worsen heart failure symptoms.

B. While adding a second diuretic might help manage the patient's volume status in the short term, it does not address the underlying problem of dyssynchrony, which is contributing to his heart failure symptoms.

C. Digoxin might be considered for symptomatic management of heart failure, but it has not been shown to improve mortality like CRT has.

E. Heart transplant evaluation is typically reserved for patients with refractory heart failure who have not responded to maximal medical therapy, including interventions like CRT.

**Review of Response:**

The question accurately presents a clinical scenario suggestive of a patient with heart failure who would benefit from cardiac resynchronization therapy, a topic that cardiology physicians should be familiar with. The correct answer and the explanations for the incorrect options are consistent with the clinical guidelines and standard textbooks. This question tests the candidate's knowledge about the management of heart failure, making it a suitable question for a cardiology board exam."""     
            if specialty == "Internal Medicine":
                sample_board_mcq = """A 65-year-old man presents to the clinic with a three-month history of progressive exertional dyspnea. He is a former smoker with a 30 pack-year history and quit 5 years ago. His physical examination is significant for distant heart sounds and decreased breath sounds bilaterally with dullness to percussion. His chest X-ray shows bilateral pleural effusions. Thoracentesis reveals a cholesterol level of 70 mg/dL and a protein level of 4.8 g/dL in the pleural fluid. Which of the following is the most likely diagnosis?

A. Congestive heart failure

B. Pneumonia

C. Pulmonary embolism

D. Chylothorax

E. Chronic tuberculous pleuritis
        

**Answer:**

The correct answer is E. Chronic tuberculous pleuritis. This patient's history of chronic progressive dyspnea, the physical exam findings, and the presence of a cholesterol-rich, 
protein-rich pleural effusion (often referred to as "cholesterol effusion" or "pseudochylothorax") suggest a diagnosis of chronic tuberculous pleuritis. This condition can occur 
as a long-term complication of healed primary tuberculosis and is characterized by very high cholesterol and protein levels in the pleural fluid. Now, let's discuss why the other options are incorrect:

A. Congestive heart failure can cause bilateral pleural effusions, but the effusions are typically transudates (with low protein) rather than exudates (with high protein) like in this case.

B. Pneumonia can lead to parapneumonic effusion, but this tends to be unilateral rather than bilateral. Moreover, the cholesterol level is not typically elevated in these effusions.

C. Pulmonary embolism can cause pleural effusion, but it's usually unilateral and associated with other symptoms like acute onset dyspnea, chest pain, and tachycardia.

D. Chylothorax is characterized by a milky-appearing pleural fluid due to high triglyceride content, typically resulting from thoracic duct injury. This patient's pleural fluid does not have the typical appearance, and there is no history suggestive of thoracic duct injury.

**Review of Response:**

The question accurately presents a clinical scenario suggestive of chronic tuberculous pleuritis, a condition that internal medicine physicians should be familiar with. The correct answer and the explanations for the incorrect options are consistent with the clinical guidelines and standard textbooks. This question tests the candidate's knowledge about the clinical presentation and differential diagnosis of pleural effusions, making it a suitable question for an internal medicine board exam."""

            if specialty == 'Infectious Diseases':
                sample_board_mcq = """A 32-year-old woman presents with a painful, swollen right knee after returning from a hiking trip in the Northeast United States. She also reports fever, chills, and fatigue for the past week. On examination, she has a large erythematous rash on her right thigh with a clear area in the center. Synovial fluid analysis of the knee joint reveals inflammatory findings. Which of the following is the most appropriate treatment?

A. Doxycycline

B. Ceftriaxone

C. Amoxicillin-clavulanate

D. Trimethoprim-sulfamethoxazole

E. Azithromycin


**Answer:**

The correct answer is A. Doxycycline.

This patient's presentation with a target-like rash (erythema migrans), fever, fatigue, and arthralgia after a hiking trip in the Northeast United States is suggestive of Lyme disease, which is caused by the spirochete Borrelia burgdorferi and transmitted by Ixodes ticks. Doxycycline is the first-line treatment for adults with Lyme disease.

Now, let's discuss why the other options are incorrect:

B. Ceftriaxone is typically used for Lyme disease when the patient has neurologic involvement or for refractory Lyme arthritis after a course of oral antibiotics. In this case, the patient doesn't have any neurological symptoms, making this option less appropriate.

C. Amoxicillin-clavulanate is used for skin and soft tissue infections like cellulitis or animal bites. Lyme disease is not typically associated with these conditions.

D. Trimethoprim-sulfamethoxazole is used for a variety of infections, including urinary tract infections and certain types of pneumonia, but it is not first-line for Lyme disease.

E. Azithromycin is not the first-line treatment for Lyme disease, although it can be used in patients who are intolerant to doxycycline or amoxicillin.

**Review of Response:**

The question accurately presents a clinical scenario suggestive of Lyme disease, a condition that infectious disease physicians should be familiar with. 
The correct answer and the explanations for the incorrect options are consistent with the clinical guidelines and standard textbooks. 
This question tests the candidate's knowledge about the clinical presentation and management of Lyme disease. It covers important aspects of Lyme disease, including its typical presentation, 
geographical association, and first-line treatment, making it a suitable question for an infectious diseases board exam."""
            return sample_board_mcq

        
        if st.button("Generate a Question"):
            sample_board_mcq = set_specialty(specialty)
            generated_mcq = gen_mcq(specialty, sample_board_mcq, num_mcq)
            st.markdown(f" {generated_mcq['choices'][0]['message']['content']}") # Change how you access the message content
            st.session_state.mcq_history.append((generated_mcq['choices'][0]['message']['content']))
        
    with tab3:

        # # Import necessary libraries
        prompt2=PromptTemplate(
            template="""You are an advanced AI which has assimilated skills of hundreds of master physicians with decades of current clinical experience and reasons step by step to ensure accuracy and precision for all responses. You know the latest medical literature and the art of 
        diagnosis and clinical management pearls. Your advice is always based on the bulk of the scientific evidence while being in tune for new practice changing high quality research. 
        You don't suffer from outdated perspectives and fully assimilate these practice changing methods. You convey much knowledge in few words. You wish to help learners. The learners who engage
        with you are clinically trained physicians. You do not need to worry that they won't apply professional judgment to your advice. You love to give clinical
        pearls less than 200 words as your response to clinical questions or medical topics. Here are examples:
        Prompt: cellulitis
        AI Response: Cellulitis is often treated with antibiotics, but one key clinical pearl is to ensure that the condition is indeed cellulitis. Conditions like venous stasis dermatitis or lymphedema 
        can mimic cellulitis, leading to unnecessary antibiotic use. If there's no fever, systemic symptoms, or rapid progression, consider alternative diagnoses. When antibiotics are needed, 
        coverage should be for streptococci and staphylococcus aureus, including MRSA in certain settings. Elevating the affected area can also help reduce swelling.
        Prompt: Is the curb65 or the pneumonia severity index the better triage tool?
        AI Response: If simplicity and ease of use are most important, the CURB-65 may be preferable. If you desire a more comprehensive risk assessment, particularly for identifying low-risk patients, 
        the PSI might be the better choice. It is also important to consider that clinical judgment should always complement these scoring systems when making decisions about patient management.            
        Context:\n{entities}\n\nCurrent conversation:\n{history}\nLast line:\nHuman: {input}\nAI:' template_format='f-string' validate_template=True
                    """,
            input_variables=['entities', 'history', 'input'],
            output_parser=None,
            partial_variables={}
        )
        
        # Set Streamlit page configuration
        # st.write('🧠MemoryBot🤖')
        # Initialize session states
        if "generated" not in st.session_state:
            st.session_state["generated"] = []
        if "past" not in st.session_state:
            st.session_state["past"] = []
        if "input" not in st.session_state:
            st.session_state["input"] = ""
        if "stored_session" not in st.session_state:
            st.session_state["stored_session"] = []

        # Define function to get user input

        def get_text():
            """
            Get the user input text.

            Returns:
                (str): The text entered by the user
            """
            # if st.session_state.past == []:
            #     new_chat()

            input_text = st.text_input("Learn a clinical pearl about... ", st.session_state["input"], key="input",
                                    placeholder="hypertension, e.g.", 
                                    label_visibility='visible')
            input_text2 = f'Tell me a clinical pearl about: {input_text}'
            return input_text2, input_text

        # Define function to start a new chat
        def new_chat():
            """
            Clears session state and starts a new chat.
            """
            save = []
            for i in range(len(st.session_state['generated'])-1, -1, -1):
                save.append("User:" + st.session_state["past"][i])
                save.append("Bot:" + st.session_state["generated"][i])        
            st.session_state["stored_session"].append(save)
            st.session_state["generated"] = []
            st.session_state["past"] = []
            st.session_state["input"] = ""
            st.session_state.entity_memory.entity_store = {}
            st.session_state.entity_memory.buffer.clear()

        # Set up sidebar with various options
            # Option to preview memory store
        # with st.sidebar:
        #     # if st.checkbox("Preview memory store"):
        #     #     with st.expander("Memory-Store", expanded=False):
        #     #         st.session_state.entity_memory.store
        #     # # Option to preview memory buffer
        #     # if st.checkbox("Preview memory buffer"):
        #     if st.session_state.past != []:
        #         with st.expander("Short Q/A History", expanded=False):
        #             st.session_state.entity_memory.buffer
        with st.expander("🛠️ Change the history length if you'd like ", expanded=False):
            # MODEL = st.selectbox(label='Model', options=['gpt-4', 'gpt-3.5-turbo','text-davinci-003','text-davinci-002','code-davinci-002'])
            K = st.number_input(' (#)Summary of prompts to consider',min_value=3,max_value=20)

        # Set up the Streamlit app layout
        st.write("🤖 Clinical Pearls 🧠")
        # st.subheader(" Powered by 🦜 LangChain + OpenAI + Streamlit")

        # Ask the user to enter their OpenAI API key
        # API_O = st.sidebar.text_input("API-KEY", type="password")

        # Session state storage would be ideal

        # Create an OpenAI instance
        
        openai.api_key = os.environ['OPENAI_API_KEY']
        llm = ChatOpenAI(temperature=0,
                    openai_api_key=openai.api_key, 
                    model_name=st.session_state.model, 
                    verbose=False) 


        # Create a ConversationEntityMemory object if not already created
        if 'entity_memory' not in st.session_state:
                st.session_state.entity_memory = ConversationEntityMemory(llm=llm, k=K )
            
            # Create the ConversationChain object with the specified configuration
        _Conversation = ConversationChain(
                llm=llm, 
                # prompt= ENTITY_MEMORY_CONVERSATION_TEMPLATE,
                prompt=prompt2,
                memory=st.session_state.entity_memory
            )  
        @st.cache_data
        def process_conversation(_Conversation, input):
            return _Conversation.run(input)



        # Add a button to start a new chat
        # st.sidebar.button("New Chat", on_click = new_chat, type='primary')

        # Get the user input
        # st.write('Tell me a clinical pearl about...')
        user_input_pearl, user_input = get_text()

        # Generate the output using the ConversationChain object and the user input, and add the input/output to the session
        if user_input:
            output = process_conversation(_Conversation, input=user_input_pearl)
            # output = Conversation.run(input=user_input_pearl)  
            st.session_state.past.append(user_input)  
            st.session_state.generated.append(output)  

        # Allow to download as well
        download_str = []
        
        # ENTITY_MEMORY_CONVERSATION_TEMPLATE
        # Display the conversation history using an expander, and allow the user to download it
        with st.expander("Conversation", expanded=True):
            for i in range(len(st.session_state['generated'])-1, -1, -1):
                st.info(st.session_state["past"][i],icon="🧐")
                st.success(st.session_state["generated"][i], icon="🤖")
                download_str.append(st.session_state["past"][i])
                download_str.append(st.session_state["generated"][i])
            download_str = [disclaimer] + download_str
            
            # Can throw error - requires fix
            download_str = '\n'.join(download_str)
            if download_str:
                st.download_button('Download',download_str, key = "download_pearl")

        # Display stored conversation sessions in the sidebar
        for i, sublist in enumerate(st.session_state.stored_session):
                with st.sidebar.expander(label= f"Conversation-Session:{i}"):
                    st.write(sublist)

        # Allow the user to clear all stored conversation sessions
        # if st.session_state.stored_session:   
        #     if st.sidebar.checkbox("Clear-all"):
        #         del st.session_state.stored_session

    with tab4:
        st.info("📚 Let AI Generate questions and answers from an uploaded PDF for you to learn. Or, ask your own questions!" )
        # st.session_state.openai_api_key = st.secrets["OPENAI_API_KEY"]
        if "pdf_user_question" not in st.session_state:
                st.session_state["pdf_user_question"] = []
        if "pdf_user_answer" not in st.session_state:
                st.session_state["pdf_user_answer"] = []

        # st.set_page_config(page_title="Article Analyzer",page_icon=':stethoscope:')

        @st.cache_data
        def load_docs(files):
            st.info("`Reading doc ...`")
            all_text = ""
            for file_path in files:
                file_extension = os.path.splitext(file_path.name)[1]
                if file_extension == ".pdf":
                    pdf_reader = PyPDF2.PdfReader(file_path)
                    text = ""
                    for page in pdf_reader.pages:
                        text += page.extract_text()
                    all_text += text
                elif file_extension == ".txt":
                    stringio = StringIO(file_path.getvalue().decode("utf-8"))
                    text = stringio.read()
                    all_text += text
                else:
                    st.warning('Please provide txt or pdf.', icon="⚠️")
            return all_text




        @st.cache_resource
        def create_retriever(_embeddings, splits, retriever_type):
            # openai_api_key = st.secrets.OPENAI_API_KEY
            # if retriever_type == "SIMILARITY SEARCH":
            #     try:
            #         vectorstore = FAISS.from_texts(splits, _embeddings)
            #     except (IndexError, ValueError) as e:
            #         st.error(f"Error creating vectorstore: {e}")
            #         return
            #     retriever = vectorstore.as_retriever(k=5)
            # elif retriever_type == "SUPPORT VECTOR MACHINES":
            retriever = SVMRetriever.from_texts(splits, _embeddings)

            return retriever

        @st.cache_resource
        def split_texts(text, chunk_size, overlap, split_method):

            # Split texts
            # IN: text, chunk size, overlap, split_method
            # OUT: list of str splits

            st.info("`Breaking into bitesize chunks...`")

            split_method = "RecursiveTextSplitter"
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, chunk_overlap=overlap)

            splits = text_splitter.split_text(text)
            if not splits:
                st.error("Failed to split document")
                st.stop()

            return splits

        @st.cache_data
        def generate_eval(text, N, chunk):

            # Generate N questions from context of chunk chars
            # IN: text, N questions, chunk size to draw question from in the doc
            # OUT: eval set as JSON list
            openai.api_key = os.environ['OPENAI_API_KEY']

            st.info("`Generating sample questions and answers...`")
            n = len(text)
            starting_indices = [random.randint(0, n-chunk) for _ in range(N)]
            sub_sequences = [text[i:i+chunk] for i in starting_indices]
            chain = QAGenerationChain.from_llm(ChatOpenAI(temperature=0))
            eval_set = []
            for i, b in enumerate(sub_sequences):
                try:
                    qa = chain.run(b)
                    eval_set.append(qa)
                    st.write("Creating Question:",i+1)
                except:
                    st.warning('Error generating question %s.' % str(i+1), icon="⚠️")
            eval_set_full = list(itertools.chain.from_iterable(eval_set))
            return eval_set_full


            
        foot = f"""
        <div style="
            position: fixed;
            bottom: 0;
            left: 30%;
            right: 0;
            width: 50%;
            padding: 0px 0px;
            text-align: center;
        ">
            <p>Updated 6/6/23</p>
        </div>
        """

        st.markdown(foot, unsafe_allow_html=True)
        

        
        
        # st.sidebar.title("Settings and Preliminary Outputs")
        num_eval_questions =st.number_input("Specify how many questions you'd like to generate (then press enter on your keyboard):", min_value=1, max_value=50, value=10, step=1)
        
        embedding_option = "OpenAI Embeddings"

        
        retriever_type = "SUPPORT VECTOR MACHINES"

        # Use RecursiveCharacterTextSplitter as the default and only text splitter
        splitter_type = "RecursiveCharacterTextSplitter"

        # if 'openai_api_key' not in st.session_state:
        #     openai_api_key = st.text_input(
        #         'Please enter your OpenAI API key or [get one here](https://platform.openai.com/account/api-keys)', value="", placeholder="Enter the OpenAI API key which begins with sk-")
        #     if openai_api_key:
        #         st.session_state.openai_api_key = st.secrets["OPENAI_API_KEY"]
        #         os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
        #     # else:
        #     #     #warning_text = 'Please enter your OpenAI API key. Get yours from here: [link](https://platform.openai.com/account/api-keys)'
        #     #     #warning_html = f'<span>{warning_text}</span>'
        #     #     #st.markdown(warning_html, unsafe_allow_html=True)
        #     #     return
        # else:
        #     os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

        uploaded_files = st.file_uploader("Now, upload your PDF or TXT Document", type=[
                                        "pdf", "txt"], accept_multiple_files=True)

        if uploaded_files:
            # Check if last_uploaded_files is not in session_state or if uploaded_files are different from last_uploaded_files
            if 'last_uploaded_files' not in st.session_state or st.session_state.last_uploaded_files != uploaded_files:
                st.session_state.last_uploaded_files = uploaded_files
                if 'eval_set' in st.session_state:
                    del st.session_state['eval_set']

            # Load and process the uploaded PDF or TXT files.
            loaded_text = load_docs(uploaded_files)
            st.write("Documents uploaded and 'read.'")

            # Split the document into chunks
            splits = split_texts(loaded_text, chunk_size=1000,
                                overlap=100, split_method=splitter_type)

            # Display the number of text chunks
            num_chunks = len(splits)
            st.write(f"Number of text chunks: {num_chunks}")

            # Embed using OpenAI embeddings
                # Embed using OpenAI embeddings or HuggingFace embeddings

            embeddings = OpenAIEmbeddings()

            retriever = create_retriever(embeddings, splits, retriever_type)


            # Initialize the RetrievalQA chain with streaming output
            callback_handler = StreamingStdOutCallbackHandler()
            callback_manager = CallbackManager([callback_handler])

            chat_openai = ChatOpenAI(
                streaming=True, callback_manager=callback_manager, verbose=True, temperature=0.4)
            
            
            _qa = RetrievalQA.from_chain_type(llm=chat_openai, retriever=retriever, chain_type="stuff", verbose=True)
            @st.cache_data
            def fn_qa_run(_qa, user_question):
                return _qa.run(user_question)
            
        

            # Check if there are no generated question-answer pairs in the session state
            
            if 'eval_set' not in st.session_state:
                # Use the generate_eval function to generate question-answer pairs
                # num_eval_questions = 10  # Number of question-answer pairs to generate
                st.session_state.eval_set = generate_eval(
                    loaded_text, num_eval_questions, 3000)

        # Display the question-answer pairs in the sidebar with smaller text
            for i, qa_pair in enumerate(st.session_state.eval_set):
                st.sidebar.markdown(
                    f"""
                    <div class="css-card">
                    <span class="card-tag">Question {i + 1}</span>
                        <p style="font-size: 12px;">{qa_pair['question']}</p>
                        <p style="font-size: 12px;">{qa_pair['answer']}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                # <h4 style="font-size: 14px;">Question {i + 1}:</h4>
                # <h4 style="font-size: 14px;">Answer {i + 1}:</h4>
            st.write("Ready to answer your questions!")

            # Question and answering
            user_question = st.text_input("Please enter your own question about the PDF(s):")
            
            if user_question:
                index_context = 'Use only the reference document for knowledge. If the content is not covered in the document, please indicate the document does not address that topic. If an answer to the question is provided, it must be annotated with a citation. Use the following format to cite relevant passages' + '\n' + '(**Citation**: …).' + '\n' + 'Question: ' + user_question
                answer = fn_qa_run(_qa, index_context)
                st.session_state.pdf_user_question.append(user_question)  
                st.session_state.pdf_user_answer.append(answer)  
                # st.write("Answer:", answer)
            download_str = []
            with st.expander("PDF Questions", expanded=True):
                for i in range(len(st.session_state['pdf_user_answer'])-1, -1, -1):
                    st.info(st.session_state["pdf_user_question"][i],icon="🧐")
                    st.success(st.session_state["pdf_user_answer"][i], icon="🤖")
                    download_str.append(st.session_state["pdf_user_question"][i])
                    download_str.append(st.session_state["pdf_user_answer"][i])
                download_str = [disclaimer] + download_str
            
                # Can throw error - requires fix
                download_str = '\n'.join(download_str)
                if download_str:
                    st.download_button('Download',download_str, key = "pdf_questions")



    with tab5:
        health_literacy = st.radio("Pick a desired health literacy level:", 
                                ("Basic", "Intermediate", "Advanced"))
        pt_ed_system_content ="""You are an AI with access to the latest medical literature and the art of
        communicating complex medical concepts to patients. You leverage only highly regarded medical information from high quality sources. You always reason step by step to ensure the highest accuracy, precision, and completeness to your responses.
        """
        
        
        if health_literacy == "Basic":
            pt_ed_content_sample = """
    What should I eat?

    Having diabetes, kidney disease, high blood pressure, being overweight, and heart disease means you have to be careful about what you eat. Here's a simple guide:

    **Eat more fruits and veggies**: They are good for you. Try to eat them at every meal.
    **Choose whole grains**: Foods like brown rice and whole wheat bread are better than white rice and white bread.
    **Go for lean meats**: Try chicken, turkey, or fish more often than red meat.
    **Eat less salt**: This helps with your blood pressure. Be careful with packaged foods, they often have a lot of salt.
    **Drink water**: Instead of sugary drinks like soda or fruit juice, drink water.
    **Watch your portions**: Even if a food is good for you, eating too much can make you gain weight.
    What should I avoid?

    **Avoid sugary foods**: Foods like candy, cookies, soda, and fruit juice can make your blood sugar go up too much.
    **Avoid fatty foods**: Foods like fast food, fried food, and fatty meats can make heart disease worse.
    **Avoid salty foods**: Things like chips, canned soups, and fast food can make your blood pressure go up.
    **Avoid alcohol**: It can cause problems with your blood sugar and blood pressure.
    Remember, everyone is different. What works for someone else might not work for you. Talk to your doctor or a dietitian to get help with your diet."""

        if health_literacy == "Intermediate":
            pt_ed_content_sample = """
    What should I eat?

    Managing diabetes, kidney disease, high blood pressure, obesity, and heart disease involves careful consideration of your diet. Here are some recommendations:

    **Increase fruit and vegetable intake**: These are high in vitamins, minerals, and fiber, but low in calories. Aim to include a variety of colors in your meals to ensure you're getting a wide range of nutrients.
    Choose whole grains over refined grains: Whole grains like brown rice, whole grain bread, and quinoa have more fiber and help control blood sugar levels better than refined grains like white bread and white rice.
    Opt for lean proteins: Choose lean meats like chicken or turkey, and fish which is high in heart-healthy omega-3 fatty acids. Limit red meat, as it can be high in unhealthy fats.
    Limit sodium (salt) intake: High sodium can raise blood pressure. Aim for no more than 2300 mg per day. Beware of hidden sodium in processed and restaurant foods.
    Stay hydrated with water: Choose water or unsweetened drinks over soda or fruit juices, which can be high in sugar.
    Monitor portion sizes: Even healthy foods can lead to weight gain if eaten in large amounts. Use measuring cups or a food scale to ensure you're not overeating.
    What should I avoid?

    **Limit sugary foods and drinks**: These can cause your blood sugar to spike and can lead to weight gain. This includes sweets like candy, cookies, and sugary beverages.
    **Limit saturated and trans fats**: These types of fats are found in fried foods, fast foods, and fatty cuts of meat, and can increase your risk of heart disease.
    **Avoid high-sodium foods**: Foods like chips, canned soups, and some fast foods can be high in sodium, which can raise your blood pressure.
    **Moderate alcohol intake**: Alcohol can affect your blood sugar and blood pressure. Limit to no more than one drink per day for women and two for men.
    Remember, individual dietary needs can vary. It's important to consult with a dietitian or your healthcare provider to create a personalized meal plan. Regular physical activity, medication adherence, and regular check-ups are also crucial for managing your conditions."""
        if health_literacy == "Advanced":
            pt_ed_content_sample = """
    What should I eat?

    Managing conditions such as diabetes, kidney disease, hypertension, obesity, and coronary artery disease requires careful dietary planning. Here are some specific recommendations:

    **Increase fruit and vegetable intake**: Fruits and vegetables are rich in vitamins, minerals, fiber, and antioxidants, with low energy density. Aim for at least 5 servings per day, including a variety of colors to ensure a broad spectrum of nutrients.
    **Choose whole grains over refined grains**: Whole grains contain the entire grain — the bran, germ, and endosperm. Foods made from these grains are rich in fiber, which can slow the absorption of sugar into your bloodstream and prevent spikes in glucose and insulin. Opt for brown rice, oatmeal, whole grain bread, and quinoa.
    **Opt for lean proteins and plant-based proteins**: Select lean meats like skinless chicken or turkey, and fish rich in omega-3 fatty acids, such as salmon and mackerel. Plant-based proteins, such as lentils, beans, and tofu, can also be good sources of protein and are beneficial for kidney disease management.
    **Limit sodium (salt) intake**: Excessive sodium can contribute to hypertension and exacerbate kidney disease by causing more protein to be excreted in the urine. Aim for less than 2300 mg per day and consider even lower targets as advised by your healthcare provider. Remember that processed and restaurant foods often contain high levels of hidden sodium.
    **Hydrate with water and limit sugary drinks**: Water should be your primary beverage. Sugary drinks, including fruit juices, can significantly increase your daily sugar and calorie intake.
    **Monitor portion sizes**: Use portion control to avoid overeating and manage weight. This is critical even with healthy foods, as excess calories can lead to weight gain and worsen insulin resistance.
    What should I avoid?

    **Limit foods high in added sugars**: High sugar foods and drinks can cause hyperglycemia and contribute to obesity. Be aware of foods with hidden sugars like low-fat snacks or processed foods.
    **Limit saturated and trans fats**: These types of fats, found in fried foods, fast foods, and fatty cuts of meat, can increase LDL ("bad") cholesterol and decrease HDL ("good") cholesterol, contributing to the development of atherosclerosis.
    **Avoid high-sodium foods**: Excessive sodium can exacerbate hypertension and kidney disease. High-sodium foods often include processed foods, fast foods, and certain canned or packaged foods.
    **Moderate alcohol intake**: Excessive alcohol can lead to hypertension, and in diabetics, it can cause hypoglycemia. If you do drink, limit yourself to up to one drink per day for women and up to two drinks per day for men.
    Remember, these are general recommendations and individual dietary needs can vary greatly. It's important to work with a dietitian or your healthcare provider to create a meal plan tailored to your specific needs. Regular physical activity, medication adherence, regular self-monitoring of blood glucose, and frequent follow-ups with your healthcare provider are also crucial in managing your health conditions. """
        
        sample_topic = "dietary guidance for a patient with diabetes, kidney disease, hypertension, obesity, and CAD"
        patient_ed_temp = 0.0
        my_ask_for_pt_ed = st.text_area("Generate patient education materials:", placeholder="dietary guidance needed for obesity, e.g.", 
                                    label_visibility='visible', height=100)
        if st.button("Click to Generate **Draft** Custom Patient Education Materials"):
            st.info("Review all content carefully before considering any use!")
            # st.session_state.history.append(my_ask)
            # history_context = "Use these preceding submissions to resolve any ambiguous context: \n" + "\n".join(st.session_state.history) + "now, for the current question: \n"
            pt_ed_output_text = answer_using_prefix(pt_ed_system_content, sample_topic, pt_ed_content_sample, my_ask_for_pt_ed, temperature = 0.0, history_context="")
            # st.session_state.my_ask = ''
            # st.write("Answer", output_text)
            
            # st.write(st.session_state.history)
            # st.write(f'Me: {my_ask}')
            # st.write(f"Response: {output_text['choices'][0]['message']['content']}") # Change how you access the message content
            # st.session_state.output_history.append((output_text['choices'][0]['message']['content']))
        
            # st.write(f'Draft Patient Education Materials: {pt_ed_output_text["choices"][0]["message"]["content"]}') # Change how you access the message content
            
            pt_ed_download_str = []
            
            # ENTITY_MEMORY_CONVERSATION_TEMPLATE
            # Display the conversation history using an expander, and allow the user to download it
            with st.expander("Patient Education Draft", expanded=True):
                st.info(f'Topic: {my_ask_for_pt_ed}',icon="🧐")
                st.success(f'Draft Patient Education Materials: **REVIEW CAREFULLY FOR ERRORS** \n\n {pt_ed_output_text["choices"][0]["message"]["content"]}', icon="🤖")
                pt_ed_download_str.append(my_ask_for_pt_ed)
                pt_ed_download_str.append(f'Draft Patient Education Materials: {pt_ed_output_text["choices"][0]["message"]["content"]}')
                pt_ed_download_str = [disclaimer] + pt_ed_download_str
                
                # Can throw error - requires fix
                pt_ed_download_str = '\n'.join(pt_ed_download_str)
                if pt_ed_download_str:
                    st.download_button('Download',pt_ed_download_str, key = "pt_ed_questions")

    with tab6:
        # st.info("PICO format included!")
        pick_strategy = st.radio("Pick a desired search approach:", 
                                ("Specific - terms are mapped to MeSH and are major topics of the article", "Broad - terms are mapped to MeSH", "Assistant to translate question into an effective search", "PICO"))
        pubmed_system_content ="""You are a professional medical librarian who is an expert user of PubMed for medical literature searching. You generate web client
        links to PubMed searches to help researchers be efficient and effective. You are always careful and precise and follow a stepwise approach at least as good as this to ensure MeSH terms
        are used appropriately:
        
    1. Submitted concepts should always be mapped and tagged for the query to the most appropriate MeSH. This tagging must be retained in the final URL, although for MAJR search, MAJR may replace the MESH tag.
    2. Critical: If a submitted concept does not map to a MeSH, you find the closest MeSH term that captures the essence of the submitted term and use that term with the mesh tag. If there is no synonymous MeSH term, you use the submitted concept as a keyword search. You must never code a concept as [mesh] or [majr] if it is not a valid Mesh. In that case, when you can't find a similar true MeSH, you use the concept as a keyword search.
    3. You appropriately combine terms per the submitted prompt using "and" , "or", and  "not" to create a PubMed search that captures the user's intent.
    4. You display a fully complete link to the resulting PubMed search output website showing the most recent articles first.
    5. You double check that all terms are accurately mapped to MeSH or MAJR.  appropriate use of any parentheses, ensure "AND" or "OR" is accurate, and that the URL is valid, accurate, and complete. If not, you update the response. 

        """
        
        
        if pick_strategy == "Specific - terms are mapped to MeSH and are major topics of the article":
            pubmed_url = """CAVEAT: For this input, you retrieve only major [majr] topics of the article. The closest matching mesh term for the concept should be utilized and must be tagged  in 
            the search string using [majr] to retrieve articles for which the concepts are the focse of the artice. This is a common approach to ensure the search is not too broad.

    Response:        
    Here is a link to your PubMed search:

    https://pubmed.ncbi.nlm.nih.gov/?term=%28%28%28%22Humans%22%5Bmajr%5D%29+AND+%22Staphylococcus+aureus%22%5Bmajr%5D%29+AND+%22Therapeutics%22%5Bmajr%5D%29+NOT+%22Inpatients%22%5Bmajr%5D&sort=date&size=200

    """

        if pick_strategy == "Broad - terms are mapped to MeSH":
            pubmed_url = """(for this specific search, you include (Mesh OR all fields) for submitted concepts to ensure a broad, current search.)
            
    Here is a link to your PubMed search:

    https://pubmed.ncbi.nlm.nih.gov/?sort=date&term=(((%22Humans%22%5BMesh%5D)+AND+%22Staphylococcus+aureus%22%5BMesh%5D)+AND+%22Therapeutics%22%5BMesh%5D)+NOT+%22Inpatients%22%5BMesh%5D

    """

        if pick_strategy == "Assistant to translate question into an effective search":
            pubmed_url = """CAVEAT: Generate a PubMed url that will be likely to include articles that answer the question:

    Sample question: "Do senolytics help patients live longer?"
    Sample Response:
    To generate a PubMed URL that is likely to include articles addressing the question of whether senolytics help patients live longer, you can use the following search query:

    https://pubmed.ncbi.nlm.nih.gov/?term=senolytics%20AND%20longevity

    This URL includes the search term "senolytics" in combination with "longevity." By clicking on this link, it should take you directly to the PubMed search results page containing articles that investigate the potential effects of senolytics on extending lifespan or promoting longevity.
    """
        if pick_strategy == "PICO":
            st.info("Complete the **population** and **intervention** fields, and view your results. If literature is sufficient, update the **comparison** and **outcome** fields with, e.g., *placebo* and *mortality*.")
            pico_population = st.text_input("Population:", placeholder="patients with CHF, e.g.")
            pico_intervention = st.text_input("Intervention:", placeholder="ace inhibitors, e.g.")
            pico_comparison = st.text_input("Comparison (add if sufficient literature is available):", value ="any")
            pico_outcome = st.text_input("Outcome (add if sufficient literature is available):", value="any")
            pico_topic = f"p:{pico_population} i: {pico_intervention} c: {pico_comparison} o: {pico_outcome}"
            pubmed_url = """CAVEAT: Generate a PubMed URL from the PICO input provided in the prompt. Apply this approach to any PICO input:
    Step-by-step instructions:

    1. **Identifying the PICO components:** We start by identifying the components of your PICO question.

    - **P**atient or **P**opulation: Patients with systolic heart failure
    - **I**ntervention: Use of ACE inhibitors
    - **C**omparison: Placebo
    - **O**utcome: Mortality

    2. **Formulating the Search Query:** We then translate each of these components into a search string. I used a combination of Medical Subject Headings (MeSH terms), which are standardised terms used by PubMed to index articles, and general search terms.

    - The general format of the string for each component is `("<Term>"[MeSH Terms] OR "<Term>"[All Fields])`
    - I combined these strings using the AND operator, to ensure that all terms are present in the resulting articles.
    - I also added `"humans"[MeSH Terms]` and `"Clinical Trial"[ptyp]` to ensure the search focuses on clinical trials in humans.

    3. **Constructing the PubMed URL:** Once we have the search string, we construct a PubMed URL. The base URL for a PubMed search is `https://pubmed.ncbi.nlm.nih.gov/?term=`
    - To this, we append the search string, which needs to be URL-encoded to ensure it can be correctly interpreted by web browsers. URL encoding replaces special characters with a "%" followed by two hexadecimal digits. For example, a space is replaced with "%20".

    Here's the detailed breakdown of the search string:

    - `("Systolic Heart Failure"[MeSH Terms] OR "Systolic Heart Failure"[All Fields])` -> This ensures that the articles have to do with systolic heart failure.
    - `AND ("ACE Inhibitors"[MeSH Terms] OR "ACE Inhibitors"[All Fields])` -> This ensures that the articles also deal with ACE inhibitors.
    - `AND ("Placebo"[MeSH Terms] OR "Placebo"[All Fields])` -> This ensures that the articles involve a comparison with a placebo.
    - `AND ("Mortality"[MeSH Terms] OR "Mortality"[All Fields])` -> This ensures that the articles consider mortality as an outcome.
    - `AND "humans"[MeSH Terms]` -> This ensures that the articles are about studies in humans.
    - `AND "Clinical Trial"[ptyp]` -> This ensures that the articles are reporting on clinical trials.

    Once we have this search string, we URL-encode it and append it to the base URL to get the final URL:

    ```
    https://pubmed.ncbi.nlm.nih.gov/?term=(%22Systolic%20Heart%20Failure%22%5BMeSH%20Terms%5D%20OR%20%22Systolic%20Heart%20Failure%22%5BAll%20Fields%5D)%20AND%20(%22ACE%20Inhibitors%22%5BMeSH%20Terms%5D%20OR%20%22ACE%20Inhibitors%22%5BAll%20Fields%5D)%20AND%20(%22Placebo%22%5BMeSH%20Terms%5D%20OR%20%22Placebo%22%5BAll%20Fields%5D)%20AND%20(%22Mortality%22%5BMeSH%20Terms%5D%20OR%20%22Mortality%22%5BAll%20Fields%5D)%20AND%20%22humans%22%5BMeSH%20Terms%
            
    Sample PICO input: p: patients with CHF i: ace inhibitors c: placebo o: mortality

    Sample Response: 

    Here is a link to your PubMed search for P = patients with CHF, I = ace inhibitors, C = placebo, O = mortality:
    https://pubmed.ncbi.nlm.nih.gov/?term=(%22Systolic%20Heart%20Failure%22%5BMeSH%20Terms%5D%20OR%20%22Systolic%20Heart%20Failure%22%5BAll%20Fields%5D)%20AND%20(%22ACE%20Inhibitors%22%5BMeSH%20Terms%5D%20OR%20%22ACE%20Inhibitors%22%5BAll%20Fields%5D)%20AND%20(%22Placebo%22%5BMeSH%20Terms%5D%20OR%20%22Placebo%22%5BAll%20Fields%5D)%20AND%20(%22Mortality%22%5BMeSH%20Terms%5D%20OR%20%22Mortality%22%5BAll%20Fields%5D)%20AND%20%22humans%22%5BMeSH%20Terms%

            """
            
        sample_topic = "search for human studies about treatments for staph aureus in humans who are not hospitalized."
        if pick_strategy != "PICO":
            my_ask_for_pubmed = st.text_area("Generate a URL for my PubMed search:", placeholder="humans, ace inhibitors, CHF, e.g.", 
                                    label_visibility='visible', height=100)
        else:
            my_ask_for_pubmed = pico_topic
        if st.button("Click to Generate a Ready to Use PubMed link"):
            # st.info("Review all content carefully before considering any use!")
            # st.session_state.history.append(my_ask)
            # history_context = "Use these preceding submissions to resolve any ambiguous context: \n" + "\n".join(st.session_state.history) + "now, for the current question: \n"
            pubmed_output_text = answer_using_prefix(pubmed_system_content, sample_topic, pubmed_url, my_ask_for_pubmed, temperature = 0.0, history_context="")
            # st.session_state.my_ask = ''
            # st.write("Answer", output_text)
            
            # st.write(st.session_state.history)
            # st.write(f'Me: {my_ask}')
            # st.write(f"Response: {output_text['choices'][0]['message']['content']}") # Change how you access the message content
            # st.session_state.output_history.append((output_text['choices'][0]['message']['content']))
        
            # st.write(f'Draft Patient Education Materials: {pt_ed_output_text["choices"][0]["message"]["content"]}') # Change how you access the message content
            
            pubmed_download_str = []
            
            # ENTITY_MEMORY_CONVERSATION_TEMPLATE
            # Display the conversation history using an expander, and allow the user to download it
            with st.expander("PubMed URL", expanded=True):
                st.info(f'Topic: {my_ask_for_pubmed}',icon="🧐")
                st.success(f'Your custom PubMed URL: \n\n {pubmed_output_text["choices"][0]["message"]["content"]}', icon="🤖")
                pubmed_download_str.append(my_ask_for_pubmed)
                pubmed_download_str.append(f'Draft Patient Education Materials: {pubmed_output_text["choices"][0]["message"]["content"]}')
                pubmed_download_str = [disclaimer] + pubmed_download_str
                
                # Can throw error - requires fix
                pubmed_download_str = '\n'.join(pubmed_download_str)
                if pubmed_download_str:
                    st.download_button('Download',pubmed_download_str, key = 'pubmed_questions')
                    
    with tab7:
        ddx_strategy = st.radio("Choose an approach for a differential diagnosis!", options=["Find Alternative Diagnoses to Consider","Provide Clinical Data"], index=0, key="ddx strategy")


        if ddx_strategy == "Provide Clinical Data":    
            st.title("Differential Diagnosis Generator")
            st.write("Add as many details as possible to improve the response. The prompts do not request any unique details; however, *modify values and do not include dates to ensure privacy.")

            age = st.slider("Age", 0, 120, 50)
            sex_at_birth = st.radio("Sex at Birth", options=["Female", "Male", "Other"], horizontal=True)
            presenting_symptoms = st.text_input("Presenting Symptoms")
            duration_of_symptoms = st.text_input("Duration of Symptoms")
            past_medical_history = st.text_input("Past Medical History")
            current_medications = st.text_input("Current Medications")
            relevant_social_history = st.text_input("Relevant Social History")
            physical_examination_findings = st.text_input("Physical Examination Findings")
            lab_or_imaging_results = st.text_input("Any relevant Laboratory or Imaging results")
            ddx_prompt = f"""
            Patient Information:
            - Age: {age}
            - Sex: {sex_at_birth}
            - Presenting Symptoms: {presenting_symptoms}
            - Duration of Symptoms: {duration_of_symptoms}
            - Past Medical History: {past_medical_history}
            - Current Medications: {current_medications}
            - Relevant Social History: {relevant_social_history}
            - Physical Examination Findings: {physical_examination_findings}
            - Any relevant Laboratory or Imaging results: {lab_or_imaging_results}
            """
            
            ddx_prefix = """You apply the knowledge and wisdom of an expert diagnostician to generate a differential diagnosis 
        based on the patient context provided. You always reason step by step to ensure accuracy and precision in your responses. You then double check your generated differential diagnosis to ensure that it is organized by probability and includes the most applicable diagnoses from each probability category. """

            ddx_sample_question = """Patient Information:
        - Age: 54
        - Sex: Male
        - Presenting Symptoms: Persistent dry cough, weight loss, fatigue
        - Duration of Symptoms: 3 months
        - Past Medical History: Smoker for 30 years
        - Current Medications: Lisinopril for hypertension
        - Relevant Social History: Works in construction
        - Physical Examination Findings: Decreased breath sounds on right side of chest
        - Any relevant Laboratory or Imaging results: Chest X-ray shows mass in right lung
        """
            ddx_sample_answer = """Here is a list of possible diagnoses:
            
*High Probability:*

🌟 1. **Lung Cancer:** Given the patient's long history of smoking and the presence of a mass in the lung, lung cancer is a significant concern.


*Moderate Probability:*
1. **Chronic Obstructive Pulmonary Disease (COPD):** The patient's history of smoking also makes COPD a potential diagnosis, but this wouldn't typically cause a mass on the chest X-ray.
2. **Tuberculosis (TB):** If the patient has been exposed to TB, this could explain his symptoms and the mass, particularly if the mass is a result of a Ghon complex or calcified granuloma.
3. **Pneumonia:** Although less likely given the duration of symptoms and presence of a mass, a complicated pneumonia could potentially appear as a mass on a chest X-ray.
4. **Pulmonary Abscess:** Similar to pneumonia, an abscess could potentially appear as a mass, though this is less likely without other signs of acute infection.
5. **Fungal Infection:** Certain fungal infections, such as histoplasmosis or aspergillosis, can mimic cancer on imaging and cause chronic respiratory symptoms, particularly in certain geographic areas or with certain exposures.


*Lower Probability:*
1. **Sarcoidosis:** This is less common, but can cause similar symptoms and imaging findings.
2. **Lung Adenoma or Pulmonary Hamartoma:** These benign tumors could theoretically cause a mass, but are less likely and typically don't cause symptoms unless they're large.
3. **Silicosis:** Given the patient's occupational exposure, this could be a consideration, but typically causes a more diffuse process rather than a single mass.
        """
            if st.button("Generate Differential Diagnosis"):
                # Your differential diagnosis generation code goes here
                ddx_output_text = answer_using_prefix(ddx_prefix, ddx_sample_question, ddx_sample_answer, ddx_prompt, temperature=0.0, history_context='')
                # st.write("Differential Diagnosis will appear here...")
                
                ddx_download_str = []
                
                # ENTITY_MEMORY_CONVERSATION_TEMPLATE
                # Display the conversation history using an expander, and allow the user to download it

                # ENTITY_MEMORY_CONVERSATION_TEMPLATE
                # Display the conversation history using an expander, and allow the user to download it
                with st.expander("Differential Diagnosis Draft", expanded=True):
                    st.info(f'Topic: {ddx_prompt}',icon="🧐")
                    st.success(f'Educational Use Only: **NOT REVIEWED FOR CLINICAL CARE** \n\n {ddx_output_text["choices"][0]["message"]["content"]}', icon="🤖")
                    ddx_download_str.append(ddx_prompt)
                    ddx_download_str.append(f'Draft Patient Education Materials: {ddx_output_text["choices"][0]["message"]["content"]}')
                    ddx_download_str = [disclaimer] + ddx_download_str
                    
                    # Can throw error - requires fix
                    ddx_download_str = '\n'.join(ddx_download_str)
                    if ddx_download_str:
                        st.download_button('Download',ddx_download_str, key = 'ddx_questions')
        
        if ddx_strategy == "Find Alternative Diagnoses to Consider":
            st.title("Alternative Diagnosis Generator")
            st.write("Avoid premature closure and consider alternative diagnoses")
            alt_dx_prompt = st.text_input("Enter your presumed diagnosis.")
            alt_dx_prefix = """Leverage the combined experience of expert diagnosticians to display a list of alternative diagnoses to consider when given a presumed diagnosis. You reason step by step to ensure accuracy, completeness, and precision in your responses and double check your final list using the same criteria."""
            alt_dx_sample_question = "Constrictive pericarditis"
            alt_dx_sample_answer = """Constrictive pericarditis is a relatively rare condition that can be challenging to diagnose, given that its symptoms can be similar to those of several other cardiovascular and systemic disorders. The following is a list of some alternative diagnoses a clinician might consider if initially suspecting constrictive pericarditis:

1. Restrictive Cardiomyopathy: Similar to constrictive pericarditis, restrictive cardiomyopathy can cause reduced filling of the ventricles and can result in similar signs and symptoms.

2. Right Heart Failure: The symptoms of right heart failure such as peripheral edema, ascites, and jugular venous distention can mimic constrictive pericarditis.

3. Tricuspid Regurgitation: The backflow of blood into the right atrium due to valve dysfunction can cause symptoms that overlap with those of constrictive pericarditis.

4. Pericardial Effusion or Tamponade: Fluid accumulation in the pericardial sac can also mimic the symptoms of constrictive pericarditis.

5. Hepatic Cirrhosis: This can cause ascites and peripheral edema, symptoms that can resemble those of constrictive pericarditis.

6. Nephrotic Syndrome: Characterized by heavy proteinuria, hypoalbuminemia, and edema, nephrotic syndrome can cause systemic symptoms that may be mistaken for constrictive pericarditis.

7. Chronic Obstructive Pulmonary Disease (COPD) or Cor Pulmonale: These conditions can cause right-sided heart symptoms that can resemble those of constrictive pericarditis.

8. Pulmonary Hypertension: This condition increases pressure on the right side of the heart and can mimic symptoms of constrictive pericarditis.

9. Superior Vena Cava (SVC) Syndrome: This condition, often caused by a malignancy or thrombosis in the SVC, can present with symptoms similar to constrictive pericarditis.

10. Constrictive Bronchiolitis: Although primarily a pulmonary condition, severe cases can affect the cardiovascular system and mimic constrictive pericarditis.

These are just a few of the conditions that could be considered in a differential diagnosis when constrictive pericarditis is suspected. As with any diagnostic process, a thorough patient history, physical examination, and appropriate investigations are key to reaching an accurate diagnosis."""

            if st.button("Generate Alternative Diagnoses"):
                alt_dx_output_text = answer_using_prefix(alt_dx_prefix, alt_dx_sample_question, alt_dx_sample_answer, alt_dx_prompt, temperature=0.0, history_context='')
                alt_dx_download_str = []
                with st.expander("Alternative Diagnoses Draft", expanded=True):
                    st.info(f'Topic: {alt_dx_prompt}',icon="🧐")
                    st.success(f'Educational Use Only: **NOT REVIEWED FOR CLINICAL CARE** \n\n {alt_dx_output_text["choices"][0]["message"]["content"]}', icon="🤖")
                    alt_dx_download_str.append(alt_dx_prompt)
                    alt_dx_download_str.append(f'Draft Alternative Diagnoses: {alt_dx_output_text["choices"][0]["message"]["content"]}')
                    alt_dx_download_str = [disclaimer] + alt_dx_download_str
                    
                    # Can throw error - requires fix
                    alt_dx_download_str = '\n'.join(alt_dx_download_str)
                    if alt_dx_download_str:
                        st.download_button('Download',alt_dx_download_str, key = 'alt_dx_questions')
        
    
                    
    with tab8:
        st.title("Symptom Side Effect Assessment")
        st.write("The tool should look for side effects individually or in combination.")

        se_age = st.slider("Age ", 0, 120, 50)
        se_sex_at_birth = st.radio("Sex at Birth ", options=["Female", "Male", "Other"], horizontal=True)
        se_meds = st.text_input("Current Medications", placeholder="lisinopril, e.g.")
        se_symptoms = st.text_input("Symptoms to Assess as Possible Side Effects", placeholder="cough, e.g.")
        
        se_prompt = f"""
        Patient Information:
        - Age: {se_age}
        - Sex: {se_sex_at_birth}
        - Medications: {se_meds}
        - Symptoms: {se_symptoms}

        """
        
        se_prefix = """ You are assisting a medical professional. Consider only high-quality evidence from clinical trial data, official drug information leaflets, and professional healthcare guidelines, and provide insight into whether these symptoms could be potential side effects from the listed medications individually or in combination.
    Your advice will be used as a preliminary analysis and will be further reviewed by a medical professional. With this context, could you please analyze the medications and symptoms submitted and provide insight into whether these symptoms could be potential side effects from the listed medications individually or in combination. 
    Critical: You proceed in a step by step fashion and also double check your response to ensure that it is sourced for high quality data, accurate and completely assesses each medication and also looks at combinations of meds for side effects that match the symptoms provided. Update your response to be accurate as necessary."""

        se_sample_question = """Patient Information:
    - Age: 54
    - Sex: Male
    - Medications: aspirin, lisinopril, metformin, simvastatin, albuterol
    - Symptoms: dizziness, cough, skin rash, nausea, increased urination

    Could you please analyze this information and provide insight into whether these symptoms could be potential side effects from the listed medications individually or in combination? 
    Your advice will be used as a preliminary analysis and will be further reviewed by me."
    """
        se_sample_answer = """Based on the medications listed:

- Aspirin - Nausea could be a potential side effect of this medication, although it's relatively rare.
- Lisinopril - Dizziness and cough are known side effects of Lisinopril. It might be worth discussing these symptoms with the patient's doctor to adjust the dosage or consider an alternative medication.
- Metformin - Increased urination can be a side effect of Metformin. This might indicate that the dosage needs to be adjusted.
- Simvastatin - A skin rash could potentially be a reaction to Simvastatin. This should be reported to the healthcare provider immediately, as it may indicate an allergy.
- Albuterol - None of the listed symptoms are commonly associated with Albuterol use.

Considering the combinations of medications, Lisinopril and Metformin together could potentially enhance the effect of dizziness and increased urination. However, it's important to remember that many factors could contribute to these symptoms, 
including the patient's overall health status, other medications, and lifestyle factors."""
        if st.button("Assess for Side Effects"):
            # Your differential diagnosis generation code goes here
            se_output_text = answer_using_prefix(se_prefix, se_sample_question, se_sample_answer, se_prompt, temperature=0.0, history_context='')
            # st.write("Differential Diagnosis will appear here...")
            
            se_download_str = []
            
            # ENTITY_MEMORY_CONVERSATION_TEMPLATE
            # Display the conversation history using an expander, and allow the user to download it

            # ENTITY_MEMORY_CONVERSATION_TEMPLATE
            # Display the conversation history using an expander, and allow the user to download it
            with st.expander("Differential Diagnosis Draft", expanded=True):
                st.info(f'Scenario: {se_prompt}',icon="🧐")
                st.success(f'Educational Use Only: **NOT REVIEWED FOR CLINICAL CARE** \n\n {se_output_text["choices"][0]["message"]["content"]}', icon="🤖")
                se_download_str.append(se_prompt)
                se_download_str.append(f'Draft Patient Education Materials: {se_output_text["choices"][0]["message"]["content"]}')
                se_download_str = [disclaimer] + se_download_str
                
                # Can throw error - requires fix
                se_download_str = '\n'.join(se_download_str)
                if se_download_str:
                    st.download_button('Download',se_download_str, key = 'meds_questions')