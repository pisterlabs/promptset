from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain_community.vectorstores import Chroma


examples = [
    {
        "context": """
You are in the role of an abstractor who will analyze eligibility criteria for a clinical trial and represent the information as a list of individual criteria in a tabular format that will contain the following columns: 
Type: listing whether criterion is an Exclusion or Inclusion criterion
Original Text: the original text of the criterion
Disease/Condition: If the criterion contains a disease or condition name it by its canonical name
Procedure: If the criterion contains a therapeutic procedure name it by its canonical name
Drug:  If the criterion contains a therapeutic drug name it by its canonical name
Biomarker:  If the criterion contains a biomarker name it by its canonical name
Computable Rule: Translate the criteria into a logical expression that could be interpreted programmatically
Here is the criteria to analyze:
    Inclusion Criteria
    •	Age 18 or older
    •	Willing and able to provide informed consent
    •	Metastatic breast cancer, biopsy proven
    o	Estrogen receptor (ER)+/HER2-, defined as > 5% ER+ staining
    o	HER2+ (regardless of ER status), including HER2-low and high expressors
    •	History of at least 6 months, sustained response to systemic therapy (clinically or radiographically defined as complete or stable response without progression)
    •	Isolated site of disease progression on fludeoxyglucose F-18 (FDG) positron emission tomography (PET) scan
    •	Consented to 12-245
    •	Eastern Cooperative Oncology Group (ECOG) performance status 0-1

    Exclusion Criteria
    •	Pregnancy
    •	Serious medical comorbidity precluding radiation, including connective tissue disorders
    •	Intracranial disease (including previous intracranial involvement)
    •	Previous radiotherapy to the intended treatment site that precludes developing a treatment plan that respects normal tissue tolerances
""",
        "answer": """
| Type | Original Text | Disease/Condition | Procedure | Drug | Biomarker | Computable Rule |
| --- | --- | --- | --- | --- | --- | --- |
| Inclusion | Metastatic breast cancer, biopsy proven | Metastatic breast cancer | | | | diagnosis == "Metastatic breast cancer" |
| Inclusion | Estrogen receptor (ER)+/HER2-, defined as > 5% ER+ staining | | | | HER2- | HER2- > 5% ER+ |
| Inclusion | HER2+ (regardless of ER status), including HER2-low and high expressors | | | | HER2+ | HER2+ is True |
| Exclusion | Previous radiotherapy to the intended treatment site that precludes developing a treatment plan that respects normal tissue tolerances | | Prior radiation therapy | | | Prior radiation therapy is True |
""",
    },
    {
        "context": """
You are in the role of an abstractor who will analyze eligibility criteria for a clinical trial and represent the information as a list of individual criteria in a tabular format that will contain the following columns: 
Type: listing whether criterion is an Exclusion or Inclusion criterion
Original Text: the original text of the criterion
Disease/Condition: If the criterion contains a disease or condition name it by its canonical name
Procedure: If the criterion contains a therapeutic procedure name it by its canonical name
Drug:  If the criterion contains a therapeutic drug name it by its canonical name
Biomarker:  If the criterion contains a biomarker name it by its canonical name
Computable Rule: Translate the criteria into a logical expression that could be interpreted programmatically
Here is the criteria to analyze:
    Inclusion Criteria
    •	Signed informed consent must be obtained prior to performing any specific pre-screening and screening procedure
    •	Male or female >= 18 years of age at the time of informed consent
    •	Histologically or cytologically confirmed diagnosis of advanced/metastatic differentiated thyroid cancer
    •	Radio active iodine refractory disease
    •	BRAFV600E mutation positive tumor sample as per Novartis designated central laboratory result
    •	Has progressed on at least 1 but not more than 2 prior VEGFR targeted therapy
    •	Eastern Cooperative Oncology Group performance status >= 2
    •	At least one measurable lesion as defined by RECIST 1.1
    •	Anaplastic or medullary carcinoma of the Tyroid

    Exclusion Criteria
    •	Previous treatment with BRAF inhibitor and/or MEK inhibitor
    •	Concomitant RET Fusion Positive Thyroid cancer
    •	Receipt of any type of small molecule kinase inhibitor within 2 weeks before randomization
    •	Receipt of any type of cancer antibody or systemic chemotherapy within 4 weeks before randomization
    •	Receipt of radiation therapy for bone metastasis within 2 weeks or any other radiation therapy within 4 weeks before randomization
    •	A history or current evidence/risk of retinal vein occlusion or central serous retinopathy
""",
        "answer": """
| Type | Original Text | Disease/Condition | Procedure | Drug | Biomarker | Computable Rule |
| --- | --- | --- | --- | --- | --- | --- |
| Inclusion | Histologically or cytologically confirmed diagnosis of advanced/metastatic differentiated thyroid cancer | Thryoid cancer | | | | diagnosis == "Thyroid cancer" |
| Inclusion | BRAFV600E mutation positive tumor sample as per Novartis designated central laboratory result | | | | BRAFV600E | BRAFV600E is True |
| Inclusion | Has progressed on at least 1 but not more than 2 prior VEGFR targeted therapy | | VEGFR targeted therapy | | | VEGFR >= 1, VEGFR < 2 |
| Exclusion | Previous treatment with BRAF inhibitor and/or MEK inhibitor | | | BRAF inhibitor, MEK inhibitor | | BRAF inhibitor is True OR MEK inhibitor is True |
""",
    },
    {
        "context": """
You are in the role of an abstractor who will analyze eligibility criteria for a clinical trial and represent the information as a list of individual criteria in a tabular format that will contain the following columns: 
Type: listing whether criterion is an Exclusion or Inclusion criterion
Original Text: the original text of the criterion
Disease/Condition: If the criterion contains a disease or condition name it by its canonical name
Procedure: If the criterion contains a therapeutic procedure name it by its canonical name
Drug:  If the criterion contains a therapeutic drug name it by its canonical name
Biomarker:  If the criterion contains a biomarker name it by its canonical name
Computable Rule: Translate the criteria into a logical expression that could be interpreted programmatically
    Inclusion Criteria
    •	Participants must be ‚â• 18 years of age
    •	Histologically or cytologically confirmed diagnosis of metastatic solid tumors
    •	Eastern Cooperative Oncology Group (ECOG) performance status 0-1
    •	All participants should have at least 1 measurable disease per RECIST v1.1. An irradiated lesion can be considered measurable only if progression has been demonstrated on the irradiated lesion.
    •	Body weight within [45 - 150 kg] (inclusive)
    •	All Contraceptive use by men and women should be consistent with local regulations regarding the methods of contraception for those participating in clinical studies.
    •	Capable of giving signed informed consent
    •	Any clinically significant cardiac disease
    •	History of or current interstitial lung disease or pneumonitis

    Exclusion Criteria
    •	Uncontrolled or unresolved acute renal failure
    •	Prior solid organ or hematologic transplant.
    •	Known positivity with human immunodeficiency virus (HIV), known active hepatitis A, B, and C, or uncontrolled chronic or ongoing infectious requiring parenteral treatment.
    •	Receipt of a live-virus vaccination within 28 days of planned treatment start
    •	Participation in a concurrent clinical study in the treatment period.
    •	Inadequate hematologic, hepatic and renal function
    •	Participant not suitable for participation, whatever the reason, as judged by the Investigator, including medical or clinical conditions.
""",
        "answer": """
| Type | Original Text | Disease/Condition | Procedure | Drug | Biomarker | Computable Rule |
| --- | --- | --- | --- | --- | --- | --- |
| Inclusion | Histologically or cytologically confirmed diagnosis of metastatic solid tumors | Metastatic solid tumor | | | | diagnosis == "Metastatic solid tumor" |
| Exclusion | Prior solid organ or hematologic transplant. | | Solid organ transplantation | | | Solid organ transplantation is True |
"""
    },
    {
        "context": """
You are in the role of an abstractor who will analyze eligibility criteria for a clinical trial and represent the information as a list of individual criteria in a tabular format that will contain the following columns: 
Type: listing whether criterion is an Exclusion or Inclusion criterion
Original Text: the original text of the criterion
Disease/Condition: If the criterion contains a disease or condition name it by its canonical name
Procedure: If the criterion contains a therapeutic procedure name it by its canonical name
Drug:  If the criterion contains a therapeutic drug name it by its canonical name
Biomarker:  If the criterion contains a biomarker name it by its canonical name
Computable Rule: Translate the criteria into a logical expression that could be interpreted programmatically
    Inclusion Criteria:
    •	Adults with a confirmed diagnosis of unresectable, locally advanced and/or metastatic Stage IIIB/IV NSCLC, Stage III/IV PDAC and/or Stage III/IV CRC with no curative-intent treatment options and documented activating KRAS mutation (without known additional actionable driver mutations such as EGFR, ALK or ROS1)
    •	Documented progression and measurable disease after ‚â• 1 prior line of systemic therapy (‚â• 2 and ‚â§ 4 prior lines for NSCLC) with adequate washout period and resolution of treatment-related toxicities to ‚â§ Grade 2
    •	ECOG PS of 0-2 (0-1 for PDAC) and a life expectancy > 3 months in the opinion of the Investigator
    •	Adequate hematological, liver, and renal function
    •	Men and women of childbearing potential must use adequate birth control measures for the duration of the trial and at least 90 days after discontinuing study treatment
    •	Symptomatic and/or untreated CNS or brain metastasis, pre-existing ILD or pericardial/pleural effusion of ‚â• grade 2 or requiring chronic oxygen therapy for COPD or pleural effusions
    •	Serious concomitant disorder including infection
    •	Known positive test for HIV, HCV, HBV surface antigen

    Exclusion Criteria:
    •	Concurrent malignancy in the previous 2 years
    •	Prior menin inhibitor therapy
    •	Requiring treatment with a strong or moderate CYP3A inhibitor/inducer
    •	Significant cardiovascular disease or QTcF or QTcB prolongation.
    •	Major surgery within 4 weeks prior to first dose
    •	Women who are pregnant or lactating.
""",
        "answer": """
| Type | Original Text | Disease/Condition | Procedure | Drug | Biomarker | Computable Rule |
| --- | --- | --- | --- | --- | --- | --- |
| Inclusion | Adults with a confirmed diagnosis of unresectable, locally advanced and/or metastatic Stage IIIB/IV NSCLC, Stage III/IV PDAC and/or Stage III/IV CRC with no curative-intent treatment options and documented activating KRAS mutation (without known additional actionable driver mutations such as EGFR, ALK or ROS1) | Metastatic Lung Non-Small Cell Carcinoma, Metastatic Pancreatic Ductal Adenocarcinoma, Metastatic Colorectal Carcinoma | | | KRAS | diagnosis == "Metastatic Lung Non-Small Cell Carcinoma" OR diagnosis == "Metastatic Pancreatic Ductal Adenocarcinoma" OR diagnosis == "Metastatic Colorectal Carcinoma", KRAS is True |
| Exclusion | Prior menin inhibitor therapy | | | Menin inhibitor | | Menin inhibitor is True |
"""
    }
]

example_selector = SemanticSimilarityExampleSelector.from_examples(
    # This is the list of examples available to select from.
    examples,
    # This is the embedding class used to produce embeddings which are used to measure semantic similarity.
    HuggingFaceEmbeddings(),
    # This is the VectorStore class that is used to store the embeddings and do a similarity search over.
    Chroma,
    # This is the number of examples to produce.
    k=1,
)

example_prompt = PromptTemplate(
    input_variables=["context", "answer"], template="{context}{answer}"
)

prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    suffix="""You are in the role of an abstractor who will analyze eligibility criteria for a clinical trial and represent the information as a list of individual criteria in a tabular format that will contain the following columns: 
Type: listing whether criterion is an Exclusion or Inclusion criterion
Original Text: the original text of the criterion
Disease/Condition: If the criterion contains a disease or condition name it by its canonical name
Procedure: If the criterion contains a therapeutic procedure name it by its canonical name
Drug:  If the criterion contains a therapeutic drug name it by its canonical name
Biomarker:  If the criterion contains a biomarker name it by its canonical name
Computable Rule: Translate the criteria into a logical expression that could be interpreted programmatically
    {criteria}
""",
    input_variables=["criteria"],
)


prompt_zero = PromptTemplate.from_template("""You are in the role of an abstractor who will analyze eligibility criteria for a clinical trial and represent the information as a list of individual criteria in a tabular format that will contain the following columns: 
Type: listing whether criterion is an Exclusion or Inclusion criterion
Original Text: the original text of the criterion
Disease/Condition: If the criterion contains a disease or condition name it by its canonical name
Procedure: If the criterion contains a therapeutic procedure name it by its canonical name
Drug:  If the criterion contains a therapeutic drug name it by its canonical name
Biomarker:  If the criterion contains a biomarker name it by its canonical name
Computable Rule: Translate the criteria into a logical expression that could be interpreted programmatically
    {criteria}
""")