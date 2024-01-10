#%% Import
import ast
import re  # for converting embeddings saved as strings back to arrays
import openai  # for calling the OpenAI API
import pandas as pd  # for storing text and embeddings data
import tiktoken  # for counting tokens
from scipy import spatial  # for calculating vector similarities for search
import numpy as np
import os
from dash import Dash,dcc, html, Input, Output, State  # pip install dash
import dash_bootstrap_components as dbc                # pip install dash-bootstrap-components
from fuzzywuzzy import fuzz, process

EMBEDDING_MODEL = "text-embedding-ada-002" # works best for embedding
GPT_MODEL = "gpt-3.5-turbo" # maybe can change, unsure
embedding_encoding = "cl100k_base"  # this is the encoding for text-embedding-ada-002
max_tokens = 8000  # the maximum for text-embedding-ada-002 is 8191

# Load your API key from an environment variable or secret management service
openai.api_key = "YOUR_OPENAI_API_KEY"
#openai.api_key = ""

### -----------------------------------------------------------------------------------
### -----------------------------------------------------------------------------------
### -----------------------------------------------------------------------------------

################################# TESTING API AND READING CSVs #################################
#%% Test the API connection
response = openai.Completion.create(
    engine="text-davinci-001",
    prompt="Test prompt",
    max_tokens=5
)
# Check the response
print(response)

#%% Read in CSVs located in the Embeddings folder
# Create an empty list to store DataFrames
dfs = []
# Specify the path to the Embeddings folder
embeddings_folder = 'Embeddings'  # Change this to the path of your Embeddings folder
# Iterate through CSV files in the folder
for filename in os.listdir(embeddings_folder):
    if filename.endswith('.csv'):
        csv_path = os.path.join(embeddings_folder, filename)
        df_chunk = pd.read_csv(csv_path)
        dfs.append(df_chunk)
# Concatenate all DataFrames
df = pd.concat(dfs, ignore_index=True)

#%% Converting string representations to NumPy Arrays
# Ensures that the embedding column contains actual arrays of numerical values instead of string representations  
df['embedding'] = df['embedding'].apply(eval).apply(np.array)

### -----------------------------------------------------------------------------------
### -----------------------------------------------------------------------------------
### -----------------------------------------------------------------------------------

################################# HELPER FUNCTIONS AND VARIABLES #################################
#%% 
# List of synonyms in the Manual
# Can add words/definitions as needed
synonyms = {
    "VA": "Department of Veteran Affairs",
    "Veteran Affairs": "Department of Veteran Affairs",
    #"VR&E": "Vocational Rehabilitation and Employment",
    "VR&E": "Veteran Readiness and Employment",
    "M28C": "Veteran Readiness & Employment manual",
    "U.S.C.": "United States Code",
    "RO": "Regional Office",
    "ROs": "Regional Offices",
    "CFR": "Code of Federal Regulations",
    "VBA": "Veterans Benefits Administration",
    "KM": "Knowledge Management",
    "KMP": "Knowledge Management Portal",
    "SAAs": "State Approving Agencies",
    "SAA": "State Approving Agency",
    "ALAC": "Administrative and Loan Accounting Center",
    "OJT": "On-the-Job Training",
    "LLC": "Limited Liability Corporations",
    "BAH": "Basic Allowance for Housing",
    "P911SA": "Post 9/11 Subsistence Allowance",
    "VHA": "Veterans Health Administration",
    "CWT": "Compensated Work Therapy",
    "ELR": "Education Liaison Representative",
    "ELRs": "Education Liaison Representatives",
    "NCD": "Non-College Degree",
    "IWRP": "Individualized Written Rehabilitation Plan",
    "IWRPs": "Individualized Written Rehabilitation Plans",
    "IEAP": "Individualized Employment Assistance Plan",
    "IEAPs": "Individualized Employment Assistance Plans",
    "NPWE": "Non-Paid Work Experience",
    "WEAMS": "Web Enabled Approval Management System",
    "VRC": "Vocational Rehabilitation Counselor",
    "VRCs": "Vocational Rehabilitation Counselors",
    "FM": "Full-time Modifier",
    "IDES": "Integrated Disability Evaluation System",
    "DoD": "Department of Defense",
    "ROJ": "Regional Office of Jurisdiction",
    "MAS": "Management Quality Assurance Service",
    "MOA": "Memorandum of Agreement",
    "MOAs": "Memorandum of Agreements",
    "MOU": "Memorandum of Understanding",
    "MOUs": "Memorandum of Understandings",
    "FAR": "Federal Acquisition Regulation",
    "PII": "Personally Identifiable Information",
    "ADA": "Americans with Disabilities Act",
    "SCO": "School Certifying Official(s)",
    "SCOs": "School Certifying Officials",
    "SE": "Supported Employment",
    "TOE": "Transfer of Entitlement",
    "VAMCs": "Veterans Administration Medical Centers",
    "VAMC": "Veterans Administration Medical Center",
    "STAR": "Systematic Technical Assurance Review",
    "VIST": "Visual Impairment Services Team",
    "NSO": "National Service Organizations",
    "OGC": "Office of General Counsel",
    "EC": "Employment Coordinator",
    "ECs": "Employment Coordinators",
    "QA": "Quality Assurance",
    "EEO": "Equal Employment Opportunity",
    "ORMDI": "Office of Resolution Management, Diversity, & Inclusion",
    "FPEI": "For-Profit Educational Institutions",
    "SOAR": "Strategic Oversight and Analysis Review",
    "JRL": "Job Resource Lab",
    "CER": "Counseling Evaluation and Rehabilitation",
    "VBMS": "Veterans Benefits Management System",
    "DCS": "Data Control Sheet",
    "RMN": "Records Management Number",
    "CAPRI": "Compensation and Pension Records Interchange",
    "BIRLS": "Beneficiary Identification and Records Locator Subsystem",
    "RCS": "Records Control Schedule",
    "RMC": "Records Management Center",
    "NARA": "National Archives and Records Administration",
    "FBE": "File Bank Extraction",
    "RMO": "Records Management Officer",
    "DEA": "Dependents Education and Assistance",
    "SF": "Standard Form",
    "SFs": "Standard Forms",
    "VAF": "VA Form",
    "VAFs": "VA Forms",
    "OMB": "Office of Management and Budget",
    "FOIA": "Freedom of Information Act",
    "SOR": "System of Records",
    "SORs": "Systems of Records",
    "SORN": "System of Records Notices",
    #"GED": "General Education Development",
    #"GED": "General Eligibility Determination",
    "CLEP": "College Level Examination Program",
    "CEEB": "College Entrance Examination Board",
    #"ACT": "American College Testing",
    "PA": "Privacy Act",
    #"POA": "Power of Attorney",
    "POA": "Program Outcome Accuracy",
    "VSO": "Veterans Service Organization",
    "VSOs": "Veteran Service Organizations",
    "CD": "compact disc",
    "DVOP": "Disabled Veterans Outreach Program",
    "LVER": "Local Veterans Employment Representative",
    "LVERs": "Local Veterans' Employment Representatives",
    "CSEM": "Common Security Employee Manager",
    "LOB": "Line of Business",
    #"SO": "State Office",
    "CSS": "Common Security Systems",
    "FARs": "Federal Acquisition Regulations",
    "SAM": "Subsistence Allowance Module",
    "EAA": "Employment Adjustment Allowance",
    "CAATS": "Centralized Administrative Accounting Transaction System",
    "ALAC": "Administrative and Loan Accounting Center",
    "SSD": "Support Services Division",
    "ACO": "Administrative Contracting Officer",
    "FMS": "Financial Management System",
    "C&P": "Compensation and Pension",
    "PIF": "Pending Issue File",
    "PHF": "Payment History File",
    "SVRC": "Supervisory Vocational Rehabilitation Counselor",
    "SVRCs": "Supervisory Vocational Rehabilitation Counselors",
    "ISO": "Information Security Officer",
    "BVA": "Board of Veterans Appeals",
    "VACOLS": "Veterans Appeals Control and Locator System",
    "IPPS": "Invoice Payment Processing System",
    "OBIEE": "Oracle Business Intelligence Enterprise Edition",
    "PA&I": "Performance Analysis and Integrity",
    "VVC": "VA Video Connect",
    "FCMT": "Federal Case Management Tool",
    "IDES": "Integrated Disability Evaluation Service",
    "VTA": "Veterans Tracking Application",
    "COTS": "Commercial Off the Shelf",
    "e-VA": "Electronic Virtual Assistant",
    "SARA": "Semi-Autonomous Research Assistant",
    "AI": "Artificial Intelligence",
    "LTS": "Long Term Solution",
    "OFO": "Office of Field Operations",
    "RAM": "Resource Allocation Model",
    "FTE": "Full Time Equivalent",
    "GOE": "General Operating Expense",
    "HR": "Human Resources",
    "VSOC": "VetSuccess on Campus",
    "DEU": "Delegated Examining Unit",
    "PSS": "Program Support Specialist",
    "RFL": "Revolving Fund Loan",
    "VACO": "VA Central Office",
    "OTM": "Office of Talent Management",
    "TMS": "Talent Management System",
    "IDP": "Individual Development Plans",
    "CRC": "Certified Rehabilitation Counselor",
    "DOT": "Dictionary of Occupational Title(s)",
    "DOTs": "Dictionary of Occupational Titles",
    "O*NET": "Occupational Information Network",
    "O*Net": "Occupational Information Network",
    "OOH": "Occupational Outlook Handbook",
    "IL": "Independent Living",
    "RSA": "Rehabilitation Services Administration",
    "DOL": "Department of Labor",
    "VETS": "Veterans Employment and Training Service",
    "MEPSS": "Medical Electronic Performance Support System",
    "MEPSS": "C&P Medical EPSS",
    "TPSS": "Training Performance Support System",
    "IWT": "Instructor-led Web-based Training",
    "CBTS": "Competency Based Training System",
    "NTC": "National Training Curriculum",
    "COR": "Contract Officer's Representative",
    "TM": "Training Manager",
    "CRCC": "Commission on Rehabilitation Counselor Certification",
    "CEU": "Continuing Education Unit",
    "CEUs": "Continuing Education Units",
    "VRP": "Vocational Rehabilitation Panel",
    "SEH": "Serious Employment Handicap",
    "SRT": "Special Restorative Training",
    "SVT": "Special Vocational Training",
    "IILP": "Individualized Independent Living Plan",
    "IILPs": "Individualized Independent Living Plans",
    #"FAC": "Field Advisory Committee",
    #"FAC": "Federal Acquisition Certification",
    "VACOR": "Veteran's Advisor Committee on Rehabilitation",
    "SECVA": "Secretary of Veterans Affairs",
    "DFO": "Designated Federal Officer",
    "ED": "Department of Education",
    "NIDRR": "National Institute on Disability and Rehabilitation Research",
    "SEP": "Self-Employment Panel",
    "SBA": "Small Business Administration",
    "SBDC": "Small Business Development Center",
    "SBDCs": "Small Business Development Centers",
    "APP": "Applicant",
    "EP": "Evaluation and Planning",
    "EE": "Extended Evaluation",
    "RTE": "Rehabilitation To the point of Employability",
    "JR": "Job Ready",
    "REH": "Rehabilitated",
    "INT": "Interrupted",
    "DIS": "Discontinued",
    "RC": "Reason Code",
    "RCs": "Reason Codes",
    "DRC": "Designated Reason Code",
    "DRCs": "Designated Reason Codes",
    "IEEP": "Individualized Extended Evaluation Plan",
    "IEEPs": "Individualized Extended Evaluation Plans",
    "EH": "Employment Handicap",
    "OI&T": "Office of Information and Technology",
    "SI": "Seriously Injured",
    "VSI": "Very Seriously Injured",
    "SM/V": "Service Member or Veteran",
    "NDAA": "National Defense Authorization Act",
    "POC": "point of contact",
    "VSC": "Veterans Service Center",
    "SCD": "Service Connected Disability",
    "SCDs": "Service Connected Disabilities",
    "PEB": "Physical Evaluation Board",
    "MSSR": "Military Service Status Referral",
    "TAP": "Transition Assistance Program",
    "DTAP": "Disable Transition Assistance Program",
    "PDHRA": "Post-Deployment Health Reassessment",
    "YRRP": "Yellow Ribbon Reintegration Program",
    "LMI": "Labor Market Information",
    "SEI": "Special Employer Incentive",
    "USERRA": "Uniform Services Employment and Reemployment Rights Act",
    "MEB": "Medical Evaluation Board",
    "SM": "Service Member",
    "E2I": "Education and Employment Initiative",
    "BDD": "Benefits Delivery at Discharge",
    "PEBLO": "Physical Evaluation Board Liaison Officer",
    "MSC": "Military Service Coordinators",
    "MSC/SM": "Military Service Coordinator/Servicemember",
    "Pub. L.": "Public Law",
    "TA": "Tuition Assistance",
    "CUE": "Clear and Unmistakable Error",
    "ETD": "Eligibility Termination Date",
    "EOD": "Enterance on Duty",
    "RAD": "Release from Active Duty",
    "LDES": "Legacy Disability Evaluation System",
    "AutoGED": "Automated GED",
    "RL": "Revocable License",
    "VARO": "VA Regional Office",
    "VPN": "Virtual Private Network",
    "CWINRS": "Corporate WINRS",
    "VETSNET": "Veterans Services Network",
    "SSA": "Social Security Administration",
    "COVERS": "Control of Veterans Record System",
    "CRM-UD": "Customer Relationship Management - Unified Desktop",
    "HIPAA": "Health Insurance Portability and Accountability Act",
    "OTED": "Outreach, Transition, and Economic Development",
    "EDU": "Education Service",
    "PCPG": "Personalized Career Planning and Guidance",
    "RPO": "Regional Processing Office",
    "RPOs": "Regional Processing Offices",
    "SME": "Subject Matter Expert",
    "CBOC": "Community-Based Outpatient Clinic",
    "VITAL": "Veterans Integration to Academic Leadership",
    "COB": "Close of Business",
    "MAP-D": "Modern Award Processing",
    "VR": "Vocational Rehabilitation",
    "HLR": "Higher-Level Review(s)",
    "SC": "Supplemental Claim",
    "SCs": "Supplemental Claims",
    "HLRs": "Higher-Level Reviews",
    "VCAA": "Veterans Claims Assistance Act",
    "NOD": "Notice of Disagreement",
    "AMO": "Appeals Management Office",
    "CAVC": "Court of Appeals for Veterans Claims",
    "COVA": "Court of Veterans Appeals",
    "FPS": "Federal Protective Service",
    "OIG": "Office of the Inspector General",
    "STR": "Service Treatment Records",
    "GRS": "General Records Schedule",
    "MRG": "Maximum Rehabilitation Gain",
    "HON": "Honorable",
    "UHC": "Under Honorable Conditions",
    "IRND": "Initial Rating Notification Date",
    "CEST": "Claim Established",
    "CADJ": "Claim Adjusted",
    "TOE": "Transfer of Entitlement",
    "IRM": "Information Resource Management",
    "NSCD": "Non-Service Connected Disability",
    "NSCD": "Non-Service Connected Disabilities",
    "RWT": "Reduced Work Tolerance",
    "CSA": "Controlled Substance Act",
    "VIS": "Veterans Information Systems",
    "COE": "Certificate of Eligibility",
    "MGIB": "Montgomery GI Bill",
    "VEAP": "Veterans Educational Assistance Program",
    "KSA": "Knowledge, Skills, and Abilities",
    "OPM": "Office of Personnel Management",
    "BLS": "Bureau of Labor Statistics",
    "IHL": "Institution of Higher Learning",
    "IHLs": "Institutions of Higher Learning",
    "T&F": "Tuition and Fees",
    "CEP": "Customer Engagement Portal",
    "ADAAA": "ADA Amendments Act",
    "POE": "Principles of Excellence",
    "VET TEC": "Veterans Employment Through Technology Education Course",
    "SSN": "Social Security Number",
    "OIF": "Operation Iraqi Freedom",
    "OEF": "Operation Enduring Freedom",
    "OND": "Operation New Dawn",
    "OEF/OIF/OND": "Operation Enduring Freedom/Operation Iraqi Freedom/Operation New Dawn",
    "TWE": "Transitional Work Experience",
    "VRSs": "Vocational Rehabilitation Specialists",
    "VRS": "Vocational Rehabilitation Specialist",
    "CARF": "Commission on Accreditation of Rehabilitation Facilities",
    "USPRA": "U.S. Psychiatric Rehabilitation Association",
    "SCORE": "Service Corps of Retired Executives",
    "VBOC": "Veterans Business Outreach Center",
    "VBOCs": "Veterans Business Outreach Centers",
    "OSDBU": "Office of Small and Disadvantaged Business Utilization",
    "VEP": "Veteran Entrepreneur Portal",
    "CVE": "Center for Verification and Evaluation",
    "SDVOSB": "Service Disabled Veteran Owned Small Business",
    "SDVOSBs": "Service Disabled Veteran Owned Small Businesses",
    "B2B": "Boots to Business",
    "CO": "Contracting Officer",
    "COs": "Contracting Officers",
    "CMA": "Case Management Appointment",
    "CMAs": "Case Management Appointments",
    "FL": "Form Letter",
    #"AT": "Assistive Technology",
    "ADL": "Activities of Daily Living",
    "CILA": "Comprehensive Independent Living Assessment",
    "CILAs": "Comprehensive Independent Living Assessments",
    "SAH": "Specially Adapted Housing",
    "RLC": "Regional Loan Center",
    "GPC": "Government Purchase Card",
    "LGY": "Loan Guaranty",
    "HISA": "Home Improvements and Structural Alterations",
    "SAHSHA": "Specially Adapted Housing Special Home Adaption",
    "MPR": "Minimum Property Requirements",
    "TRA": "Temporary Residence Adaptation",
    "VAAR": "Veterans Affairs Acquisition Regulations",
    #"CMS": "Construction Management Service",
    #"CMS": "Case Management Solution"
    "JAN": "Job Accommodation Network",
    "FAA": "Federal Aviation Administration",
    "FLSA": "Fair Labor Standards Act",
    "NTTHD": "National Technology Help Desk",
    "ATV": "All-Terrain Vehicle",
    "ATVs": "All-Terrain Vehicles",
    "JLV": "Joint Legacy Viewer",
    "GUI": "Graphic User Interface",
    "PTSD": "Post Traumatice Stress Disorder",
    "ADI": "Assistance Dogs International",
    "IGDF": "International Guide Dog Federation",
    "TDIU": "Total Disability based on Individual Unemployability",
    "TIN": "Tax Identification Number",
    "FSC": "Financial Services Center",
    "EFT": "Electronic Funds Transfer",
    "BOC": "Budget Object Code",
    "BOCs": "Budget Object Codes",
    "Ed/Voc": "Educational Vocational",
    "RB": "Readjustment Benefits",
    "CY": "Calendar Year",
    "TOP": "Treasury Offset Program",
    "HCA": "Head of the Contracting Activity",
    "IDIQ": "Indefinite-Delivery/Indefinite-Quantity",
    "SAC-F": "Strategic Acquisition Center - Frederick",
    "CLIN": "Contract Line Item Number",
    "CLINs": "Contract Line Item Numbers",
    "COR": "Contracting Officer's Representative(s)",
    "ACOR": "Alternate Contracting Officer's Representative(s)",
    "FAITAS": "Federal Acquisition Institute Training Application System",
    "QS": "Quality Surveillance",
    "iFAMS": "Integrated Financial and Acquisition Management System",
    "CPARS": "Contractor Performance Assessment Reporting System",
    "PWS": "Performance Work Statement",
    "OT": "Occupational Therapist",
    "VAC": "Veterans Affairs Canada",
    "MQAS": "Management Quality Assurance Service",
    "AO": "Approving Official",
    "SPL": "Single Purchase Limit",
    "SCLS": "Service Contract Labor Standards",
    "FSSI": "Federal Strategic Sourcing Initiative",
    "A/OPC": "Agency/Organization Program Coordinators",
    "MAC": "Master Accounting Code",
    "CCP": "Charge Card Portal",
    "MPL": "Micro Purchase Limit",
    "OIC": "Office of Internal Controls",
    "BT": "Beneficiary Travel",
    "DAV": "Disabled American Veterans",
    "VTS": "VA Transportation Systems",
    "GSA": "General Service Administration",
    "CBA": "Centrally Billed Account",
    #"FAS": "Financial Accounting System",
    "FAS": "Finance and Accounting System(s)",
    "STEM": "Science Technology Engineering Math",
    "VCE": "Veterans Claims Examiner",
    "BDN": "Benefits Delivery Network",
    "FY": "Fiscal Year",
    "CWE": "Community-Based Work Experience",
    "COLA": "Cost of Living Allowance",
    "CH31SA": "Chapter 31 Subsistence Allowance",
    "Ch31SA": "Chapter 31 Subsistence Allowance",
    "SFC": "Sub Facility Code(s)",
    "OHA": "Overseas Housing Allowance",
    "PR": "Protected Rate",
    "ECH": "Equivalent Credit Hours",
    "GPA": "Grade Point Average",
    "WF": "Withdrawal Fail",
    "DMC": "Debt Management Center",
    "COWC": "Committee on Waivers and Compromises",
    "CAUT": "Claims Authorization",
    "TDD": "Telecommunications Device for the Deaf",
    "HSPD": "Homeland Security Presidential Directive",
    "SAC": "Special Agreement Check",
    "NACI": "National Agency Check with Written Inquiries",
    "U.S.": "United States",
    "FMP": "Foreign Medical Program",
    "NCIC": "National Crime Information Center",
    "FBP": "Federal Bonding Program",
    "NOK": "Next of Kin",
    "NOD": "Notice of Death",
    "DOL/VETS": "Department of Labor, Veterans Employment and Training Service",
    "VOW": "Veterans Opportunity to Work",
    "VRA": "Veterans Recruitment Appointment",
    "VRAs": "Veterans Recruitment Appointments",
    "VEPM": "Veterans Employment Program Manager",
    "VEPMs": "Veterans Employment Program Managers",
    "SPPC": "Special Placement Program Coordinator",
    "SPPCs": "Special Placement Program Coordinators",
    "DVAAP": "Disabled Veterans Affirmative Action Program",
    "VEOA": "Veterans Employment Opportunities Act",
    "SOC": "Standard Occupational Classification",
    "VR&E Officer": "Veteran Readiness and Employment Officer",
    "FECA": "Federal Employees Compensation Act",
    "A&I": "Office of Performance Analysis and Integrity",
    "OPA&I": "Office of Performance Analysis and Integrity",
    "DCS ID": "Document Control Sheet Identification",
    "WIT": "Workforce Information Tool",
    "EDRPA": "Entitlement Determination and Rehabilitation Planning Accuracy",
    "RSDA": "Rehabilitation Services Delivery Accuracy",
    "FA": "Fiscal Accuracy",
    "CCA": "Case Closure Accuracy",
    "CH-36A": "Chapter 36 Accuracy",
    "SCR": "Supplemental Claim Review",
    "SCRs": "Supplemental Claim Reviews",
    "IPERA": "Improper Payments Elimination and Recovery Act",
    "SOP": "Standard Operating Procedure(s)",
    "CAM": "Corrective Actions Management",
    "EDA": "Entitlement Determination Accuracy",
    "CAP": "Combined Assessment Program",
    "FMAS": "Financial Management Assurance Service",
    "eCMS": "Electronic Contract Management Sysytem",
    "USB": "Under Secretary for Benefits",
    "FYTD": "Fiscal Year To Date",
    "ISC": "Intensive Services Coordinator",
    "PVA": "Paralyze Veterans of America",
    "VFW": "Veterans of Foreign Wars",
    "WMP": "Workload Management Plan",
    "IG": "Inspector General",
    "GAO": "General Accounting Office",
    "HR": "Human Resources",
    "DMZ": "Demilitarized Zone",
    "ORT": "Outreach Report Tool",
    #"Azure RMS": "Azure Rights Management",
    "FAC-COR": "Federal Acquisition Certification for Contracting Officer's Representatives",
    "PIV": "Personal Identity Verification",
    "QARB": "Quality Assurance Review Board",
    "Pre-Chapter 31": "CWINRS Pre-Chapter 31",
    "pre-Chapter 31": "CWINRS pre-Chapter 31",
    "Entitlement Determination Notification Date": "The Entitlement Determination Notification Date must match the date the claimant was notified in writing of the entitlement decision.  The Entitlement Determination Notification Date must also match the date the entitlement determination is documented on the VA Form 28-1902b",
    "The Entitlement Determination Notification Date": "The Entitlement Determination Notification Date must match the date the claimant was notified in writing of the entitlement decision.  The Entitlement Determination Notification Date must also match the date the entitlement determination is documented on the VA Form 28-1902b",
    "entitlement determination notification date": "The Entitlement Determination Notification Date must match the date the claimant was notified in writing of the entitlement decision.  The Entitlement Determination Notification Date must also match the date the entitlement determination is documented on the VA Form 28-1902b",
    "the Entitlement Determination Notification Date": "The Entitlement Determination Notification Date must match the date the claimant was notified in writing of the entitlement decision.  The Entitlement Determination Notification Date must also match the date the entitlement determination is documented on the VA Form 28-1902b",
    "QA Deferral Period": "The amount of time given to agencies to implement a new or changed policy or procedure",
    "QA deferral period": "The amount of time given to agencies to implement a new or changed policy or procedure",
}

# Function to get the response from ChatGPT
def get_response(query):
    # Call the ask function to get the response
    response = ask(query, df)
    return response

# Function to see how related a string is to the input
def strings_ranked_by_relatedness(
    query: str,
    df: pd.DataFrame,
    relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y),
    top_n: int = 100
) -> tuple[list[str], list[float]]:
    """Returns a list of strings and relatednesses, sorted from most related to least."""
    query_embedding_response = openai.Embedding.create(
        model=EMBEDDING_MODEL,
        input=query,
    )
    query_embedding = query_embedding_response["data"][0]["embedding"]
    strings_and_relatednesses = [
        (row["Content"], relatedness_fn(query_embedding, row["embedding"]))
        for i, row in df.iterrows()
    ]
    strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
    strings, relatednesses = zip(*strings_and_relatednesses)
    return strings[:top_n], relatednesses[:top_n]

# Return the number of tokens in a string
def num_tokens(text: str, model: str = GPT_MODEL) -> int:
    """Return the number of tokens in a string."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

# Return a message for GPT, with relevant source texts pulled from a dataframe
def query_message(
    query: str,
    df: pd.DataFrame,
    model: str,
    token_budget: int
) -> str:
    """Return a message for GPT, with relevant source texts pulled from a dataframe."""
    for term, replacement in synonyms.items():
        query = query.replace(term, replacement)
    strings, relatednesses = strings_ranked_by_relatedness(query, df)
    introduction = 'Use the scraped webpages pertaining to the VA Manual to answer the subsequent question. If the answer cannot be found in the data, write "I could not find an answer. Please rephrase the question and/or check spelling."'
    question = f"\n\nQuestion: {query}"
    message = introduction
    for string in strings:
        next_article = f'\n\nWebpage section:\n"""\n{string}\n"""'
        if num_tokens(message + next_article + question, model=model) > token_budget:
            break
        else:
            message += next_article
    if len(strings) == 0:
        message += '\n\nI could not find an answer. Please rephrase the question and/or check spelling.'
    message = re.sub(r'[\\\n]', '', message)
    return message + question

# Answers a query using GPT and a dataframe of relevant texts embeddings.
def ask(
    query: str,
    df: pd.DataFrame = df,
    model: str = GPT_MODEL, 
    token_budget: int = 4096 - 500,
    print_message: bool = False,
) -> str:
    """Answers a query using GPT and a dataframe of relevant texts and embeddings."""
    message = query_message(query, df, model=model, token_budget=token_budget)
    if print_message:
        print(message)
    messages = [
        {"role": "system", "content": "You answer questions about the VR&E Manual."},
        {"role": "user", "content": message},
    ]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0
    )
    response_message = response["choices"][0]["message"]["content"]
    return response_message

### -----------------------------------------------------------------------------------
### -----------------------------------------------------------------------------------
### -----------------------------------------------------------------------------------

###################################### INTERFACE RUN ######################################
#%% Instantiate the Dash app
app = Dash(__name__, external_stylesheets=[dbc.themes.LITERA])

# Define the layout of the Dash app
app.layout = html.Div(
    [
        html.Div(
            html.Img(id='image1', src="/assets/USDeptVeteransAffairs.png", style={'text-align': 'left', 'margin': '0', 'padding': '0', 'width': '40%', 'height': 'auto'}),
            #style={'text-align': 'left', 'margin': '0', 'padding': '0'}  # Reset margins and paddings
            className='mb-4',
            style={'display': 'flex', 'align-items': 'left', 'flex-direction': 'column', 'background-color': '#051B44', 'color': 'white', 'margin': '0', 'padding': '0'}
        ),
        html.Div(
            [
                html.H1("Chatbot Assistant Demo for U.S. Department of Veterans Affairs Manual", className='text-center', style={'margin': '0 auto', 'padding': '0', 'max-width': '70%'}),
                #html.Img(id='image', src="/assets/VA_Seal2.png", style={'max-width': '100%', 'height': 'auto'})  # Update the image file name and dimensions
            ],
            className='mb-4',
            style={'display': 'flex', 'align-items': 'center', 'flex-direction': 'column', 'margin': '0 auto'}
        ),
        html.Div(
            dcc.Input(
                id='input-text',
                type='text',
                placeholder='Type your question here',
                style={'width': '100%', 'max-width': '500px', 'margin': '0 auto'}
            ),
            className='mb-4',
            style={'display': 'flex', 'justify-content': 'center'}
        ),
        html.Div(
            html.Button('Ask', id='submit-button', n_clicks=0, className='d-block mx-auto'),
            className='mb-4',
            style={'display': 'flex', 'justify-content': 'center'}
        ),
        dcc.Loading(
            children=[
                html.Div(id='output-text', style={'max-width': '70%', 'margin': '0 auto'})  # Set the maximum width for the output text here)
            ],
            type="circle",
        ),
        html.Div(
            html.Img(id='image2', src="/assets/VA_Seal2.png", style={'max-width': '100%', 'height': 'auto'}),
            className='mb-4',
            style={'display': 'flex', 'align-items': 'center', 'flex-direction': 'column', 'margin': '0 auto'}
        ),
    ],
    style={'width': '100%', 'height': '100vh'}  # Set the container to take full height of the viewport
)

# Define the callback function
@app.callback(
    Output('output-text', 'children'),
    Input('submit-button', 'n_clicks'),
    State('input-text', 'value')
)
def update_output(n_clicks, text_input):
    if n_clicks > 0:
        # Get the response from ChatGPT (implement this part)
        response = get_response(text_input)
        # Return the generated text as the output
        return response

if __name__ == '__main__':
    app.run_server(debug=False)       

# %%
