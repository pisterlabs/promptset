import os
import re
import tenacity
import configparser
import markdown

from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain.prompts import PromptTemplate
import cfg
import json
import tiktoken
# main class for reading the pdf and communicate with openai


config = configparser.ConfigParser()
config.read('apikey.ini')
chat_api_list = config.get('OpenAI', 'OPENAI_API_KEYS')[1:-1].replace('\'', '').split(',')
os.environ["OPENAI_API_KEY"] = chat_api_list[0]

TOP_K = 20
PROMPTS = {
    'general':
        """You are tasked with the role of a climate scientist, assigned to analyze a company's sustainability report. Based on the following extracted parts from the sustainability report, answer the given QUESTIONS. 
If you don't know the answer, just say that you don't know. Don't try to make up an answer.
Format your answers in JSON format with the following keys: COMPANY_NAME, COMPANY_SECTOR, and COMPANY_LOCATION.

QUESTIONS: 
1. What is the company of the report?
2. What sector does the company belong to? 
3. Where is the company located?

=========
{context}
=========
Your FINAL_ANSWER in JSON (ensure there's no format error):
""",
    'tcfd_qa_source': """As a senior equity analyst with expertise in climate science evaluating a company's sustainability report, you are presented with the following background information:

{basic_info}

With the above information and the following extracted components (which may have incomplete sentences at the beginnings and the ends) of the sustainability report at hand, please respond to the posed question, ensuring to reference the relevant parts ("SOURCES").
Format your answer in JSON format with the two keys: ANSWER (this should contain your answer string without sources), and SOURCES (this should be a list of the source numbers that were referenced in your answer).

QUESTION: {question}
=========
{summaries}
=========

Please adhere to the following guidelines in your answer:
1. Your response must be precise, thorough, and grounded on specific extracts from the report to verify its authenticity.
2. If you are unsure, simply acknowledge the lack of knowledge, rather than fabricating an answer.
3. Keep your ANSWER within {answer_length} words.
4. Be skeptical to the information disclosed in the report as there might be greenwashing (exaggerating the firm's environmental responsibility). Always answer in a critical tone.
5. cheap talks are statements that are costless to make and may not necessarily reflect the true intentions or future actions of the company. Be critical for all cheap talks you discovered in the report.
6. Always acknowledge that the information provided is representing the company's view based on its report.
7. Scrutinize whether the report is grounded in quantifiable, concrete data or vague, unverifiable statements, and communicate your findings.
{guidelines}

Your FINAL_ANSWER in JSON (ensure there's no format error):
""",
    'user_qa_source': """As a senior equity analyst with expertise in climate science evaluating a company's sustainability report, you are presented with the following background information:

{basic_info}

With the above information and the following extracted components (which may have incomplete sentences at the beginnings and the ends) of the sustainability report at hand, please respond to the posed question, ensuring to reference the relevant parts ("SOURCES").
Format your answer in JSON format with the two keys: ANSWER (this should contain your answer string without sources), and SOURCES (this should be a list of the source numbers that were referenced in your answer).

QUESTION: {question}
=========
{summaries}
=========

Please adhere to the following guidelines in your answer:
1. Your response must be precise, thorough, and grounded on specific extracts from the report to verify its authenticity.
2. If certain information is unclear or unavailable, admit the lack of knowledge rather than devising an answer.
3. Answer the question strictly based on the provided extracts. If the information available is insufficient, clearly state that the question cannot be answered based on the given report.
4. Keep your ANSWER within {answer_length} words.
5. Be skeptical to the information disclosed in the report as there might be greenwashing (exaggerating the firm's environmental responsibility). Always answer in a critical tone.
6. cheap talks are statements that are costless to make and may not necessarily reflect the true intentions or future actions of the company. Be critical for all cheap talks you discovered in the report.
7. Always acknowledge that the information provided is representing the company's view based on its report.
8. Scrutinize whether the report is grounded in quantifiable, concrete data or vague, unverifiable statements, and communicate your findings.

Your FINAL_ANSWER in JSON (ensure there's no format error):
""",
    'tcfd_summary_source': """Your task is to analyze and summarize any disclosures related to the following <CRITICAL_ELEMENT> in a company's sustainability report:

<CRITICAL_ELEMENT>: {question}

Provided below is some basic information about the company under evaluation:

{basic_info}

In addition to the above, the following extracted sections of the sustainability report have been made available to you for review:

{summaries}

Your task is to summarize the company's disclosure of the aforementioned <CRITICAL_ELEMENT>, based on the information presented in these extracts. Please adhere to the following guidelines in your summary:
1. If the <CRITICAL_ELEMENT> is disclosed in the report, try to summarize by direct extractions from the report. Reference the source of this information from the provided extracts to confirm its credibility.
2. If the <CRITICAL_ELEMENT> is not addressed in the report, state this clearly without attempting to extrapolate or manufacture information.
3. Keep your SUMMARY within {answer_length} words.
4. Be skeptical to the information disclosed in the report as there might be greenwashing (exaggerating the firm's environmental responsibility). Always answer in a critical tone.
5. cheap talks are statements that are costless to make and may not necessarily reflect the true intentions or future actions of the company. Be critical for all cheap talks you discovered in the report.
6. Always acknowledge that the information provided is representing the company's view based on its report.
7. Scrutinize whether the report is grounded in quantifiable, concrete data or vague, unverifiable statements, and communicate your findings.
{guidelines}

Your summarization should be formatted in JSON with two keys:
1. SUMMARY: This should contain your summary without source references.
2. SOURCES: This should be a list of the source numbers that were referenced in your summary.

Your FINAL_ANSWER in JSON (ensure there's no format error):
""",
    'tcfd_qa': """As a senior equity analyst with expertise in climate science evaluating a company's sustainability report, you are presented with the following essential information about the report:

{basic_info}

With the above information and the following extracted components (which may have incomplete sentences at the beginnings and the ends) of the sustainability report at hand, please respond to the posed question. 
Your answer should be precise, comprehensive, and substantiated by direct extractions from the report to establish its credibility.
If you don't know the answer, just say that you don't know. Don't try to make up an answer.

QUESTION: {question}
=========
{summaries}
=========
""",
    'tcfd_assessment': """Your task is to rate a sustainability report's disclosure quality on the following <CRITICAL_ELEMENT>:

<CRITICAL_ELEMENT>: {question}

These are the <REQUIREMENTS> that outline the necessary components for high-quality disclosure pertaining to the <CRITICAL_ELEMENT>:

<REQUIREMENTS>:
====
{requirements}
====

Presented below are select excerpts from the sustainability report, which pertain to the <CRITICAL_ELEMENT>:

<DISCLOSURE>:
====
{disclosure}
====

Please analyze the extent to which the given <DISCLOSURE> satisfies the aforementioned <REQUIREMENTS>. Your ANALYSIS should specify which <REQUIREMENTS> have been met and which ones have not been satisfied.
Your response should be formatted in JSON with two keys:
1. ANALYSIS: A paragraph of analysis (be in a string format). No longer than 150 words.
2. SCORE: An integer score from 0 to 100. A score of 0 indicates that most of the <REQUIREMENTS> have not been met or are insufficiently detailed. In contrast, a score of 100 suggests that the majority of the <REQUIREMENTS> have been met and are accompanied by specific details.

Your FINAL_ANSWER in JSON (ensure there's no format error):
""",
    'scoring': """Your task is to rate the disclosure quality of a sustainability report. You'll be provided with a <REPORT SUMMARY> that contains {question_number} (DISCLOSURE_REQUIREMENT, DISCLOSURE_CONTENT) pairs. DICLOSURE_REQUIREMENT corresponds to a key piece of information that the report should disclose. DISCLOSURE_CONTENT summarizes the report's disclosed information on that topic. 
For each pair, you should assign a score reflecting the depth and comprehensiveness of the disclosed information. A score of 1 denotes a detailed and comprehensive disclosure. A score of 0.5 suggests that the disclosed information is lacking in detail. A score of 0 indicates that the requested information is either not disclosed or is disclosed without any detail.
Please format your response in a JSON structure, with the keys 'COMMENT' (providing your overall assessment of the report's quality) and 'SCORES' (a list containing the {question_number} scores corresponding to each question-and-answer pair).

<REPORT SUMMARY>:
====
{summaries}
====
Your FINAL_ANSWER in JSON (ensure there's no format error):
""",
  'to_question': """Examine the following statement and transform it into a question, suitable for a ChatGPT prompt, if it is not already phrased as one. If the statement is already a question, return it as it is.
Statement: {statement}"""
#     'scoring': """Your role is that of a climate scientist rating the disclosure quality of a sustainability report. You'll be provided with a <REPORT SUMMARY> that contains {question_number} question-and-answer pairs. Each pair corresponds to a key piece of information that the report should disclose, with the answer summarizing the report's disclosed information on that topic. Your responsibility is to assess the quality of these disclosures.
# For each question-and-answer pair, assign a score based on the question-anwering quality and the disclosure detailedness and comprehensiveness. If the question is thoroughly answered and the disclosed information is thoroughly detailed, assign a score of 1. If the question is only partially answered or the dsclosed information lacks substantial detail, assign a score of 0.5. If the information asked by the question is either not disclosed or disclosed without any detail, assign a score of 0.
# Please format your response in a JSON structure, with the keys 'COMMENT' (providing your overall assessment of the report's quality) and 'SCORES' (a list containing the {question_number} scores corresponding to each question-and-answer pair).

# <REPORT SUMMARY>:
# ---
# {summaries}
# ---
# FINAL_ANSWER in JSON (ensure there's no format error):
# """,
}

QUERIES = {
    'general': ["What is the company of the report?", "What sector does the company belong to?", "Where is the company located?",
                #"What climate-related issues are discussed in this report?"
          ],
    'tcfd_1': "How does the company's board oversee climate-related risks and opportunities?",
    'tcfd_2': "What is the role of management in assessing and managing climate-related risks and opportunities?",
    'tcfd_3': "What are the most relevant climate-related risks and opportunities that the organisation has identified over the short, medium, and long term? Are risks clearly associated with a horizon?",
    'tcfd_4': "How do climate-related risks and opportunities impact the organisation's businesses strategy, economic and financial performance, and financial planning?",
    'tcfd_5': "How resilient is the organisation's strategy when considering different climate-related scenarios, including a 2°C target or lower scenario? How resilient is the organisation's strategy when considering climate physical risks?",
    'tcfd_6': "What processes does the organisation use to identify and assess climate-related risks?",
    'tcfd_7': "How does the organisation manage climate-related risks?",
    'tcfd_8': "How are the processes for identifying, assessing, and managing climate-related risks integrated into the organisation's overall risk management?",
    'tcfd_9': "What metrics does the organisation use to assess climate-related risks and opportunities? How do the metrics help ensure that the performance is in line with its strategy and risk management process?",
    'tcfd_10': "Does the organisation disclose its Scope 1, Scope 2, and, if appropriate, Scope 3 greenhouse gas (GHG) emissions? What are the related risks and do they differ depending on the scope?",
    'tcfd_11': "What targets does the organisation use to understand/quantify/benchmark climate-related risks and opportunities? How is the organization performing against these targets?",
}
TCFD_ASSESSMENT = {
    'tcfd_1': """In describing the board's oversight of climate-related issues, organizations should consider including a discussion of the following:
1. processes and frequency by which the board and/or board committees (e.g., audit, risk, or other committees) are informed about climate-related issues;
2. whether the board and/or board committees consider climate-related issues when reviewing and guiding strategy, major plans of action, risk management policies, annual budgets, and business plans as well as setting the organization’s performance objectives, monitoring implementation and performance, and overseeing major capital expenditures, acquisitions, and divestitures; and 
3. how the board monitors and oversees progress against goals and targets for addressing climate-related issues.
""",
    'tcfd_2': """In describing management's role related to the assessment and management of climate-related issues, organizations should consider including the following information:
1. whether the organization has assigned climate-related responsibilities to management-level positions or committees; and, if so, whether such management positions or committees report to the board or a committee of the board and whether those responsibilities include assessing and/or managing climate-related issues;
2. a description of the associated organizational structure(s);
3. processes by which management is informed about climate-related issues; and
4. how management (through specific positions and/or management committees) monitors climate-related issues.
""",
    'tcfd_3': """In describing the climate-related risks and opportunities the organization has identified over the short, medium, and long term, organizations should provide the following information:
1. a description of what they consider to be the relevant short-, medium-, and long-term time horizons, taking into consideration the useful life of the organization's assets or infrastructure and the fact that climate-related issues often manifest themselves over the medium and longer terms;
2. a description of the specific climate-related issues potentially arising in each time horizon (short, medium, and long term) that could have a material financial impact on the organization; and
3. a description of the process(es) used to determine which risks and opportunities could have a material financial impact on the organization. 
Organizations should consider providing a description of their risks and opportunities by sector and/or geography, as appropriate.
""",
    'tcfd_4': """In describing impact of climate-related risks and opportunities on the organization's businesses, strategy, and financial planning, organizations should discuss how identified climate-related issues have affected their businesses, strategy, and financial planning. 
Organizations should consider including the impact on their businesses, strategy, and financial planning in the following areas:
1. Products and services
2. Supply chain and/or value chain
3. Adaptation and mitigation activities
4. Investment in research and development
5. Operations (including types of operations and location of facilities)
6. Acquisitions or divestments
7. Access to capital
Organizations should describe how climate-related issues serve as an input to their financial planning process, the time period(s) used, and how these risks and opportunities are prioritized. Organizations' disclosures should reflect a holistic picture of the interdependencies among the factors that affect their ability to create value over time. 
Organizations should describe the impact of climate-related issues on their financial performance (e.g., revenues, costs) and financial position (e.g., assets, liabilities). If climate-related scenarios were used to inform the organization's strategy and financial planning, such scenarios should be described.
Organizations that have made GHG emissions reduction commitments, operate in jurisdictions that have made such commitments, or have agreed to meet investor expectations regarding GHG emissions reductions should describe their plans for transitioning to a low-carbon economy, which could include GHG emissions targets and specific activities intended to reduce GHG emissions in their operations and value chain or to otherwise support the transition.
""",
    'tcfd_5': """In describing the resilience of the organization's strategy, organizations should describe how resilient their strategies are to climate-related risks and opportunities, taking into consideration a transition to a low-carbon economy consistent with a 2°C or lower scenario and, where relevant to the organization, scenarios consistent with increased physical climate-related risks.
Organizations should consider discussing:
1. where they believe their strategies may be affected by climate-related risks and opportunities; 
2. how their strategies might change to address such potential risks and opportunities;
3. the potential impact of climate-related issues on financial performance (e.g., revenues, costs) and financial position (e.g., assets, liabilities); and
4. the climate-related scenarios and associated time horizon(s) considered.
""",
    'tcfd_6': """In describing the organization's processes for identifying and assessing climate-related risks, organizations should describe their risk management processes for identifying and assessing climate-related risks. An important aspect of this description is how organizations determine the relative significance of climate-related risks in relation to other risks. 
Organizations should describe whether they consider existing and emerging regulatory requirements related to climate change (e.g., limits on emissions) as well as other relevant factors considered.
Organizations should also consider disclosing the following:
1. processes for assessing the potential size and scope of identified climate-related 
risks and
2. definitions of risk terminology used or references to existing risk classification 
frameworks used.
""",
    'tcfd_7': """In describing  the organization's processes for managing climate-related risks, organizations should describe their processes for managing climate-related risks, including how they make decisions to mitigate, transfer, accept, or control those risks. In addition, organizations should describe their processes for prioritizing climate-related risks, including how materiality determinations are made within their organizations. 
""",
    'tcfd_8': """In describing how processes for identifying, assessing, and managing climate-related risks are integrated into the organization's overall risk management, organizations should describe how their processes for identifying, assessing, and managing climate-related risks are integrated into their overall risk management.
""",
    'tcfd_9': """In describing the metrics used by the organization to assess climate-related risks and opportunities in line with its strategy and risk management process, organizations should provide the key metrics used to measure and manage climate-related risks and opportunities, as well as metrics consistent with the cross-industry.
Organizations should consider including metrics on climate-related risks associated with water, energy, land use, and waste management where relevant and applicable.
Where climate-related issues are material, organizations should consider describing whether and how related performance metrics are incorporated into remuneration policies.
Where relevant, organizations should provide their internal carbon prices as well as climate-related opportunity metrics such as revenue from products and services designed for a low-carbon economy. 
Metrics should be provided for historical periods to allow for trend analysis. Where appropriate, organizations should consider providing forward-looking metrics for the cross-industry, consistent with their business or strategic planning time horizons. In addition, where not apparent, organizations should provide a description of the methodologies used to calculate or estimate climate-related metrics.
""",
    'tcfd_10': """In disclosing Scope 1, Scope 2, and, if appropriate, Scope 3 greenhouse gas (GHG) emissions, and the related risks, organizations should provide their Scope 1 and Scope 2 GHG emissions independent of a materiality assessment, and, if appropriate, Scope 3 GHG emissions and the related risks. All organizations should consider disclosing Scope 3 GHG emissions.
GHG emissions should be calculated in line with the GHG Protocol methodology to allow for aggregation and comparability across organizations and jurisdictions. As appropriate, organizations should consider providing related, generally accepted industry-specific GHG efficiency ratios.
GHG emissions and associated metrics should be provided for historical periods to allow for trend analysis. In addition, where not apparent, organizations should provide a description of the methodologies used to calculate or estimate the metrics.
""",
    'tcfd_11': """In describing the targets used by the organization to manage climate-related risks and opportunities and performance against targets, organizations should describe their key climate-related targets such as those related to GHG emissions, water usage, energy usage, etc., in line with the cross-industry, where relevant, and in line with anticipated regulatory requirements or market constraints or other goals. Other goals may include efficiency or financial goals, financial loss tolerances, avoided GHG emissions through the entire product life cycle, or net revenue goals for products and services designed for a low-carbon economy. 
In describing their targets, organizations should consider including the following:
1. whether the target is absolute or intensity based;
2. time frames over which the target applies;
3. base year from which progress is measured; and
4. key performance indicators used to assess progress against targets.
Organizations disclosing medium-term or long-term targets should also disclose associated interim targets in aggregate or by business line, where available.
Where not apparent, organizations should provide a description of the methodologies used to calculate targets and measures.
""",
}
TCFD_GUIDELINES = {
    'tcfd_1': """8. Please concentrate on the board's direct responsibilities and actions pertaining to climate issues, without discussing the company-wide risk management system or other topics.
""",
    'tcfd_2': """8. Please focus on their direct duties related to climate issues, without introducing other topics such as the broader corporate risk management system.
""",
    'tcfd_3': """8. Avoid discussing the company-wide risk management system or how these risks and opportunities are identified and managed.
""",
    'tcfd_4': """8. Please do not include the process of risk identification, assessment or management in your answer.
""",
    'tcfd_5': """8. In your response, focus solely on the resilience of strategy in these scenarios, and refrain from discussing processes of risk identification, assessment, or management strategies.
""",
    'tcfd_6': """8. Restrict your answer to the identification and assessment processes, without discussing the management or integration of these risks.
""",
    'tcfd_7': """8. Please focus on the concrete actions and strategies implemented to manage these risks, excluding the process of risk identification or assessment.
""",
    'tcfd_8': """8. Please focus on the integration aspect and avoid discussing the process of risk identification, assessment, or the specific management actions taken.
""",
    'tcfd_9': """8. Do not include information regarding the organization's general risk identification and assessment methods or their broader corporate strategy and initiatives.
""",
    'tcfd_10': """8. Confirm whether the organisation discloses its Scope 1, Scope 2, and, if appropriate, Scope 3 greenhouse gas (GHG) emissions. If so, provide any available data or specific figures on these emissions. Additionally, identify the related risks. The risks should be specific to the GHG emissions rather than general climate-related risks.
""",
    'tcfd_11': """8. Please detail the precise targets and avoid discussing the company's general risk identification and assessment methods or their commitment to disclosure through the TCFD.
""",
}
SYSTEM_PROMPT = "You are an AI assistant in the role of a Senior Equity Analyst with expertise in climate science that analyzes companys' sustainability reports."


def remove_brackets(string):
    return re.sub(r'\([^)]*\)', '', string).strip()


def _docs_to_string(docs, num_docs=TOP_K, with_source=True):
    output = ""
    docs = docs[:num_docs]
    for doc in docs:
        output += "Content: {}\n".format(doc.page_content)
        if with_source:
            output += "Source: {}\n".format(doc.metadata['source'])
        output += "\n---\n"
    return output


def _find_answer(string, name="ANSWER"):
    for l in string.split('\n'):
        if name in l:
            start = l.find(":") + 3
            end = len(l) - 1
            return l[start:end]
    return string


def _find_sources(string):
    pattern = r'\d+'
    numbers = [int(n) for n in re.findall(pattern, string)]
    return numbers


def _find_float_numbers(string):
    pattern = r"[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?"
    float_numbers = [float(n) for n in re.findall(pattern, string)]
    return float_numbers


def _find_score(string):
    for l in string.split('\n'):
        if "SCORE" in l:
            d = re.search(r'[-+]?\d*\.?\d+', l)
            break
    return d[0]


class Reader:
    def __init__(self, llm_name='gpt-3.5-turbo', answer_key_name='ANSWER', max_token=512, q_name='Q', a_name='A',
                 queries=QUERIES, qa_prompt='tcfd_qa_source', guidelines=TCFD_GUIDELINES,
                 assessments=TCFD_ASSESSMENT,
                 answer_length='60',
                 root_path='./',
                 gitee_key='',
                 user_name='defualt', language='en'):
        self.user_name = user_name  # user name
        self.language = language
        self.root_path = root_path
        self.max_token = max_token
        self.llm_name = llm_name
        #
        self.tiktoken_encoder = tiktoken.encoding_for_model(self.llm_name)
        self.cur_api = 0
        self.file_format = 'md'  # or 'txt'
        self.prompts = PROMPTS
        self.assessments = assessments
        self.queries = queries
        self.guidelines = guidelines
        self.qa_prompt = qa_prompt
        self.answer_key_name = answer_key_name
        self.q_name = q_name
        self.a_name = a_name
        self.answer_length = answer_length
        self.basic_info_answers = []
        self.answers = []
        self.assessment_results = []
        self.user_questions = []
        self.user_answers = []
        # self.save_image = False
        # if self.save_image:
        #    self.gitee_key = self.config.get('Gitee', 'api')
        # else:
        #    self.gitee_key = ''

    async def qa_with_chat(self, report_list):
        htmls = []
        for report_index, report in enumerate(report_list):
            basic_info_prompt = PromptTemplate(template=self.prompts['general'], input_variables=["context"])
            if "turbo" in self.llm_name:
                # title = "Title: " + report.title + '\n'
                # first_page = "First Page: " + report.pdf[0].get_text() + '\n'
                message = [
                    SystemMessage(content=SYSTEM_PROMPT),
                    HumanMessage(content=basic_info_prompt.format(
                        context=_docs_to_string(report.section_text_dict['general'], with_source=False)))
                ]
                llm = ChatOpenAI(temperature=0, max_tokens=256)
                output_text = llm(message).content
            else:
                message = basic_info_prompt.format(
                    context=_docs_to_string(report.section_text_dict['general'], with_source=False))
                llm = OpenAI(temperature=0, max_tokens=256)
                output_text = llm(message)
            print(output_text)
            try:
                basic_info_dict = json.loads(output_text)
            except ValueError as e:
                basic_info_dict = {'COMPANY_NAME': _find_answer(output_text, name='COMPANY_NAME'),
                                   'COMPANY_SECTOR': _find_answer(output_text, name='COMPANY_SECTOR'),
                                   'COMPANY_LOCATION': _find_answer(output_text, name='COMPANY_LOCATION')}
            basic_info_string = """Company name: {name}\nCompany sector: {sector}\nCompany Location: {location}""" \
                .format(name=basic_info_dict['COMPANY_NAME'], sector=basic_info_dict['COMPANY_SECTOR'],
                        location=basic_info_dict['COMPANY_LOCATION'])
            self.basic_info_answers.append(basic_info_dict)

            tcfd_questions = {k: v for k, v in self.queries.items() if 'tcfd' in k}
            tcfd_prompt = PromptTemplate(template=self.prompts[self.qa_prompt],
                                         input_variables=["basic_info", "summaries", "question", "guidelines",
                                                          "answer_length"])
            answers = {}
            messages = []
            keys = []
            for k, q in tcfd_questions.items():
                num_docs = 20
                current_prompt = tcfd_prompt.format(basic_info=basic_info_string,
                                                    summaries=_docs_to_string(report.section_text_dict[k]),
                                                    question=q, guidelines=self.guidelines[k],
                                                    answer_length=self.answer_length)
                if '16k' not in self.llm_name:
                    while len(self.tiktoken_encoder.encode(current_prompt)) > 3500 and num_docs > 10:
                        num_docs -= 1
                        current_prompt = tcfd_prompt.format(basic_info=basic_info_string, summaries=_docs_to_string(report.section_text_dict[k], num_docs=num_docs), question=q, guidelines=self.guidelines[k], answer_length=self.answer_length)
                if "turbo" in self.llm_name:
                    message = [
                        SystemMessage(content=SYSTEM_PROMPT),
                        HumanMessage(content=current_prompt)
                    ]
                else:
                    message = current_prompt
                keys.append(k)
                messages.append(message)
            if "turbo" in self.llm_name:
                llm = ChatOpenAI(temperature=0, max_tokens=512)
            else:
                llm = OpenAI(temperature=0, max_tokens=512)
            outputs = await llm.agenerate(messages)
            output_texts = {k: g[0].text for k, g in zip(keys, outputs.generations)}

            for k, text in output_texts.items():
                try:
                    answers[k] = json.loads(text)
                    if 'SOURCES' not in answers[k].keys() or self.answer_key_name not in answers[k].keys():
                        raise ValueError("Key name(s) not defined!")
                except ValueError as e:
                    answers[k] = {self.answer_key_name: _find_answer(text, name=self.answer_key_name),
                                  'SOURCES': _find_sources(text)}
                page_source = []
                for s in answers[k]['SOURCES']:
                    try:
                        page_source.append(report.page_idx[s])
                    except Exception as e:
                        pass
                answers[k]['PAGE'] = list(set(page_source))
                answers[k][self.answer_key_name] = remove_brackets(answers[k][self.answer_key_name])
                print(answers[k])
            self.answers.append(answers)

            questionnaire_governance = ""
            questionnaire_strategy = ""
            questionnaire_risk = ""
            questionnaire_metrics = ""
            for idx, (k, q) in enumerate(tcfd_questions.items()):
                if 2 > idx >= 0:
                    if idx == 0:
                        questionnaire_governance += "In governance:\n\n"
                    questionnaire_governance += self.q_name + "{}: {}\n\n".format(int(idx + 1), q)
                    questionnaire_governance += self.a_name + "{}: {}\n\n".format(int(idx + 1),
                                                                                  answers[k][self.answer_key_name])
                    questionnaire_governance += "\n"
                elif 5 > idx >= 2:
                    if idx == 2:
                        questionnaire_strategy += "In strategy:\n\n"
                    questionnaire_strategy += self.q_name + "{}: {}\n\n".format(int(idx + 1), q)
                    questionnaire_strategy += self.a_name + "{}: {}\n\n".format(int(idx + 1),
                                                                                answers[k][self.answer_key_name])
                    questionnaire_strategy += "\n"
                elif 8 > idx >= 5:
                    if idx == 5:
                        questionnaire_risk += "In risk management:\n\n"
                    questionnaire_risk += self.q_name + "{}: {}\n\n".format(int(idx + 1), q)
                    questionnaire_risk += self.a_name + "{}: {}\n\n".format(int(idx + 1),
                                                                            answers[k][self.answer_key_name])
                    questionnaire_risk += "\n"
                elif idx >= 8:
                    if idx == 8:
                        questionnaire_metrics += "In metrics and targets:\n\n"
                    questionnaire_metrics += self.q_name + "{}: {}\n\n".format(int(idx + 1), q)
                    questionnaire_metrics += self.a_name + "{}: {}\n\n".format(int(idx + 1),
                                                                               answers[k][self.answer_key_name])
                    questionnaire_metrics += "\n"
            questionnaire = questionnaire_governance + questionnaire_strategy + questionnaire_risk + questionnaire_metrics

            htmls.append(markdown.markdown(questionnaire))
        return htmls

    async def analyze_with_chat(self, report_list):
        htmls = []
        for report_index, report in enumerate(report_list):
            tcfd_assessment_prompt = PromptTemplate(template=self.prompts['tcfd_assessment'],
                                                    input_variables=["question", "requirements", "disclosure"])
            tcfd_questions = {k: v for k, v in self.queries.items() if 'tcfd' in k}
            assessments = {}
            messages = []
            keys = []
            for idx, k in enumerate(self.assessments.keys()):
                num_docs = 20
                current_prompt = tcfd_assessment_prompt.format(question=self.queries[k],
                                                               requirements=self.assessments[k],
                                                               disclosure=_docs_to_string(
                                                                   report.section_text_dict[k], with_source=False))
                if '16k' not in self.llm_name:
                    while len(self.tiktoken_encoder.encode(current_prompt)) > 3200 and num_docs > 10:
                        num_docs -= 1
                        current_prompt = tcfd_assessment_prompt.format(question=self.queries[k],
                                                                       requirements=self.assessments[k],
                                                                       disclosure=_docs_to_string(
                                                                           report.section_text_dict[k],
                                                                           num_docs=num_docs,
                                                                           with_source=False))
                if "turbo" in self.llm_name:
                    message = [
                        SystemMessage(content=SYSTEM_PROMPT),
                        HumanMessage(content=current_prompt)
                    ]
                else:
                    message = current_prompt
                keys.append(k)
                messages.append(message)
            if "turbo" in self.llm_name:
                llm = ChatOpenAI(temperature=0, max_tokens=512)
            else:
                llm = OpenAI(temperature=0, max_tokens=512)
            outputs = await llm.agenerate(messages)
            output_texts = {k: g[0].text for k, g in zip(keys, outputs.generations)}

            for k, text in output_texts.items():
                try:
                    assessments[k] = json.loads(text)
                    if 'SCORE' not in assessments[k].keys() or 'ANALYSIS' not in assessments[k].keys():
                        raise ValueError("Key name(s) not defined!")
                except ValueError as e:
                    assessments[k] = {'ANALYSIS': _find_answer(text, name='ANALYSIS'),
                                      'SCORE': _find_score(text)}
                analysis_text = remove_brackets(assessments[k]['ANALYSIS'])
                if "<CRITICAL_ELEMENT>" in analysis_text:
                    analysis_text = analysis_text.replace("<CRITICAL_ELEMENT>", "TCFD recommendation point")
                if "<DISCLOSURE>" in analysis_text:
                    analysis_text = analysis_text.replace("<DISCLOSURE>", "report's disclosure")
                if "<REQUIREMENTS>" in analysis_text:
                    analysis_text = analysis_text.replace("<REQUIREMENTS>", "TCFD guidelines")
                assessments[k]['ANALYSIS'] = analysis_text
                print(assessments[k])
            self.assessment_results.append(assessments)

            questionnaire_governance = ""
            questionnaire_strategy = ""
            questionnaire_risk = ""
            questionnaire_metrics = ""
            for idx, (k, q) in enumerate(tcfd_questions.items()):
                if 2 > idx >= 0:
                    if idx == 0:
                        questionnaire_governance += "In governance:\n\n"
                    questionnaire_governance += self.q_name + "{}: {}\n\n".format(int(idx + 1), q)
                    questionnaire_governance += "Analysis{}: {}\n\n".format(int(idx + 1), assessments[k]['ANALYSIS'])
                    questionnaire_governance += "Score{}: {}\n\n".format(int(idx + 1), assessments[k]['SCORE'])
                    questionnaire_governance += "\n"
                elif 5 > idx >= 2:
                    if idx == 2:
                        questionnaire_strategy += "In strategy:\n\n"
                    questionnaire_strategy += self.q_name + "{}: {}\n\n".format(int(idx + 1), q)
                    questionnaire_strategy += "Analysis{}: {}\n\n".format(int(idx + 1), assessments[k]['ANALYSIS'])
                    questionnaire_strategy += "Score{}: {}\n\n".format(int(idx + 1), assessments[k]['SCORE'])
                    questionnaire_strategy += "\n"
                elif 8 > idx >= 5:
                    if idx == 5:
                        questionnaire_risk += "In risk management:\n\n"
                    questionnaire_risk += self.q_name + "{}: {}\n\n".format(int(idx + 1), q)
                    questionnaire_risk += "Analysis{}: {}\n\n".format(int(idx + 1), assessments[k]['ANALYSIS'])
                    questionnaire_risk += "Score{}: {}\n\n".format(int(idx + 1), assessments[k]['SCORE'])
                    questionnaire_risk += "\n"
                elif idx >= 8:
                    if idx == 8:
                        questionnaire_metrics += "In metrics and targets:\n\n"
                    questionnaire_metrics += self.q_name + "{}: {}\n\n".format(int(idx + 1), q)
                    questionnaire_metrics += "Analysis{}: {}\n\n".format(int(idx + 1), assessments[k]['ANALYSIS'])
                    questionnaire_metrics += "Score{}: {}\n\n".format(int(idx + 1), assessments[k]['SCORE'])
                    questionnaire_metrics += "\n"
            questionnaire = questionnaire_governance + questionnaire_strategy + questionnaire_risk + questionnaire_metrics
            all_scores = [float(s['SCORE']) for s in assessments.values()]

            htmls.append(markdown.markdown(questionnaire + '\n\n' + "Average score: {}".format(sum(all_scores) / 11)))
        return htmls
           


