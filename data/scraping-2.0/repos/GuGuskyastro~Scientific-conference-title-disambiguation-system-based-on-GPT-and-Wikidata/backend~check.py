import yaml, os
from langchain import PromptTemplate,LLMChain
from backend.agent_utils import AgentUtils
from backend.api_connector import APIConnector
from backend.main import Agent

model = 'gpt-4'
connector = APIConnector(model_name=model)
llm = connector.llm
client = connector.client
utils = AgentUtils(llm=llm, client=client)
agent = Agent(model_name=model)


def check_metadata(filename):
    """
    Check the metadata according to the Qid in the output result to prevent GPT from making mistakes when summarizing the answer

    Args:
        filename (str): file need to be check.

    """

    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as test_data:
            test_data = yaml.safe_load(test_data)
    else:
        test_data = []

    for i in range(0, len(test_data)):
        if test_data[i]['Conference Info']['Conference Qid'] is not None:
            if '"' in test_data[i]['Conference Info']['Conference Qid']:
                test_data[i]['Conference Info']['Conference Qid'] = test_data[i]['Conference Info']['Conference Qid'].strip('"')

            query_result = utils.qid_query(test_data[i]['Conference Info']['Conference Qid'])

            test_data[i]['Conference Info']['Conference startDate'] = query_result[0]['StartDate']
            test_data[i]['Conference Info']['Conference endDate'] = query_result[0]['EndDate']
            test_data[i]['Conference Info']['Conference location'] = query_result[0]['Location']
            test_data[i]['Conference Info']['Conference officialWebsite'] = query_result[0]['OfficialWebsite']

    with open(filename, 'w', encoding="utf-8") as file:
        yaml.dump(test_data, file, default_flow_style=False)

def check_match(filename):
    """
        Send the output metadata and the original citation to GPT to determine whether it really matches.

        Args:
            filename (str): file need to be check.

        Returns:
            The re-judgment result of GPT.

    """

    template = os.path.join(os.path.dirname(__file__), 'templates.yaml').replace("\\", "/")
    with open(template, 'r', encoding='utf-8') as file:
        templates = yaml.safe_load(file)

    check_template = templates['check_template']

    llm_chain = LLMChain(
        llm=llm,
        prompt=PromptTemplate.from_template(check_template)
    )

    input_list = []

    with open(filename, "r", encoding="utf-8") as test_data:
        test_data = yaml.safe_load(test_data)

    for i in range(0, len(test_data)):
        if test_data[i]['Conference Info']['Conference Qid'] is not None:
            citation_text = test_data[i]['Citation text']
            conferneceTitle_in_citation = test_data[i]['Conference Info']['Conference title']
            text1 = "Text1.The original citation text:" + citation_text + " The conference in this citation:" + conferneceTitle_in_citation

            query_result = utils.qid_query(test_data[i]['Conference Info']['Conference Qid'])
            title = query_result[0]['Title']
            startDate = query_result[0]['StartDate']
            endDate = query_result[0]['EndDate']
            text2 = " | Text2.Matched possible conference metadata:" + ' Title:' + title + ';StartDate:' + startDate + ';EndDate:' + endDate

            megerd = text1 + text2

            input_list.append({"infomation": megerd})

    return llm_chain.apply(input_list)


def correction_replace_empty(filename,filename_after_check):
    """
            Correction method 1 based on the check results, this method will empty the error entry.

            Args:
                filename (str): file need to be check.
                filename_after_check (str): file after correction

            Returns:
                file corrected after using this method.

    """

    match_result = check_match(filename)
    print(match_result)
    check_list = []

    with open(filename, "r", encoding="utf-8") as data:
        result = yaml.safe_load(data)

    for i in range(0, len(result)):
        if result[i]['Conference Info']['Conference Qid'] is not None:
            check_list.append(result[i]['Conference Info']['Conference Qid'])

    for i in range(0, len(check_list)):
        if match_result[i]['text'] == 'Wrong':
            for n in range(0, len(result)):
                if result[n]['Conference Info']['Conference Qid'] == check_list[i]:
                    result[n]['Conference Info']['Conference Qid'] = None
                    result[n]['Conference Info']['Conference startDate'] = None
                    result[n]['Conference Info']['Conference endDate'] = None
                    result[n]['Conference Info']['Conference location'] = None
                    result[n]['Conference Info']['Conference officialWebsite'] = None

    with open(filename_after_check, 'a', encoding="utf-8") as result_all_after_check_file:
        yaml.dump(result, result_all_after_check_file)


def correction_call_GPT4(filename,filename_after_check):
    """
        Correction method 2 based on the check results. This method re-calls GPT-4 to process the error entry and fills with the GPT-4 parsing result.

        Args:
            filename (str): file need to be check.
            filename_after_check (str): file after correction

        Returns:
            file corrected after using this method.

    """
    match_result = check_match(filename)
    print(match_result)
    check_list = []

    with open(filename, "r", encoding="utf-8") as data:
        result = yaml.safe_load(data)

    for i in range(0, len(result)):
        if result[i]['Conference Info']['Conference Qid'] is not None:
            check_list.append(result[i]['Conference Info']['Conference Qid'])

    for i in range(0, len(check_list)):
        if match_result[i]['text'] == 'Wrong':
            for n in range(0, len(result)):
                if result[n]['Conference Info']['Conference Qid'] == check_list[i]:
                    text = result[n]['Citation text']
                    Agent.generate_result(agent, text, show_token=True, use_integrate_agent=True,output_file='GPT4_correction.yaml')
                    with open('GPT4_correction.yaml', "r", encoding="utf-8") as corr_data:
                        correction = yaml.safe_load(corr_data)
                    result[n]['Conference Info']['Conference Qid'] = correction[0]['Conference Info']['Conference Qid']
                    result[n]['Conference Info']['Conference startDate'] = correction[0]['Conference Info']['Conference startDate']
                    result[n]['Conference Info']['Conference endDate'] = correction[0]['Conference Info']['Conference endDate']
                    result[n]['Conference Info']['Conference location'] = correction[0]['Conference Info']['Conference location']
                    result[n]['Conference Info']['Conference officialWebsite'] = correction[0]['Conference Info']['Conference officialWebsite']



    with open(filename_after_check, 'a', encoding="utf-8") as result_all_after_check_file:
        yaml.dump(result, result_all_after_check_file)