import os
import load_data
import query_record
import answer_question
import pandas as pd
from parsers import Parser

from llms import ChatGLM
llm = ChatGLM()
llm.load_model("THUDM/chatglm-6b-int8")

# from langchain.chat_models import ChatOpenAI
# llm = ChatOpenAI()

llm_parser = Parser(llm)

data_dir = './data'
mock_data_file = 'std_mock.xlsx'


def demo_parse_raw_data():
    doc_files = list()
    for file in os.listdir(data_dir):
        if file.endswith('txt'):
            doc_files.append(os.path.join(data_dir, file))
    doc_data = list()
    for idx, file in enumerate(doc_files):
        with open(file, 'r') as f:
            doc_data.append({'idx': idx, 'text': f.read()})
    doc_list = load_data.digest_doc(llm_parser, doc_data)
    doc_df = pd.DataFrame(doc_list)
    doc_df.to_excel(os.path.join(data_dir, mock_data_file))


def demo():
    records = load_data.load_std_data(llm_parser, os.path.join(data_dir, mock_data_file))

    target_patient = query_record.query_patient(records, '10086')
    condition = '入院后的第2次红细胞检验'
    # condition_terms = ['多发性骨髓瘤患者', '确诊时', '检查检测项目', '血液检查', '血钙水平', '总血清钙水平', '血清校正总钙水平']
    # condition = ','.join(condition_terms)
    target_record = query_record.query_condition(llm_parser, target_patient, condition)

    output = answer_question.answer(llm_parser, target_record, condition)
    print(output)


if __name__ == '__main__':
    demo()
