import re
import dateparser
import pandas as pd
from typing import List
from collections import OrderedDict
from data_structure import MedicalRecord
from langchain.text_splitter import CharacterTextSplitter


def digest_doc(parser, data: List, max_length: int = 1000):
    """
    Extract id, timestamp for each item of record based on minimal information model.
    :param parser: parser object from parsers module
    :param data: a list of record. Each record should follow the format like this: {'idx': idx, 'text': text}
    :param max_length: max length of llm's input
    :return: list of record with more information
    """
    doc_spliter = CharacterTextSplitter(chunk_size=max_length)
    doc_list = list()
    for line in data:
        patient_ids = list()
        records = list()
        if len(line['text']) < max_length:
            mini_info = parser.extract_mini_info(line['text'])
            patient_ids.extend(mini_info['patient_id'])
            records.extend([x for x in mini_info['record'] if dateparser.parse(x['record_datetime']) is not None])
        else:
            content_docs = doc_spliter.create_documents([line['text']])
            for sub_content in content_docs:
                mini_info = parser.extract_mini_info(sub_content.page_content)
                if 'patient_id' in mini_info:
                    patient_ids.extend(mini_info['patient_id'])
                if 'record' in mini_info:
                    records.extend([x for x in mini_info['record'] if dateparser.parse(x['record_datetime']) is not None])
        patient_ids = list(set(tuple(d.items()) for d in patient_ids))
        unique_recs = OrderedDict()
        for d in records:
            if d['record_datetime'] not in unique_recs:
                unique_recs[d['record_datetime']] = list()
            unique_recs[d['record_datetime']].append(d['title'])
        if len(unique_recs) > 1:
            components = list()
            for ts, rec in unique_recs.items():
                m = re.search('\n(.*?' + ts + ')', line['text'])
                if m is None:
                    continue
                offset = m.start(1)
                components.append([offset, ts, ','.join(rec)])
            for i, compo in enumerate(components):
                if i == 0:
                    content = line['text'][: components[i+1][0]]
                elif i == (len(components) - 1):
                    content = line['text'][compo[0]:]
                else:
                    content = line['text'][compo[0]: components[i+1][0]]
                doc = {'id': patient_ids, 'timestamp': compo[1], 'title': compo[2], 'content': content}
                doc_list.append(doc)
        else:
            doc = {'id': patient_ids, 'timestamp': list(unique_recs.keys())[0],
                   'title': ','.join(list(unique_recs.values())[0]), 'content': line['text']}
            doc_list.append(doc)
    return doc_list


def load_std_data(parser, data_path: str):
    """
    Load demo data in standard format
    :param parser: parser object from parsers module
    :param data_path: xlsx data path. Each row should contain id, timestamp, title and content.
    :return: a dict of MedicalRecord objects
    """
    mock_data = pd.read_excel(data_path, dtype=str)
    records = dict()
    for row in mock_data.values.tolist():
        idx = row[0]
        timestamp = row[1]
        title = row[2]
        content = row[3]
        if idx not in records:
            records[idx] = MedicalRecord(idx)
        records[idx].add_record(parser, timestamp, title, content)
    for record in records.values():
        record.create_docs()
    return records
