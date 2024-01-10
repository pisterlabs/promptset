import json
import logging
import os
import re

from src.conversation_bot.document_ingestion.ingestion import IngestionToVectorDb as ingestion

data_dir = "/mnt/e/Personal/Samarth/repository/DMAC_ChatGPT/data/"
audio_conversations = os.path.join(data_dir, "Audio_data_2.json")


def read_audio_data():
    with open(audio_conversations, 'r') as fd:
        data = json.load(fd)
    return data['Data']


def load_employee_conversations(employee_name):
    conversation_dict = dict()
    conversation_dict['data'] = list()
    conv_data = read_audio_data()

    for records in conv_data:
        if records['content']['Assigned'] == employee_name:
            text = records['content']['text']
            text = re.sub("\s\s+", " ", text)
            text = "conversation between Customer {} and {} from sales:{}".format(records['content']['Customer'],
                                                                                  employee_name, text)
            conversation_dict['data'].append({
                'name': employee_name,
                'text': text,
                'customer': records['content']['Customer'],
                'date': '',
                'call duration': ''
            })
    if len(conversation_dict['data']) > 0:
        tmp_filename = os.path.join(data_dir, 'conversation.json')
        with open(tmp_filename, 'w+') as fc:
            json.dump(conversation_dict, fc, indent=4)
        return tmp_filename
    else:
        logging.warning("No conversations are found to analyze")
        return 0


def load_customer_conversations(customer_name):
    conversation_dict = dict()
    conversation_dict['data'] = list()
    conv_data = read_audio_data()

    for records in conv_data:
        if records['content']['Customer'] == customer_name:
            text = records['content']['text']
            text = re.sub("\s\s+", " ", text)
            text = "conversation between Customer {} and {} from sales:{}".format(customer_name,
                                                                                  records['content']['Assigned'], text)
            conversation_dict['data'].append({
                'name': customer_name,
                'text': text,
                'customer': records['content']['Customer'],
                'date': '',
                'call duration': ''
            })
    if len(conversation_dict['data']) > 0:
        tmp_filename = os.path.join(data_dir, 'conversation.json')
        with open(tmp_filename, 'w+') as fc:
            json.dump(conversation_dict, fc, indent=4)
        return tmp_filename
    else:
        logging.warning("No conversations are found to analyze")
        return 0


def load_embeddings(docs_path, embedder='openai', out_dir='faiss_damac_conversation'):
    out_dir = os.path.join(data_dir, out_dir)

    docs = ingestion.data_loader(docs_path)
    if embedder == 'openai':
        from langchain.embeddings.openai import OpenAIEmbeddings
        embeddings = OpenAIEmbeddings()
    else:
        from src.conversation_bot.utils.embedder_model import HuggingFaceEmbeddings
        embeddings = HuggingFaceEmbeddings()

    vector_store = ingestion.save_to_local_vectorstore(docs, embedding=embeddings)
    vector_store.save_local(out_dir)
    return out_dir


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='Data Loader for conversations')
    # parser.add_argument('-n', '--employee', type=str, default=None, help='name of the employee', required=True)
    # parser.add_argument('-e','--embedding',type=str,default='openai',help = 'Embedding provided',required=False)
    # args = parser.parse_args()
    load_embeddings(load_employee_conversations(employee_name='Juraira Manzoor'), embedder='openai')
