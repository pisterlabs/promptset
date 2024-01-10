#读取一个excel列表，读取第二列question及第三列answer，并将其‘\n’拼接，
# 生成一个json文件，格式如下：{
    #     "text": record["text"],
    #     "meta": {
    #         "timestamp": record["timestamp"],
    #         "url": record["url"],
    #         "language": "en",
    #         "source": "c4",
    #         "category":"commoncrawl"
    #     }
    # }
import pandas as pd
import json
import sys
from datetime import datetime
from tqdm import tqdm
import argparse
from functools import partial
from pathlib import Path
import os
from langchain.document_loaders import UnstructuredFileLoader, TextLoader
from langchain.document_loaders import UnstructuredEPubLoader #pip install pandoc
from langchain.document_loaders.csv_loader import CSVLoader, UnstructuredCSVLoader
from langchain.document_loaders import UnstructuredExcelLoader
from langchain.document_loaders import UnstructuredXMLLoader
from langchain.document_loaders import UnstructuredPowerPointLoader 
from langchain.document_loaders import Docx2txtLoader, UnstructuredWordDocumentLoader #pip install docx2txt
from langchain.document_loaders import PyPDFLoader #pip install PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from joblib import Parallel, delayed









parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default="/focusdata/llm_basedata/c4/en")
parser.add_argument('--output_dir', type=str, default="/focusdata/llm_basedata/c4/processed")#docx_dir
parser.add_argument('--threads', type=int, default=1)
parser.add_argument('--count_token', action='store_true', default=False)
parser.add_argument('--zh', action='store_true', default=False)

args = parser.parse_args()




def get_timestamp() -> str:
    return datetime.now().isoformat()


def process_record(text, url=None, language='en', source='c4', category='commoncrawl'):
    r_url = ''
    if url is not None:
        r_url = url
    return {
        "text": text,
        "meta": {
            "timestamp": get_timestamp(),
            "url": url,
            "language": language,
            "source": source,
            "category":category
        }
    }



def read_csv_origin(data_path,out_fp):

    if data_path.endswith('csv'):
        try:
            df = pd.read_csv(data_path, encoding='utf-8')
        except:
            df = pd.read_csv(data_path,encoding='gbk')
    elif data_path.endswith('xlsx'):
        df = pd.read_excel(data_path)
    #column_names = df.columns.tolist()  ##获取列标签
    datas = df.values.tolist()  ## 获取所有内容

    res = []
   
    for data in tqdm(datas):
        
        Q,  A = data[1],  data[2]  #English prod_trait after processed，加上了index
        
        if len(A)==0:
            #A为空则跳过
            print('A为空，skip it ....')
            continue
        url = data[0]
        QA = str(Q) + "\n" + str(A)
        res.append(process_record(QA, url=url))

    return res

def uncompress_files(data_path):

    output_dir = args.output_dir
    if not os.path.exists(args.output_dir):
        os.makedirs('tmp')
        output_dir = 'tmp'

    if data_path.endswith('zip'):
        os.system(f'unzip {data_path} -d {args.data_dir}' )
        os.system(f'mv {data_path}  {output_dir}')
    elif data_path.endswith('rar'):
        os.system(f'unrar e -r {data_path} {args.data_dir} ')
        os.system(f'mv {data_path}  {output_dir}')
    elif data_path.endswith('tar.gz'):
        os.system(f'tar -zxvf {data_path} -C {args.data_dir} ')
        os.system(f'mv {data_path}  {output_dir}')
    else:
        pass


def select_loader(data_path):
    
    if data_path.endswith('csv'):
        loader = UnstructuredCSVLoader(data_path)
    elif data_path.endswith('xlsx'):
        loader = UnstructuredExcelLoader(data_path)
    elif data_path.endswith('epub'):
        loader = UnstructuredEPubLoader(data_path)
    elif  data_path.endswith('docx') or data_path.endswith('DOCX') or data_path.endswith('doc') or data_path.endswith('DOC'):
        loader = UnstructuredWordDocumentLoader(data_path)       
    elif data_path.endswith('ppt') or data_path.endswith('pptx'):
        loader = UnstructuredPowerPointLoader(data_path)    
    elif data_path.endswith('pdf'):
        loader = PyPDFLoader(data_path)
    elif data_path.endswith('txt'):        
        loader = TextLoader(data_path)    
    elif data_path.endswith('zip') or data_path.endswith('rar') or data_path.endswith('tar.gz'):
        uncompress_files(data_path)
        loader = None
    else:
        #raise TypeError('文件格式不支持,仅支持word pdf txt ppt excel csv epub')
        print('文件格式不支持,仅支持word pdf txt ppt excel csv epub',os.path.basename(data_path))
        loader = None
    return loader


def read_files(data_path):
    #读取文档内容，并将内容划分为1500字符一段，400字符重叠，返回list

    assert os.path.isfile(data_path), '文件路径需要是file'
    loader = select_loader(data_path)
    if loader is None:        
        return ''
    elif isinstance(loader,str):
        return loader

    data_list = loader.load() #返回值是list,包含Document对象，Document对象包含page_content,metadata,metadata包含source=file_name

    res = ''
    for page in  data_list:
        text = page.page_content
        
        res += text 

    #print('res is ', res)        
    return res
    

# def read_process_files(data_path, out_fp):
#     #读取文档内容，并将内容划分为1500字符一段，400字符重叠，返回list
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1800, chunk_overlap=400)

#     loader = select_loader(data_path)
#     if loader is None:
#         return None
#     elif isinstance(loader,str):
#         docs = text_splitter.split_text([loader])
#     else: 
#         data = loader.load() #返回值是list,包含Document对象，Document对象包含page_content,metadata,metadata包含source=file_nam       
             
#         docs = text_splitter.split_documents(data)

#     res = []
#     for doc in tqdm(docs):

#         text = doc.page_content
#         text = clean_text(text)
#         source = doc.metadata['source']

#         res.append(process_record(text, source=source))

#     #写入文件
#     # with open(out_fp, "w",encoding='utf-8') as out_f:
#     #     for record in res:
#     #         out_f.write(json.dumps(record,  ensure_ascii=False) + "\n")
#     return res





    

    

if __name__ == "__main__":

    data_dir = args.data_dir
    out_dir = args.output_dir

    read_files(data_dir)
   

    