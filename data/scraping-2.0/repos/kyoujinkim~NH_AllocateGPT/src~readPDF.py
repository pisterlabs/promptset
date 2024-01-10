import html
import random
import re
from typing import List

import unicodedata
from langchain.docstore.document import Document
import datetime as dt

import fitz
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm

class PDFReader:
    def __init__(self):
        self.nonNameList = "[<>?:|/\\*]"
        self.EMAIL_PATTERN = re.compile(r'''(([a-zA-Z0-9._%+-]+)@([a-zA-Z0-9.-]+)(\.[a-zA-Z]{2,4}))''')
        self.URL_PATTERN = re.compile("(ftp|http|https)?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+")
        self.URL_PATTERN2 = re.compile("www.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+")
        self.ACCOUNT_PATTERN = re.compile('@[a-zA-Z]*')
        self.PHONE_PATTERN = re.compile('(\))?\d{2,3}( )?(\)| |-|\.)?( )?\d{3,4}( )?( |-|\.)?( )?\d{4}')
        self.removal_list = "_#@|◇·△▲▷▶➤●■□○>>-∼=ㆍ<>【】…◆"
        self.MULTIPLE_SPACES = re.compile(' +', re.UNICODE)

    def cleanse_text(self, text):
        #text = text.lower()
        text = text.replace("\n\n", '. ').replace("\n", '').replace("\'s", '')
        text = re.sub(self.EMAIL_PATTERN, ' ', text)   # 이메일 패턴 제거
        text = re.sub(self.URL_PATTERN, ' ', text)   # URL 패턴 제거
        text = re.sub(self.URL_PATTERN2, ' ', text)   # URL 패턴 제거
        text = re.sub(self.ACCOUNT_PATTERN, ' ', text)   # 트위터 계정명 패턴 제거 (e.g., @iAmPanoramic)
        text = re.sub(self.PHONE_PATTERN, ' ', text)
        text = re.sub(re.compile('<.*?>'), '', text)   # HTML 태그 제거 (e.g., <br>)
        text = html.unescape(text)   # HTML 코드 변환 (&#39; -> ')
        text = text.translate(str.maketrans(self.removal_list, '.'*len(self.removal_list)))   # 특수문자 제거
        text = re.sub(re.compile('\.\.\.'), '', text)
        text = re.sub(self.MULTIPLE_SPACES, ' ', text)   # 무의미한 공백 제거

        return text

    def getPDF(self, filelist:List, sample:float=1.0):
        '''
        지정된 경로의 PDF 문서들의 텍스트 값을 가져옵니다.
        :param datapath: PDF 문서 경로
        :param sample: PDF 문서 전체 중 일부 비율만을 추출
        :return:
        '''
        pdflist = filelist #glob(f'{datapath}/*')
        pdflist = random.sample(pdflist, int(len(pdflist)*sample))

        Docset = []
        for each_pdf in tqdm(pdflist, desc='PDF 추출'):
            content_header = []
            try:
                doc = fitz.open(each_pdf)
            except:
                print(f'{each_pdf} file is broken')
                continue

            for each_page in doc:
                pdf_info = doc.name.split('/')[-1].replace('.pdf', '')
                content = each_page.get_text()
                content_line = content.split('\n')

                content_filtered = content_line
                #content_filtered = [x for x in content_line if (sum(c.isalpha() for c in x) > sum(c.isdigit() for c in x) and len(x) > 10)]

                if len(content_filtered) > 0:
                    content_drop = [x for x in content_filtered if x not in content_header]
                    content_header.extend(content_filtered[:5])
                    content_header = list(set(content_filtered))
                else:
                    content_drop = content_filtered
                content_concat = ' '.join(content_drop)
                content_cleanse = self.cleanse_text(content_concat)

                if sum([x in content_cleanse for x in ['compliance', 'Compliance', '무단 복제 및 배포']])>0:
                    continue

                if len(content_cleanse) > 600 and (sum(c.isalpha() for c in content_cleanse) > sum(c.isdigit() for c in content_cleanse)):
                    try:
                        pdf_info_date = int(dt.datetime.strptime(pdf_info.split('(')[1].split(')')[0], '%Y, %B %d').strftime('%y%m%d'))
                    except:
                        pdf_info_date = int(dt.datetime.now().strftime('%y%m%d'))
                    new_doc = Document(
                            page_content=content_cleanse,
                            metadata={"source": pdf_info,
                                      "page": each_page.number,
                                      "date": pdf_info_date,
                                      }
                    )
                    Docset.append(new_doc)

        return Docset

    def split_text_byChunk(self, doc, chunk_size:int=500, overlap:int=100):
        text = doc.page_content
        text_list = RecursiveCharacterTextSplitter(chunk_overlap=overlap, chunk_size=chunk_size).split_text(text)
        doc_source_unicode = unicodedata.normalize('NFC', doc.metadata['source'])
        if "[기업]" in doc_source_unicode:
            compname = doc_source_unicode.split("[기업]")[1].split(",")[0]
            text_list = [compname + ", " + x for x in text_list]
        doc_list = []
        for text in text_list:
            new_doc = Document(
                page_content=text,
                metadata={"source": doc.metadata['source'], "page": doc.metadata['page'], "date": doc.metadata['date']},
            )
            doc_list.append(new_doc)

        return doc_list