from typing import List

import openai
import tiktoken
from langchain import PromptTemplate
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chat_models import ChatOpenAI
from langchain.docstore.document import Document
from tqdm import tqdm

import src.template.promptTemplate as pTemp
import src.template.queryTemplate as qTemp
from src.quoteChecker import changeQuoteNum, printQuote


class printAssetView:
    '''
    프롬프트와 데이터베이스를 받아서 LLM 입력 토큰으로 제공
    출력값으로 자산별 리스크/상승요인 요약텍스트를 출력해서 가져오는 로직
    '''
    def __init__(self,
                 llmAiEngine: str = 'gpt-3.5-turbo',
                 numberOfReason: int = 10,
                 ):

        openai.Engine = llmAiEngine
        self.tokenizer = tiktoken.encoding_for_model(llmAiEngine)
        self.numberOfReason = numberOfReason

        self.templates = pTemp.loadTemplate()

        self.PROMPT = PromptTemplate(template=self.templates['template'], input_variables=["summaries", "question"])
        self.PROMPT_S = PromptTemplate(template=self.templates['template_s'], input_variables=["summaries", "question"])
        self.PROMPT_AGG = PromptTemplate(template=self.templates['template_agg'], input_variables=["summaries", "question"])

        self.chain = load_qa_with_sources_chain(ChatOpenAI(model_name='gpt-3.5-turbo',
                                                      temperature=0.0,
                                                      frequency_penalty=0.0,
                                                      ), chain_type="stuff", prompt=self.PROMPT)
        self.chain_s = load_qa_with_sources_chain(ChatOpenAI(model_name='gpt-3.5-turbo',
                                                        temperature=0.5,
                                                        frequency_penalty=1.0,
                                                        ), chain_type="stuff", prompt=self.PROMPT_S)
        self.chain_agg = load_qa_with_sources_chain(ChatOpenAI(model_name='gpt-3.5-turbo',
                                                        temperature=1.0,
                                                        frequency_penalty=0.0,
                                                        ), chain_type="stuff", prompt=self.PROMPT_AGG)

    def define_universe(self, assetTable: List[str]):
        self.assetTable = assetTable

    def set_query(self, query:str):
        #set query for printAssetView
        self.query = query

    def set_baseDocument(self, baseDocument):
        #set baseDocument for printAssetView
        self.baseDocument = baseDocument

    def get_similarDocs(self):
        #get similar docs for printAssetView
        return self.baseDocument.similarity_search(self.query, k=self.numberOfReason)

    def printEvidence(self, docs) -> List[Document]:

        '''근거 목록 저장'''
        context_doc = []
        token_count = 0

        summ_docs_part = []
        doc_offset = 0
        doc_count = 0
        token_sub_count = 0
        for doc in docs:
            summ_docs_part.append(doc)
            doc_count += 1
            token_sub_count += len(self.tokenizer.encode(doc.page_content))

            '''요약하기 원하는 문서 토큰의 개수가 2048개 이상인 경우에만, 요약을 수행. 그 외에는 문서 모으는 작업만 수행'''
            if token_sub_count < 2048:
                pass
            else:
                output = self.chain({"input_documents": summ_docs_part, "question": self.query}, return_only_outputs=True)
                output['output_text'] = changeQuoteNum(output['output_text'], doc_offset)

                summ_docs_part = []
                doc_offset += doc_count
                doc_count = 0
                token_sub_count = 0

                '''요약을 완료한 문서의 토큰 개수가 2048개 이상인 경우에는 작업을 완전 중지'''
                context_doc.append(Document(page_content=output['output_text'], metadata={"source": ''}))
                token_count += len(self.tokenizer.encode(output['output_text']))
                if token_count >= 2048:
                    break

        '''작업을 완료한 이후, token count 조건이 충족되지 않았다면, 마지막으로 남은 문서를 요약하기 위해 추가 작업 수행'''
        if token_count == 0:
            output = self.chain({"input_documents": summ_docs_part, "question": self.query}, return_only_outputs=True)
            output['output_text'] = changeQuoteNum(output['output_text'], doc_offset)

            context_doc.append(Document(page_content=output['output_text'], metadata={"source": ''}))

        return context_doc

    def filterEvidence(self,context_doc,) -> Document:
        '''
        쿼리에 대한 근거를 필터링 및 재정렬하는 텍스트 컨센서스 생성 프로세스
        :param query:
        :param context_doc:
        :return:
        '''
        '''필터링 근거 출력'''
        output = self.chain_s({"input_documents": context_doc, "question": self.query}, return_only_outputs=True)
        rearr_context_doc = Document(page_content=output['output_text'], metadata={"source": ''})

        return rearr_context_doc

    def printConclusion(self, rearr_context_doc,)-> str:
        '''
        쿼리에 대한 응답을 생성하는 텍스트 컨센서스 생성 프로세스
        :param query:
        :param rearr_context_doc:
        :param docs:
        :return:
        '''

        '''요약 기준 결론 출력'''
        output = self.chain_agg({"input_documents": [rearr_context_doc], "question": self.query}, return_only_outputs=True)

        return output['output_text']

    def printQuote(self, rearr_context_doc, docs,) -> str:
        '''주석 출력'''
        quoteOutput = printQuote(rearr_context_doc, docs, verbose=False)

        return ''.join(quoteOutput)

    def __assetPrompt(self, asset):
        '''
        :param asset: asset name
        :return: prompt message
        '''
        prompt = qTemp.loadTemplate(asset)

        return prompt

    def printAssetView(self):
        '''
        쿼리에 대한 자산관점 요약을 생성하는 텍스트 컨센서스 생성 프로세스
        :param query:
        :param docs:
        :return:
        '''

        result = {}
        archive = {}
        pbar = tqdm(self.assetTable)
        for asset in pbar:
            '''쿼리 세팅'''
            query = self.__assetPrompt(asset)
            self.set_query(query)

            '''유사 문서 목록 찾기'''
            pbar.set_postfix_str(f"{asset} : Find Similar Docs")
            docs = self.get_similarDocs()

            '''근거 목록 저장'''
            pbar.set_postfix_str(f"{asset} : Print Evidence")
            context_doc = self.printEvidence(docs)

            '''근거 필터링'''
            pbar.set_postfix_str(f"{asset} : Filtering Evidence")
            rearr_context_doc = self.filterEvidence(context_doc)

            '''결론 출력'''
            pbar.set_postfix_str(f"{asset} : Print Conclusion")
            conclusion = self.printConclusion(rearr_context_doc)

            '''주석 출력'''
            quoteOutput = self.printQuote(rearr_context_doc, docs)

            '''전체문구'''
            totalOutput = ('\n').join([rearr_context_doc.page_content, conclusion, quoteOutput])

            result[asset] = conclusion
            archive[asset] = totalOutput

        return result, archive