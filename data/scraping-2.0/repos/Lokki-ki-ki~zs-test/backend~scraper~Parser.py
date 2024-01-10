from langchain.schema.document import Document
from langchain.llms import OpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())

LLM_VERSION = "gpt-3.5-turbo-instruct"

class EccParser:
    def __init__(self):
        self.short_length = 50
        self.openaikey = os.environ.get('OPENAI_API_KEY')
        pass

    def extract_paragraphs_based_on_keywords(self, content, keywords):
        """
        Input: content: str
               keywords: list of str
        Output: list of str
        """
        text = content.split('\n')
        present, qna = self.seperate_presentations_qna(text)
        qna = self._filter_out_questions(qna)
        present_para = self._extract_paragraphs_based_on_keyword(present, keywords)
        qna_para = self._extract_paragraphs_based_on_keyword(qna, keywords)

        return present_para + qna_para
    
    def extract_paragraphs_based_on_sample(self, content):
        # Filter out Q&A's questions and short sentences which possibly be the person introductions
        text = content.split('\n')
        present, qna = self.seperate_presentations_qna(text)
        qna = self._filter_out_questions(qna)
        present_para = "\n".join(present + qna)
        # MapReduce Summarization
        llm = OpenAI(temperature=0, openai_api_key=self.openaikey, model=LLM_VERSION)
        text_splitter = RecursiveCharacterTextSplitter(separators=["\n"], chunk_size=1000, chunk_overlap=300)
        docs = text_splitter.create_documents([present_para])
        map_prompt = """
        Please provide a summary of the following earning conference call transcripts.
        ```{text}```
        SUMMARY:
        """
        combine_prompt = """
        Please provide a summary of the following earning conference call transcripts. Please provide the output in a manner that a credit rating agency could use to write a credit rating report.
        Please keep the results under 500 words.
        ```{text}```
        SUMMARY:
        """
        map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text"])
        combine_prompt_template = PromptTemplate(template=combine_prompt, input_variables=["text"])
        summary_chain = load_summarize_chain(
            llm=llm,
            chain_type='map_reduce',
            map_prompt=map_prompt_template,
            combine_prompt=combine_prompt_template,
            verbose=True
        )
        summary = summary_chain.run(docs)
        return summary

    def _get_documents_from_list(self, splitted_text):
        docs = [Document(page_content=x) for x in splitted_text]
        return docs

    def _extract_paragraphs_based_on_keyword(self, text, keywords):
        result = []
        for keyword in keywords:
            result += [x for x in text if keyword in x]
        return result

    
    def _filter_out_questions(self, text):
        for sentence in text:
            if '?' in sentence:
                text.remove(sentence)
        return text

    def seperate_presentations_qna(self, text):
        qna_index = len(text)
        for sentence in text:
            if 'Question-and-Answer' in sentence:
                qna_index = text.index(sentence)
                break
        presentation = text[:qna_index]
        qna = text[qna_index:]
        presentation = [x for x in presentation if len(x) > self.short_length]
        qna = [x for x in qna if len(x) > self.short_length]
        return presentation, qna



