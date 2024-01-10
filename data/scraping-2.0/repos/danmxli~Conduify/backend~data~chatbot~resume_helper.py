import os
import openai
from openai import OpenAI
from dotenv import load_dotenv
import spacy
from llama_index.readers.schema.base import Document
from llama_index import VectorStoreIndex
from data.config.embed import embed_pdf
load_dotenv('.env')

SIMILARITY_THRESHOLD = 0.8

RESUME_HEADER_PROMPT = """
        """



CLEANUP_MESSAGE_PROMPT = """
        You are an assistant whose job is to cleanup and summarize a chunk of a resume document.
        You have access to all the important details of the chunk. You must include ALL the details in your summary.
        Your response must include ONLY the summary of the chunk. You must respond in third person PASSIVE point of view.
        Do not begin your summary with "this individual...". Instead, construct the summary as if you are rewriting the chunk for a resume.
        """

ANALYSIS_MESSAGE_PROMPT = """
        You are an assistant whose job is to provide a short and concise analysis of a chunk of a resume document.
        You have access to all the important details of the chunk, as well as all the details of the job description the author of the resume is currently applying to.
        Your task is to provide concise suggestions for enhancing the content to align better with the requirements of the target job.
        Construct your analysis in point form. You must respond in third person PASSIVE point of view.
        """


class ResumeHelper:

    def __init__(self) -> None:
        self.nlp = spacy.load("en_core_web_md")
        self.helper = OpenAI(api_key=os.getenv('OPENAI_KEY'))

        # vector store index for rag
        openai.api_key = os.getenv('OPENAI_KEY')
        self.index = VectorStoreIndex([])
        

    def rag(self, resume_url):
        """
        Retrieve contexual information from resume document url.
        """
        doc = embed_pdf(resume_url)["doc"]
        
        # LlamaIndex query engine from LayoutPDFReader document chunks
        for chunk in doc.chunks():
            self.index.insert(Document(text=chunk.to_context_text(), extra_info={}))
        query_engine = self.index.as_query_engine()

        print(query_engine.query("ONLY list the name, contact information, and education of the author of the resume in point form. Exclude filler words."))


    def group(self, contexts, interview_info):
        """
        Group resume chunks together by string context similarity.
        """
        groups = []

        docs = list(self.nlp.pipe(contexts))
        texts = [{'text': doc.text, 'matched': False} for doc in docs]

        for i, e in enumerate(docs):
            if texts[i]['matched']:
                continue

            group = [e.text]
            texts[i]['matched'] = True

            for j in range(i+1, len(docs)):

                if texts[j]['matched'] or docs[i].similarity(docs[j]) <= SIMILARITY_THRESHOLD or docs[j].text in group:
                    continue

                group.append(docs[j].text)
                texts[j]['matched'] = True

            grouped_chunk = "\n".join(group)
            cleaned_chunk = self.cleanup_chunk(grouped_chunk)
            groups.append(self.analyze_chunk(cleaned_chunk, interview_info))

        return groups

    def cleanup_chunk(self, grouped_chunk):
        """
        Clean up formatting errors by summarizing the chunk.
        """
        helper_config = [
            {
                "role": "system",
                "content": f"{CLEANUP_MESSAGE_PROMPT}"
            },
            {
                "role": "user",
                "content": f"Cleanup and summarize the following chunk:\n\n{grouped_chunk}"
            }
        ]
        response = self.helper.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=helper_config
        )
        return (response.choices[0].message.content)

    def analyze_chunk(self, chunk, interview_info):
        """
        Adds similarity ranking, chunk summary, and suggestions to the chunk. Returns new string object
        """

        helper_config = [
            {
                "role": "system",
                "content": f"{ANALYSIS_MESSAGE_PROMPT}"
            },
            {
                "role": "user",
                "content": f"Analyze the following chunk:\n\n{chunk}\n\nProvided is the job description:\n\n{interview_info}"
            }
        ]
        response = self.helper.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=helper_config
        )
        res_chunk = f"{chunk}///EVALUATION///{response.choices[0].message.content}"
        return res_chunk
