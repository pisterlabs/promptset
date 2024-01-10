import os
from docx import Document
from openai import OpenAI
import chromadb
import logging
from chromadb.utils import embedding_functions


# Job recommendation engine and UI to display them
class Recommendation:
    def __init__(
        self, resume, openai_key=None, jobtitle=None
    ):  # model = 'all-MiniLM-L6-v2' by default
        self.resume = resume
        self.jobtitle = jobtitle
        self.openai_key = openai_key
        self.client = OpenAI(api_key=self.openai_key)
        self.file_path = os.path.join(os.getcwd(), "talenttrove/data/jd_vectordb")
        self.chroma_client = chromadb.PersistentClient(path=self.file_path)
        self.collection = self.chroma_client.get_or_create_collection(
            name="mycareersfuture_jd",
            embedding_function=embedding_functions.DefaultEmbeddingFunction(),
        )

    def read_word_document(self):
        doc = Document(self.resume)  ### RESUME HAS TO BE A PATH OR I/O object
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text

    def get_generated_jd(self):
        text = self.read_word_document()
        logging.info("Document Read")
        try:
            # Create a chat completion using the question and context
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "Based on the given resume above, create a suitable job posting for this resume. The job posting must include the job description, job responsibilities, and requirements such as qualifications and skills. Do not include the company name and location in this job posting.",
                    },
                    {
                        "role": "user",
                        "content": f"Resume: {text}\n\n---\n\nJob Description:",
                    },
                ],
                temperature=0.3,
                # max_tokens=max_tokens,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                # stop=stop_sequence,
            )
            logging.info("CV Generated")
            return response.choices[0].message.content.strip()
        except Exception as e:
            logging.warning(e)
            print(e)
            return ""

    def search_jd(self, jd, k=10):
        results = self.collection.query(
            query_texts=[jd],
            n_results=k,
            include=["documents", "distances", "metadatas"],
        )
        return results
