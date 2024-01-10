import reflex as rx
import os
import openai
from dotenv import load_dotenv
from tutrflxapp.src.models import dumb_answer, qa_machine
from tutrflxapp.src.preprocessing import PdfVectorizer
import glob
from langchain.vectorstores import FAISS
import pickle
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


class State(rx.State):

    question: str
    chat_history: list[tuple[str, str]]
    files: list[str]
    processing = False
    complete = False

    async def handle_upload(self, files: list[rx.UploadFile]):
        global vector_store
        """Handle the upload of file(s).

        Args:
            files: The uploaded files.
        """

        for file in files:
            upload_data = await file.read()
            outfile = f".web/public/{file.filename}"

            # Save the file.
            with open(outfile, "wb") as file_object:
                file_object.write(upload_data)
            self.processing, self.complete = True, False
            # Update the files var.
            self.files.append(file.filename)
            pdf_docs = [f".web/public/{file}" for file in self.files]
            pdf_vectorizer = PdfVectorizer(pdf_docs)
            vector_store = pdf_vectorizer.get_vector_store()
            with open('vector_store.pkl', 'wb') as file:
                pickle.dump(vector_store, file)
            self.processing, self.complete = False, True

    def answer(self):

        # with open('vector_store.pkl', 'rb') as file:
        #     vector_store = pickle.load(file)
        # Our chatbot is not very smart right now...
        answer = qa_machine(self.question, vector_store)
        self.chat_history.append((self.question, answer))
        self.question = ""
        yield  # This clears the input field after the question is asked.
