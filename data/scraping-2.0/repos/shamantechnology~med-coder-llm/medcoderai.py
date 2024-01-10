"""
MedCoderAI class
Interface to RAG based AI
"""
import os
import subprocess
from concurrent.futures import as_completed
import weaviate
from weaviate.embedded import EmbeddedOptions

from trulens_eval import Feedback, LiteLLM, Tru, TruChain, Huggingface

from langchain.chat_models import ChatVertexAI
from langchain.vectorstores import Weaviate
from langchain.document_loaders import CSVLoader
# from langchain.embeddings import VertexAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import (
    ConversationBufferMemory,
    ConversationSummaryMemory
)
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)

class MedCoderAI:
    def __init__(self):
        self.cpt_csv = "./data/2024_DHS_Code_List_Addendum_11_29_2023.csv"
        self.icd_csv = "./data/Section111ValidICD10-Jan2024.csv"
        self.code_docs = []
        self.client = None
        self.vectorstore = None
        self.llm = ChatVertexAI(
            temperature=0.1
        )
        self.llm_chain = None
        self.conversation = None
        self.memory = None
        self.chain_recorder = None

        # generate docs
        self.generate_cpt_icd_docs()

    def generate_cpt_icd_docs(self):
        """
        Generate langchain docs from CSVs
        """
        print("Generating CPT/ICD to langchain docs...")
        try:
            cpt_loader = CSVLoader(self.cpt_csv)
            self.code_docs += cpt_loader.load()
            # add cpt meta information
            # for code_doc in self.code_docs:
            #     code_doc.metadata["code_type"] = "cpt"
            #     code_doc.metadata["active"] = True

            icd_loader = CSVLoader(self.icd_csv)
            self.code_docs += icd_loader.load()
            # add icd meta information
            # for code_doc in self.code_docs:
            #     code_doc.metadata["code_type"] = "icd"
            #     code_doc.metadata["active"] = True
        except Exception as err:
            print(f"Error when loading CPT/ICD data: {err}")

    def refresh_token(self) -> str:
        result = subprocess.run(["gcloud", "auth", "print-access-token"], capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error refreshing token: {result.stderr}")
            return None
        return result.stdout.strip()

    def re_instantiate_weaviate(self) -> weaviate.Client:
        try:
            token = self.refresh_token()

            if token:
                self.client = weaviate.Client(
                    additional_headers={
                        "X-Palm-Api-Key": token
                    },
                    embedded_options=EmbeddedOptions(
                        additional_env_vars={
                            "ENABLE_MODULES": "text2vec-palm"
                        }
                    )
                )
            else:
                raise ValueError
        except Exception:
            raise

    def init_conversation(self):
        print("Initilizing conversations and vectorstores")
        try:
            # start weaviate with schemas
            self.re_instantiate_weaviate()

            # self.client = weaviate.Client(
            #     url=os.getenv("WEAVIATE_CLUSTER_URL"),
            #     auth_client_secret=weaviate.AuthApiKey(
            #         api_key=os.getenv("WEAVIATE_API_KEY"))
            # )
        except Exception as err:
            print(f"failed to start weviate client: {err}")
            raise

        print(f"Adding {len(self.code_docs)} documents to vectorstore")
        try:
            # setup vectorstore and retriever
            self.vectorstore = Weaviate.from_documents(
                client=self.client, 
                documents=self.code_docs, 
                # embedding=VertexAIEmbeddings(), 
                embedding=HuggingFaceEmbeddings(),
                by_text=False
            )
        except Exception as err:
            print(f"failed to add docs to weaviate: {err}")
            raise

        template = """
        You are Betsy who is a professional medical coder. 
        Answer the users question and return the proper ICD and/or CPT codes or a list of possible ICD and/or CPT codes that one could use for the question. 
        If you cannot find the answer from the pieces of context, ask the user for more details.

        Question: {question}
        -------------------------------
        Context: {context}
        -------------------------------
        Chat History: {chat_history}
        -------------------------------"""

        qa_prompt = PromptTemplate(
            input_variables=["context", "chat_history", "question"], 
            template=template
        )

        try:
            print("Creating conversational buffer memory")
            # setup memory and convo
            self.memory = ConversationBufferMemory(
                memory_key="chat_history",
                output_key="answer",
                return_messages=True
            )
        except Exception as err:
            print(f"Creating conversational buffer memory failed: {err}")
            raise
        
        try:
            print("Creating conversational RAG")
            self.conversation = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=self.vectorstore.as_retriever(search_kwargs={"k": 10}), 
                memory=self.memory,
                verbose=True,
                combine_docs_chain_kwargs={"prompt": qa_prompt}
            )
        except Exception as err:
            print(f"Creating conversational RAG failed: {err}")
            raise

    def run(self):
        # setup conversation
        self.init_conversation()

        # setup trulens
        tru = Tru()
        feedbacks = []
        
        litellm_provider = LiteLLM(model_engine="chat-bison")
        feedbacks.append(Feedback(litellm_provider.conciseness).on_output())

        hugs = Huggingface()
        # personal identifying information check
        # This is good for names but need to find model for detecting PHI
        feedbacks.append(Feedback(hugs.pii_detection).on_input())
        
        self.chain_recorder = TruChain(
            self.conversation,
            app_id="med-coder-llm",
            initial_app_loader=self.init_conversation,
            feedbacks=feedbacks
        )

        # tru.run_dashboard()

    def ask_question(self, user_msg) -> str:
        rec = None
        with self.chain_recorder as recorder:
            resp = self.conversation({"question": user_msg})
            rec = recorder.get()

        pii_detected = False
        conciseness = 0.0
        if rec:
            for feedback_future in  as_completed(rec.feedback_results):
                feedback, feedback_result = feedback_future.result()
                
                print(f"feedback name: {feedback.name}\n result: {feedback_result.result}")

                if feedback.name == "pii_detection" and feedback_result.result != None:
                    pii_detected = True
                
                if feedback.name == "conciseness":
                    conciseness = float(feedback_result.result) if feedback_result.result else 1.0
        
        if pii_detected:
            return "I'm sorry but personal information was detected in your question. Please remove any personal information."
        elif conciseness < 0.5:
            return "Please restate your question in a way the AI can understand and give a better answer"
        else:
            return resp["answer"]