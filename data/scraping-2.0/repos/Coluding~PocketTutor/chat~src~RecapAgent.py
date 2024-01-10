import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.summarize import load_summarize_chain
from langchain.chains import RetrievalQA
import json
from langchain.chat_models import ChatOpenAI

from dotenv import load_dotenv

load_dotenv("../../.env")


class RecapQuestionAgent:
    def __init__(self, pdf_path: str, grundwissen_name: str):
        self._pdf_path = pdf_path
        self.grundwissen_name = grundwissen_name
        self.document_ques_gen, self.document_answer_gen = self.file_processing()
        self.model = ChatOpenAI(temperature=0.15, model_name=os.getenv("OPENAI_MODEL_NAME"))
        embeddings = HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

        if os.path.exists("faiss_index" + "/" + grundwissen_name):
            self.vector_store = FAISS.load_local("faiss_index" + "/" + grundwissen_name, embeddings)
        else:
            self.vector_store = FAISS.from_documents(self.document_answer_gen, embeddings)
            self.vector_store.save_local("faiss_index" + "/" + grundwissen_name)


        prompt_template = """Verwende die folgenden Kontextteile, 
        um die Frage am Ende zu beantworten. Wenn du die Antwort nicht kennst, sage einfach, 
        dass du es nicht weißt, versuche nicht, eine Antwort zu erfinden. 
        Erklären auch kurz, warum die Antwort richtig oder falsch ist. 

        {context}

        Frage: {question}
        Antworte auf deutsch:"""
        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )
        chain_type_kwargs = {"prompt": PROMPT}
        self.answer_generation_chain = RetrievalQA.from_chain_type(llm=self.model,
                                                                   chain_type="stuff",
                                                                   retriever=self.vector_store.as_retriever(),
                                                                   chain_type_kwargs=chain_type_kwargs)

        self.qa_name = "question_answers" + "/" + grundwissen_name + ".json"
        if not os.path.exists(self.qa_name):
            ques = self.generate_question()
            ques_list = ques.split("\n")
            self.filtered_ques_list = [element for element in ques_list if
                                       element.endswith('?') or element.endswith('.')]

            self.self_answer_questions()

        else:
            with open(self.qa_name, "r") as f:
                self.qas = json.load(f)
            self.filtered_ques_list = [list(key.keys())[0] for key in self.qas]

    def file_processing(self):
        # Load data from PDF
        loader = PyPDFLoader(self._pdf_path)
        data = loader.load()

        question_gen = ''

        for page in data:
            question_gen += page.page_content

        splitter_ques_gen = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )

        chunks_ques_gen = splitter_ques_gen.split_text(question_gen)

        document_ques_gen = [Document(page_content=t) for t in chunks_ques_gen]

        splitter_ans_gen = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=30)

        document_answer_gen = splitter_ans_gen.split_documents(
            document_ques_gen
        )

        return document_ques_gen, document_answer_gen

    def generate_question(self):
        prompt_template = """
                    Du bist ein Abschlussprüfer für Schüler der 12. Klasse eines Gymnasiums. Du bist Experte im Erstellen
                    von Fragen über das Fach Mathematik. Dein Ziel ist es Schüler zu testen ob sie das Thema verstanden haben
                    Das machst du indem Fragen aus dem folgendem Text erstellst:

                    ------------
                    {text}
                    ------------

                    Erstelle Fragen für Schüler des Fachs Mathematik. Es sollen wahr oder falsch fragen sein. 
                    Bitte achte darauf, dass du sowohl wahre als auch falsche Fragen stellst. 
                    Schreibe die Antwort nicht in die Frage.  

                    QUESTIONS:
                    """

        PROMPT_QUESTIONS = PromptTemplate(template=prompt_template, input_variables=["text"])
        refine_template = ("""
                    Du bist ein Abschlussprüfer für Schüler der 12. Klasse eines Gymnasiums. Du bist Experte im Erstellen
                    von Fragen über das Fach Mathematik. Dein Ziel ist es Schüler zu testen ob sie das Thema verstanden haben
                    Wir haben bereits Übungsfragen erhalten: {existing_answer}. 
                    Wir haben jetzt die Option diese bestehenden Fragen zu verbessern oder neue hinzuzufügen.
                   (nur wenn benötigt) Hier ist mehr Kontext:
                   ------------
                   {text}
                   ------------

                    Gegeben dem neuen Kontext, erstelle die originalen Fragen auf Deutsch. Wenn der Kontext nicht hilfreich ist,
                    liefere die ursprünglichen Fragen"
                    Es sollen wahr oder falsch fragen sein. 
                    Bitte achte darauf, dass du sowohl wahre als auch falsche Fragen stellst. 
                    Schreibe die Antwort nicht in die Frage.  
                   QUESTIONS:
                   """
                           )

        REFINE_PROMPT_QUESTIONS = PromptTemplate(
            input_variables=["existing_answer", "text"],
            template=refine_template,
        )

        ques_gen_chain = load_summarize_chain(llm=self.model,
                                              chain_type="refine",
                                              verbose=True,
                                              question_prompt=PROMPT_QUESTIONS,
                                              refine_prompt=REFINE_PROMPT_QUESTIONS)

        ques = ques_gen_chain.run(self.document_ques_gen)
        return ques

    def self_answer_questions(self):
        self.qas = []
        for q in self.filtered_ques_list:
            print("answered")
            self.qas.append({
                q: self.answer_question(q)
            })

        with open(self.qa_name, "w") as f:
            json.dump(self.qas, f)

    def answer_question(self, question):
        answer = self.answer_generation_chain.run(question)
        return answer

    def review_question(self, question: str, answer: str) -> str:
        review_prompt = """Du bist ein Lehrer der 12. Klasse Gymnasium.  
                Du bist Experte im Bewerten von Schülern. Der Schüler hat folgende Frage bekommen:
                {question}
                Die Frage wurde aus diesem Kontext erstellt:
                {context}""" + f"""
                 Bitte bewerte die Antwort des Schülers mit richtig oder falsch und begründe deine Entscheidung.
                 Der Schüler soll mit "wahr, weil..." oder "falsch, weil..." antworten und die Antwort kurz begründen. 
                 Achte darauf, dass auch die Begründung korrekt ist und zur gegeben Antwort passt.
                 Wenn der Schüler keine oder eine falsche Begründung gibt, dann ist die Antwort falsch.
                Antwort des Schülers: {answer}
                Antworte auf deutsch.
                Gebe als erstes Wort die Beewertung aus und dann die Begründung."""

        review_prompt_template = PromptTemplate(
            template=review_prompt,
            input_variables=["context",
                             "question"]
        )
        chain_type_kwargs = {"prompt": review_prompt_template}
        review_question_chain = RetrievalQA.from_chain_type(llm=self.model,
                                                            chain_type="stuff",
                                                            retriever=self.vector_store.as_retriever(),
                                                            chain_type_kwargs=chain_type_kwargs)

        return review_question_chain.run(question)


qa = RecapQuestionAgent(r"grundwissen\Normalverteilung.pdf",
                        "normalverteilung")
print(qa.filtered_ques_list)
print(qa.review_question(qa.filtered_ques_list[5], "wahr, sie stehen für mittelwert und modalwert"))