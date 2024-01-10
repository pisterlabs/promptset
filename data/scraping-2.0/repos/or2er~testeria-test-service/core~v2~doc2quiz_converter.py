from .template import DOC2QUIZ_TEMPLATE
from .openai import llm, tiktoken_len, embeddings_model
from langchain.chains import LLMChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import PromptTemplate
from time import time
from uuid import uuid4
from utils.math import cosine_similarity


class Doc2QuizConverter:

    def __init__(self, document):
        self.id = str(uuid4())
        self.document = document
        self.prompt = PromptTemplate(
            input_variables=["document"], template=DOC2QUIZ_TEMPLATE
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=700,
            chunk_overlap=300,
            length_function=tiktoken_len
        )
        self.chain = LLMChain(llm=llm, prompt=self.prompt)
        self.questions = []
        self.progress = 0

    def convert(self):
        start = time()

        chunks = self.text_splitter.create_documents([self.document])

        print("Start tool 2 ...")
        print("Total chunks:", len(chunks))
        print("Converting document to questions...")

        for i, chunk in enumerate(chunks):
            response = self.chain.run(chunk.page_content)

            self.progress = (i + 1) / len(chunks)

            if response == '<empty>':
                continue

            questions_ = self._extract_questions(response)

            embeddings = embeddings_model.embed_documents(
                list(map(lambda q: q["content"], questions_)))

            for j, question in enumerate(questions_):
                question["embeddings"] = embeddings[j]

            self.questions += questions_

            for question in questions_:
                print(question["content"])

            print(f"Chunk {i+1}/{len(chunks)}")

            self.progress = (i + 1) / len(chunks)

        self._remove_duplicates()

        print("Generated", len(self.questions),
              "questions in", time() - start, "s.")

    def _remove_duplicates(self):
        remove_questions = []

        for i in range(len(self.questions)):
            for j in range(i + 1, len(self.questions)):
                q1 = self.questions[i]
                q2 = self.questions[j]

                if cosine_similarity(q1["embeddings"], q2["embeddings"]) > 0.95:
                    remove_questions.append(self.questions[j])
                    # print("Remove duplicate questions:")
                    # print(q1["content"])
                    # print(q2["content"])

        new_questions = []
        for i, question in enumerate(self.questions):
            if i not in remove_questions:
                new_questions.append(question)

        self.questions = new_questions

    def _extract_questions(self, response):
        q_texts = response.split("<question>")
        questions = []

        for q_text in q_texts:
            q_text = q_text.strip()

            if q_text == "":
                continue

            lines = q_text.split("\n")

            if len(lines) < 8:
                continue

            answer = lines[6][8]

            if answer not in ['A', 'B', 'C', 'D']:
                answer = -1
            else:
                answer = ord(answer) - ord('A')

            difficulty = lines[7][12:]

            questions.append({
                "index": int(lines[0]),
                "content": lines[1],
                "choices": [
                    lines[2][3:],
                    lines[3][3:],
                    lines[4][3:],
                    lines[5][3:]
                ],
                "answer": answer,
                "difficulty": difficulty
            })

        return questions
