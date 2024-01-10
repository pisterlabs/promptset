import os

import pickledb
import pypdf
from langchain import FAISS, OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings import OpenAIEmbeddings


class PaperDistiller:

    def __init__(self, paper_name):
        self.name = paper_name
        self.answers = {}
        self.cached_answers = pickledb.load('distiller.db', auto_dump=False, sig=False)

    def split_pdf(self, chunk_chars=4000, overlap=50):
        pdf_file_obj = open("Papers/%s.pdf" % self.name, "rb")
        pdf_reader = pypdf.PdfReader(pdf_file_obj)
        splits = []
        split = ""
        for i, page in enumerate(pdf_reader.pages):
            split += page.extract_text()
            if len(split) > chunk_chars:
                splits.append(split[:chunk_chars])
                split = split[chunk_chars - overlap:]
        pdf_file_obj.close()
        print(f"Split into {len(splits)} chunks")
        return splits

    def read_or_create_index(self):

        if os.path.isdir('Index/%s' % self.name):
            print("Index Found!")
            self.ix = FAISS.load_local('Index/%s' % self.name, OpenAIEmbeddings())
        else:
            print("Creating index!")
            self.ix = FAISS.from_texts(self.split_pdf(), OpenAIEmbeddings())
            # Save index to local (save cost)
            self.ix.save_local('Index/%s' % self.name)

    def query_and_distill(self, query):

        # Answer already in memory
        if query in self.answers:
            print("Answer found!")
            return self.answers[query]
        # Answer cached (asked in the past) in pickledb
        elif self.cached_answers.get(query + "-%s" % self.name):
            print("Answered in the past!")
            return self.cached_answers.get(query + "-%s" % self.name)
        # Generate the answer
        else:
            print("Generating answer!")
            query_results = self.ix.similarity_search(query, k=2)
            chain = load_qa_chain(OpenAI(temperature=0.25), chain_type="stuff")
            self.answers[query] = chain.run(input_documents=query_results, question=query)
            self.cached_answers.set(query + "-%s" % self.name, self.answers[query])
            return self.answers[query]

    def cache_answers(self):
        self.cached_answers.dump()
