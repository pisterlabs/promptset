from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline, HuggingFacePipeline
from langchain.chains import VectorDBQA


class VectorDBQAWrapper:
    def __init__(self, model_name, text_loader_file):
        self.loader = TextLoader(text_loader_file)
        self.qna = self.loader.load()

        text_splitter_qna = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        self.texts_qna = text_splitter_qna.split_documents(self.qna)

        self.embedding_function = SentenceTransformerEmbeddings(model_name=model_name)
        self.vectordb = Chroma.from_documents(self.texts_qna, self.embedding_function)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.pipe = pipeline(
            "text2text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_length=100
        )
        self.local_llm = HuggingFacePipeline(pipeline=self.pipe)

    def run_qa(self, question):
        template = """Question: {question}
        Answer: Let's think step by step.
        Answer: """

        qa = VectorDBQA.from_chain_type(llm=self.local_llm, chain_type="stuff", vectorstore=self.vectordb)
        qa_output = qa.run(question)

        return qa_output


# Usage example:
model_name = "google/flan-t5-large"
text_loader_file = 'HDFC_Faq.txt'

vector_db_qa_wrapper = VectorDBQAWrapper(model_name=model_name, text_loader_file=text_loader_file)

question = "How to change the password?"
qa_output = vector_db_qa_wrapper.run_qa(question)

# LLM with the vector DB
print("Vector DB Output: ", qa_output)
