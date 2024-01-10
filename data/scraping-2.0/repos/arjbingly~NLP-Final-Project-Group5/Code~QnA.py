from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain import HuggingFacePipeline
from transformers import pipeline
import torch


class QuestionAnswering:
    def __init__(self, model_name="bert-large-cased-whole-word-masking-finetuned-squad"):
        # Load the pre-trained model and tokenizer
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        self.text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            AutoTokenizer.from_pretrained('microsoft/mpnet-base'),
            chunk_size=512,
            chunk_overlap=200)
        self.embedding_model = 'all-mpnet-base-v2'
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': False}
        self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model,
                                                model_kwargs=model_kwargs,
                                                encode_kwargs=encode_kwargs,
                                                )

        self.model_pipe = pipeline("question-answering",
                                   model=self.model_name,
                                   tokenizer=self.tokenizer,
                                   return_tensors='pt')

        self.temp = 0.7
        self.llm = HuggingFacePipeline(pipeline=self.model_pipe,
                                       model_kwargs={'temperature': self.temp,
                                                     'max_length': 512})
        self.k = 5  # no of docs to retrive

    def create_vector_db(self, text):
        pages = self.text_splitter.split_text(text)
        self.docs = self.text_splitter.create_documents(pages)
        self.db = Chroma.from_documents(self.docs, self.embeddings)
        self.retriever = self.db.as_retriever(search_kwargs={'k': self.k})

    def infer(self, question):
        context_result = self.retriever.get_relevant_documents(question)
        # question_input = []
        # context_input = []
        # for context in context_result:
        #     question_input.append(question)
        #     context_input.append(context.page_content)

        question_input = [question for i in range(len(context_result))]
        context_input = [context.page_content for context in context_result]

        results = self.model_pipe(question=question_input, context=context_input)
        return results

# %%
# #E.G.
# from news_fetch import NewsArticle
#
# url = 'https://www.bbc.com/news/world-europe-63863088'  # long news article
# news = NewsArticle(url)
# text = news.article.text
# qa = QuestionAnswering()
# qa.create_vector_db(text)
# question = 'Who were in pain?'
# results = qa.infer(question)

