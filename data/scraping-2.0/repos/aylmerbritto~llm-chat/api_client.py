from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from langchain.vectorstores import FAISS
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
    PromptTemplate,
)
import torch


class MusicFestivalAssistant:
    def __init__(self, db=None):
        system_message_prompt = SystemMessagePromptTemplate(
            prompt=PromptTemplate(
                input_variables=[],
                template="You are a Music Festival Assistant,\
				known for your kindness and excellent greeting skills.\
				The festival spans three days, and you've been named Avicii, \
				in honor of the legendary musician. You will utilize the festival\
				schedule to effectively answer any questions from attendees.",
            )
        )
        human_message_prompt = HumanMessagePromptTemplate(
            prompt=PromptTemplate(
                input_variables=["context", "question"],
                template="Thank you for your question! {context}\n\n\
				Question: {question}\n\nI'm here to assist you. Let's \
				see how we can make your music festival experience even better!\
				if the question is just greeting greet them back\
				Format the answer with proper punctuations too",
            )
        )

        # Combining the prompts into a ChatPromptTemplate
        combined_prompt_template = ChatPromptTemplate.from_messages(
            [system_message_prompt, human_message_prompt]
        )

        # Using the combined prompt template in the function
        combine_docs_chain_kwargs = {"prompt": combined_prompt_template}

        # Now you can use combine_docs_chain_kwargs in your function
        if db:
            self.embeddings = HuggingFaceEmbeddings(multi_process=False)
            self.db = FAISS.load_local("faiss_index", self.embeddings)
        else:
            self.db = db

        # Set up a conversation memory to keep track of the chat history
        memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )
        self.qa = ConversationalRetrievalChain.from_llm(
            OpenAI(
                temperature=0.8,
                openai_api_key="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",
            ),
            self.db.as_retriever(),
            memory=memory,
            combine_docs_chain_kwargs=combine_docs_chain_kwargs,
        )

    def answer_query(self, query):
        result = self.qa.run(query)
        return result


if __name__ == "__main__":
    assistant = MusicFestivalAssistant()
    while True:
        print(assistant.answer_query(input("Enter your Question:")))
