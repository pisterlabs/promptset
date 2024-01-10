# from Conversation.utilz import load_config, load_db, load_embeddings, condense_question_prompt, get_chat_history, question_answer_prompt
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from dotenv import load_dotenv
import sys
sys.path.append("Conversation")
import utilz

# from Converutils import load_config, load_db, load_embeddings, condense_question_prompt, get_chat_history, question_answer_prompt

# load environment to get the api keys
load_dotenv()

class chat_retrieval():
    def __init__(self) -> None:
        # get config
        config = utilz.load_config()

        # load embeddings
        embedding_function = utilz.load_embeddings()
        
        # if documents are not loaded then load documents + split
        db = utilz.load_db(embedding_function)

        # define Conversation Buffer Memory
        memory = ConversationBufferWindowMemory(
            k=2, 
            memory_key='chat_history', 
            return_messages=True, 
            output_key='answer'
        )

        # initialize the retrieval chain
        # This chain has two steps. First, it condenses the current question and the chat history into a 
        # standalone question. This is necessary to create a standanlone vector to use for retrieval. 
        # After that, it does retrieval and then answers the question using retrieval augmented generation 
        # with a separate model. Part of the power of the declarative nature of LangChain is that you can 
        # easily use a separate language model for each call. This can be useful to use a cheaper and 
        # faster model for the simpler task of condensing the question, and then a more expensive model 
        # for answering the question.
        self.qa = ConversationalRetrievalChain.from_llm(
            llm = ChatOpenAI(temperature=config["OpenAI"]["temperature"]), 
            retriever = db.as_retriever(
                # Defines the type of search that the Retriever should perform. Can be “similarity” (default), 
                # “mmr”, or “similarity_score_threshold”
                type = config["chroma_indexstore"]["search_type"],
                kwargs = {
                    # Amount of documents to return
                    'k': config["chains"]["nbr_of_source_documents"],
                    # Minimum relevance threshold for similarity_score_threshold
                    'score_threshold': config["chroma_indexstore"]["score_threshold"]
                }
            ), 
            condense_question_llm = ChatOpenAI(
                temperature=config["OpenAI"]["condensed_question_temperature"]
            ),
            memory = memory,
            return_source_documents = config["chains"]["return_source_documents"],
            condense_question_prompt = utilz.condense_question_prompt(),
            combine_docs_chain_kwargs = { "prompt": utilz.question_answer_prompt() },
            # prints out the detail of the question condensing
            verbose = True,
            # get_chat_history = utilz.get_chat_history
        )

    def answer_question(self, question: str, chat_history: list):
        output = self.qa({'question': question})
        print(output)
        return output["answer"]

if __name__ == "__main__":
    qa = chat_retrieval()
    while True:
        print("Whats Your Question:")
        query = input()
        if query == "exit":
            break
        print(qa.answer_question(query, []))


