import os  

# langchain_core modules
from operator import itemgetter  
from langchain_core.runnables import RunnablePassthrough, RunnableLambda 
from langchain.memory import ConversationBufferMemory  
from langchain.prompts import ChatPromptTemplate, PromptTemplate 

# langchain_core messages
from langchain_core.messages import AIMessage, HumanMessage, get_buffer_string  

# Vertex related imports
import vertexai  
from langchain.llms import VertexAI  
from vectorDB import VectorStorage  
from langchain.schema import format_document  

template = """Answer the question at the end based only on the following context:
{context}
Chat History:
{chat_history}
Question: {question}
"""
ANSWER_PROMPT = ChatPromptTemplate.from_template(template)



DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")


def _combine_documents(
    docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"
):
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)

class LLMChain:
    def __init__(self, retriever, llm_type="vertexai", **llm_params):
        self.retriever = retriever
        self.llm_type = llm_type
        self.llm_params = llm_params
        self.llm = self.initialize_llm()

        # Build the LLM chain
        self.build_chain()

    def initialize_llm(self):
        if self.llm_type == "vertexai":
            # Initialize Vertex AI LLM
            PROJECT_ID = 'zeta-yen-319702'
            REGION = 'us-central1'
            BUCKET = 'gs://zeta-yen-319702-temp/'

            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '../svc-gcp-key.py'

            vertexai.init(
                project=PROJECT_ID,
                location=REGION,
                staging_bucket=BUCKET
            )

            return VertexAI(
                model_name=self.llm_params.get("model_name", "text-unicorn"),
                max_output_tokens=self.llm_params.get("max_output_tokens", 256),
                temperature=self.llm_params.get("temperature", 0.8),
                top_p=self.llm_params.get("top_p", 0.8),
                top_k=self.llm_params.get("top_k", 5),
                verbose=self.llm_params.get("verbose", False),
            )


    def build_chain(self):
        # Step 1: Load memory
        loaded_memory = RunnablePassthrough.assign(
            chat_history=itemgetter("chat_history"),
            question=itemgetter("question")
        )

        # Step 2: Retrieve documents
        retrieved_documents = {
            "docs": itemgetter("question") | self.retriever,
            "question": lambda x: x["question"],
            "chat_history": lambda x: get_buffer_string(x["chat_history"]),
        }

        # Step 3: Construct inputs for the final prompt
        final_inputs = {
            "context": lambda x: _combine_documents(x["docs"]),
            "question": itemgetter("question"),
            "chat_history": itemgetter("chat_history")
        }

        # Step 4: Return answers
        answer = {
            "answer": final_inputs | ANSWER_PROMPT | self.llm,
            "docs": itemgetter("docs"),
            "prompt" : final_inputs | ANSWER_PROMPT,
        }

        # Construct the final chain
        self.final_chain = loaded_memory | retrieved_documents | answer

    def invoke(self, inputs):
        return self.final_chain.invoke(inputs)




# Test the LLMChain
if __name__ == "__main__":
    # Example usage for testing:

    # Instantiate the LLMChain with your retriever and llm type
    vdb = VectorStorage()
    retriever = vdb.load_db_and_get_retriever("faiss_index")  # Your retriever instantiation goes here
    llm_chain = LLMChain(retriever, llm_type="vertexai", model_name="text-bison", max_output_tokens=256)

    # Define inputs for the chain
    inputs = {
        "question": "What activites does it perform?",
        "chat_history": [
            HumanMessage(content="What is cardinal health"),
            AIMessage(content="Cardinal health is Dublin Ohio based organisation"),
        ]
    }

    # Invoke the LLMChain
    result = llm_chain.invoke(inputs)
    print(result['answer'])
    print("------------------------------------")
    print(result['docs'])
    print("------------------------------------")
    print(result['prompt'])
