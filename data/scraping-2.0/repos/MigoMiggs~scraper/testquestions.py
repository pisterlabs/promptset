from dotenv import load_dotenv, find_dotenv
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts.chat import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate
)
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains import (RetrievalQA,
                              RetrievalQAWithSourcesChain
                              )
from utilities.utils import parse_assessment_file, get_gpt_model
from scraper.settings import Config
from langchain.docstore.document import Document

'''
This script is used to test the questions from the assessment_questions.txt file

This will run all the questions from the assessment_questions.txt file and write the results to the assessment_questions_results.txt file

'''


_ = load_dotenv(find_dotenv())
embeddings = OpenAIEmbeddings()

config = Config('./config/config_coo.yaml')

llm_model = "gpt-3.5-turbo-16k"
llm = get_gpt_model(True, llm_model)

persist_directory = './chroma_clean_ada/'
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

print(f"Total chunks: {vectordb._collection.count()}")

retriever = MultiQueryRetriever.from_llm(
    retriever=vectordb.as_retriever(search_kwargs={"k": 7}), llm=llm
)

# Make sure that our retriever gets back 7 results
# retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k":7})

#### System Prompt Construction

system_template = """" \
Instructions:
* You are an evaluator that knows about Natural Disaster Recovery. \
* User will provider a multiple choice question and the possible answers below. \
* You will pick the best answer based on the included pieces of context. The questions \
will always go from 1 - 6, the 6th answer is always "I don't know." 
* Answers 1 - 5 will go from low to high, from the perspecive of how good \
the adherence is to the provided question. 
* These questions and answers are used for Natural Disaster Readiness. \
* The Community Resilience Assessment Framework and Tools (CRAFT) \
* Equitable Climate Resilience (ECR) platform is a resource for cities to assess and strengthen their resilience - \
the ability to to mitigate, respond to, and recover from crises. Also, after the answer, you will explain how you got to the answer, \ 
referrring to the pieces of context that gave you the answer. \n\
-------------------- \n\
Context:
{context}
"""
system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
human_template = """ {question} """
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
ChatPromptTemplate.input_variables = ["question"]

qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=retriever,
    chain_type_kwargs={
        "prompt": chat_prompt
    },
    return_source_documents=True
)

# Assuming the content is stored in 'questions.txt'
parsed_data = parse_assessment_file('./test-data/assessment_questions.txt')

# Open or create file to write results to
with open('./test-data/assessment_questions_results.txt', 'w') as file, open('./test-data/results_context.txt',
                                                                             'w') as context_file:
    # empty file first
    file.write("")

    # Print the parsed data
    for entry in parsed_data:
        print(f"Question ID: {entry['question_id']}")
        print(f"Question: {entry['question']}")
        print("Answers:")
        print(entry['answers'])

        query = entry['question'] + '\n\n Possible answers: \n' + entry['answers']
        result = qa_chain({"query": query})

        print("Answer with explanation: \n")
        print(result["result"])

        # Write to the file the question id, question, and the result answer with explanation
        file.write(f"Question ID: {entry['question_id']}\n")
        file.write(f"Question: {entry['question']}\n")
        file.write("Answer with explanation:\n")
        file.write(result["result"] + '\n')

        file.write("\n-----\n")

        # Write the question id, plus the context
        docs = result.get("source_documents", [])
        i=0
        d: Document
        for d in docs:
            context_file.write(f"Question ID: {entry['question_id']}\n\n")
            i += 1
            context_file.write(f"Document {i}:\n")
            context_file.write(f"{d.page_content}")
            context_file.write(f"\nSource: {d.metadata['source']}\n\n")

        context_file.write(f"-------------------------------\n\n")
        print("\n-----\n")
