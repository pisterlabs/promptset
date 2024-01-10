import os
import json
import pinecone
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Load vulnerability types and descriptions
vulnerability_info = {}
with open('vulnerabilitytypesdescriptions.jsonl', 'r') as f:
    for line in f:
        vulnerability = json.loads(line)
        vulnerability_info[vulnerability['vulnerabilitytype']] = vulnerability['vulnerabilitydescription']

# Initialize LangChain components
llm = ChatOpenAI(temperature=0.7, model_name="gpt-4-1106-preview")
pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENV"))
embeddings = OpenAIEmbeddings()
pinecone_index = Pinecone.from_existing_index('auditme', embeddings)
retriever = pinecone_index.as_retriever()

# Define the prompt template
template = """
You are an AI Smart Contract auditor agent in a RAG system. 
We have performed a vector search of known smart contract vulnerabilities based on the code in the USER QUESTION.
The results are below:

RELEVANT_VULNERNABILITY: {context}

With this knowledge, review the following smart contract code in USER QUESTION in detail and very thoroughly.
ONLY indentify vulnerabilities in the USER QUESTION, do not analyze the RELEVANT_VULNERNABILITY.

Think step by step, carefully. 
Is the following smart contract vulnerable to '{vulnerability_type}' attacks? 
Reply with YES or NO only. Do not be verbose. 
Think carefully but only answer with YES or NO! To help you, find here a definition of a '{vulnerability_type}' attack: {vulnerability_description}

USER QUESTION: {question}
"""

QA_CHAIN_PROMPT = PromptTemplate.from_template(
    template
)

# Process each entry in the 52samplesourcecode.jsonl
results = []
with open('52samplesourcecode.jsonl', 'r') as jsonl_input, open('52sampleresponses.jsonl', 'a') as jsonl_output:
    for line in jsonl_input:
        entry = json.loads(line)
        address = entry['address']
        source_code = entry['sourcecode']
        attack_types = entry['attacktype'].split(', ')
        
        # Retrieve relevant documents
        docs = retriever.get_relevant_documents(source_code)
        context = [doc.page_content for doc in docs]
        print(context)
        
        for attack_type in attack_types:
            vulnerability_description = vulnerability_info.get(attack_type, "Description not found.")
            
            # Construct the prompt
            prompt = QA_CHAIN_PROMPT.format(
                context = json.dumps(context),
                question = source_code,
                vulnerability_type = attack_type,
                vulnerability_description = vulnerability_description
                )
            
            # Run the prompt through the LLM
            chain = load_qa_chain(llm, chain_type="stuff", prompt=QA_CHAIN_PROMPT)
            output = chain({"input_documents": docs, "question": source_code, "vulnerability_type": attack_type, "vulnerability_description": vulnerability_description}, return_only_outputs=True)
            
            # Append the result to the entry
            results.append({
                "address": address,
                "attacktype": attack_type,
                "result": output
            })
        
        # Write the updated entry to the output file
        for result in results:
            jsonl_output.write(json.dumps(result) + '\n')
            print(result)

print("Completed processing and saved results to 52sampleresponses.jsonl")