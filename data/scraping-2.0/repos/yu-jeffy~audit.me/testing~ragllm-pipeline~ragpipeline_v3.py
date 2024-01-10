import os
import json
import pinecone
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# Load vulnerability types and descriptions
vulnerability_info = {}
with open('vulnerabilitytypesdescriptions.jsonl', 'r') as f:
    for line in f:
        vulnerability = json.loads(line)
        vulnerability_info[vulnerability['vulnerabilitytype']] = vulnerability['vulnerabilitydescription']

# Initialize LangChain components
llm = ChatOpenAI(temperature=0.7, model_name="gpt-4-1106-preview", max_tokens=5)
pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENV"))
embeddings = OpenAIEmbeddings()
pinecone_index = Pinecone.from_existing_index('auditme', embeddings)
retriever = pinecone_index.as_retriever(search_kwargs={"k": 5})

# Define the prompt template
template = """
You are an AI Smart Contract auditor agent in a RAG system. 
We have performed a vector search of known smart contract vulnerabilities based on the code in the USER QUESTION.
The results are below:

RELEVANT_VULNERABILITIES: {context}

With this knowledge, review the following smart contract code in USER QUESTION in detail and very thoroughly.
ONLY indentify vulnerabilities in the USER QUESTION, do not analyze the RELEVANT_VULNERABILITIES.

Think step by step, carefully. 
Is the following smart contract vulnerable to '{vulnerability_type}' attacks? 
Reply with YES or NO only. Do not be verbose. 
Think carefully but only answer with YES or NO! To help you, find here a definition of a '{vulnerability_type}' attack: {vulnerability_description}

USER QUESTION: {question}
"""

QA_CHAIN_PROMPT = PromptTemplate.from_template(
    template
)

# Create the results directory if it doesn't exist
results_dir = '52resultsrepeatability'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Process each entry in the 52samplesourcecode.jsonl
with open('52samplesourcecode.jsonl', 'r') as jsonl_input:
    for contract_id, line in enumerate(jsonl_input, start=1):
        # Skip to contract_id 40
        if contract_id < 46:
            continue
        entry = json.loads(line)
        address = entry['address']
        source_code = entry['sourcecode']
        attack_types = entry['attacktype'].split(', ')
        
        # Concatenate attack types and their descriptions
        concatenated_attack_types = ', '.join(attack_types)
        concatenated_descriptions = ', '.join([vulnerability_info.get(atk, "Description not found.") for atk in attack_types])
        
        print(f"Analyzing contract {contract_id} - {address}...")
        
        # Filename for the contract's results
        result_filename = f"{contract_id} - {address}.jsonl"
        result_filepath = os.path.join(results_dir, result_filename)
        
        # Open the result file for this contract
        with open(result_filepath, 'a') as jsonl_output:
            for i in range(5):  # Run each contract 38 times
                print(f"Contract {contract_id} - {address}, run {i+1} of 5...")
                # Retrieve relevant documents
                docs = retriever.get_relevant_documents(source_code)
                context = [doc.page_content for doc in docs]
                
                # Construct the prompt
                prompt = QA_CHAIN_PROMPT.format(
                    context=json.dumps(context),
                    question=source_code,
                    vulnerability_type=concatenated_attack_types,
                    vulnerability_description=concatenated_descriptions
                )
                    
                # Run the prompt through the LLM
                chain = load_qa_chain(llm, chain_type="stuff", prompt=QA_CHAIN_PROMPT)
                output = chain({"input_documents": docs, "question": source_code, "vulnerability_type": concatenated_attack_types, "vulnerability_description": concatenated_descriptions}, return_only_outputs=True)
                    
                # Result entry
                result = {
                    "address": address,
                    "attacktype": concatenated_attack_types,
                    "result": output
                }
                    
                # Write the result to the output file
                jsonl_output.write(json.dumps(result) + '\n')
                jsonl_output.flush()
                print(f"Contract {contract_id} - {address}, run {i+1} of 5 result: {result}")
                time.sleep(2)

print("Completed.")