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

# Load vulnerability types and descriptions in advance
vulnerability_types = []
with open('vulnerabilitytypesdescriptions.jsonl', 'r') as vuln_file:
    for line in vuln_file:
        vuln_info = json.loads(line)
        vulnerability_types.append((vuln_info['vulnerabilitytype'], vuln_info['vulnerabilitydescription']))

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

RELEVANT_VULNERNABILITIES: {context}

With this knowledge, review the following smart contract code in USER QUESTION in detail and very thoroughly.
ONLY indentify vulnerabilities in the USER QUESTION, do not analyze the RELEVANT_VULNERNABILITIES.

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
results_dir = '52results'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Process each entry in the 52samplesourcecode.jsonl
with open('52samplesourcecode.jsonl', 'r') as jsonl_input:
    for contract_id, line in enumerate(jsonl_input, start=1):
        #if contract_id < 39:  # Skip to contract_id 39
        #    continue
        entry = json.loads(line)
        address = entry['address']
        source_code = entry['sourcecode']
        known_attack_types = entry['attacktype'].split(', ')
        
        print(f"Analyzing contract {contract_id} - {address}...")
        
        # Filename for the contract's results
        result_filename = f"{contract_id} - {address}.jsonl"
        result_filepath = os.path.join(results_dir, result_filename)
        
        # Open the result file for this contract
        with open(result_filepath, 'a') as jsonl_output:
            for attack_type, attack_description in vulnerability_types:
                print(f"Testing {attack_type}...")
                
                # Retrieve relevant documents
                docs = retriever.get_relevant_documents(source_code)
                context = [doc.page_content for doc in docs]
                
                # Construct the prompt
                prompt = QA_CHAIN_PROMPT.format(
                    context=json.dumps(context),
                    question=source_code,
                    vulnerability_type=attack_type,
                    vulnerability_description=attack_description
                )
                
                # Run the prompt through the LLM
                chain = load_qa_chain(llm, chain_type="stuff", prompt=QA_CHAIN_PROMPT)
                output = chain({"input_documents": docs, "question": source_code, "vulnerability_type": attack_type, "vulnerability_description": attack_description}, return_only_outputs=True)
                
                # Determine the categorization of the result
                model_output = output['output_text'].strip().upper()  # Normalize response to uppercase for comparison
                is_attack_type_present = attack_type in known_attack_types
                if model_output == "YES" and is_attack_type_present:
                    outcome = "True Positive"
                elif model_output == "NO" and not is_attack_type_present:
                    outcome = "True Negative"
                elif model_output == "YES" and not is_attack_type_present:
                    outcome = "False Positive"
                elif model_output == "NO" and is_attack_type_present:
                    outcome = "False Negative"
                else:
                    outcome = "Unknown"

                # Result entry
                result = {
                    "address": address,
                    "attacktype": ', '.join(known_attack_types),
                    "attacktypetest": attack_type,
                    "attacktypematch": is_attack_type_present,
                    "result": output,
                    "outcome": outcome
                }
                
                # Write the result to the output file and flush
                jsonl_output.write(json.dumps(result) + '\n')
                jsonl_output.flush()
                print(f"Contract {contract_id} - {address}, result for {attack_type}: {outcome}")
                time.sleep(2)

print("Completed.")