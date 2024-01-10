from langchain import LLMChain
from transformers import AutoTokenizer, logging
from langchain.llms import TextGen
from langchain.memory import SimpleMemory, VectorStoreRetrieverMemory, ConversationKGMemory, CombinedMemory, ConversationBufferMemory
from langchain.prompts.prompt import PromptTemplate
from Custom_Agent import get_umls_id, CustomLLMChain, create_gptq_pipeline, get_similar_compounds
import re
from customGraphCypherQA import KnowledgeGraphRetrieval
from prompts import Entity_Extraction_Template, PUBMED_AGENT_TEMPLATE_VICUNA, Final_chain_template, PUBMED_AGENT_TEMPLATE, Coherent_sentences_template, Entity_Extraction_Template_alpaca
from PubmedEmbeddings import PubmedSearchEngine, get_abstracts
logging.set_verbosity(logging.CRITICAL)

question = "Could there be a negative drug-drug interaction between sildenafil and donepezil?"
model_url = "https://newspaper-third-enjoy-frequent.trycloudflare.com/"
llm = TextGen(model_url=model_url, max_new_tokens=512)
Entity_extraction_prompt = PromptTemplate(template=Entity_Extraction_Template_alpaca, input_variables=["input"])
Entity_extraction_chain = CustomLLMChain(prompt=Entity_extraction_prompt, llm=llm, output_key="output",)
result = Entity_extraction_chain.run(question)
print("Raw result:", result)
entities = result
entities_umls_ids = {}

for entity in entities:
    umls_id = get_umls_id(entity)
    entities_umls_ids[entity] = umls_id

names_list = []

for entity, umls_info_list in entities_umls_ids.items():
    if umls_info_list:
        umls_info = umls_info_list[0]
        match = re.search(r"Name: (.*?) UMLS_CUI: (\w+)", umls_info)
        if match:
            umls_name = match.group(1)
            umls_cui = match.group(2)
            names_list.append(umls_name)
        else:
            names_list.append(entity)
    else:
        names_list.append(entity)

print(names_list)

uri = "neo4j://localhost:7687"
username = "neo4j"
password = "NeO4J"

Graph_query = KnowledgeGraphRetrieval(uri, username, password, llm)
result = Graph_query._call(question, generate_an_answer=True)
print(result["result"])
context = result["result"]
context = context.replace('\n\n', '\n')

if "associated_genes_string" in result:
    associated_genes_string = result["associated_genes_string"]
    print("The generated context and associated genes list:", context, associated_genes_string)
    final_context = context, associated_genes_string
    print(final_context)
else:
    print("The generated context:", context)
    final_context = context
    print(final_context)

#similar_abstracts = pubmed_vector_retrieval("Could there be a negative drug-drug interaction between sildenafil and donepezil?")
search_engine = PubmedSearchEngine()
PubmedSearchResults = search_engine.query(question)
print(PubmedSearchResults)

prompt = PromptTemplate(
    template=Final_chain_template,
    input_variables=["question", "input", "relevant_pubmed_abstracts"]
)

memory = ConversationBufferMemory()
model_name_or_path = "TheBloke/airoboros-33B-gpt4-1.4-GPTQ"
model_basename = "airoboros-33b-gpt4-1.4-GPTQ-4bit--1g.act.order"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
pipe = create_gptq_pipeline(model_name_or_path, model_basename, tokenizer)
#llm = HuggingFacePipeline(pipe)

Pubmed_agent_chain = LLMChain(llm=llm, prompt=prompt)
Final_output = Pubmed_agent_chain.predict(question=question, input=final_context, relevant_pubmed_abstracts=PubmedSearchResults)
print(Final_output)





