from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import HuggingFacePipeline
from langchain.document_loaders import DirectoryLoader, TextLoader, JSONLoader
from langchain.prompts import load_prompt, PromptTemplate

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModelForSeq2SeqLM
from langchain.schema import Document
model_id = 'google/flan-t5-xxl'
tokenizer = AutoTokenizer.from_pretrained(model_id, max_length=512, load_in_8bit=True)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id, max_length=512, load_in_8bit=True)

pipe = pipeline(
    "text2text-generation",
    model=model, 
    tokenizer=tokenizer, 
    max_length=512
)

local_llm = HuggingFacePipeline(pipeline=pipe)
doc_path="/data/raw/alarms/by_receiver/aris.json"
loader = JSONLoader(file_path=doc_path, jq_schema=".alerts[].annotations.description")
document = loader.load()
document = [Document(page_content='server a is overloaded with requests', metadata={'seq_num': 1})]
print(document)

query="How can I fix the problem that is in the document?"

prompt = """

"""

PromptTemplatE()
chain = load_qa_chain(local_llm, chain_type="refine", refine_prompt=prompt)
print(chain.run())
#print(chain({"input_documents": document, "question": query}, return_only_outputs=True))