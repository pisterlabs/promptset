import pandas as pd
import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from openai import OpenAI
import pinecone
from dotenv import load_dotenv

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,4,5,6,7"

# Custom LLM Setup
base_model_id = "NousResearch/Llama-2-7b-hf"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(
    base_model_id, add_bos_token=True, trust_remote_code=True)
path_llm_model = "08_Lora_and_Rag/00_trained_lora_model/lora_finetuning/llama2-7b-AmazonVPC-finetune/checkpoint-500"
ft_model = PeftModel.from_pretrained(base_model, path_llm_model)

# Custom LLM Chat Model Class


class CustomLLMChatModel:
    def __init__(self, model, tokenizer):
        # super().__init__()
        self.model = model
        self.tokenizer = tokenizer

    def _call(self, prompt, stop_words=None):
        model_input = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        self.model.eval()
        with torch.no_grad():
            output = self.model.generate(**model_input, max_new_tokens=500)[0]
            raw_output = self.tokenizer.decode(
                output, skip_special_tokens=True)
            return raw_output.split("### Answer:")[-1].strip()

    @property
    def _identifying_params(self):  # Optional _identifying_params property
        return {"model": str(self.model), "tokenizer": str(self.tokenizer)}


# Load environment variables
load_dotenv()
pinecone_key = os.getenv("PINECONE_KEY")
client = OpenAI(api_key=os.getenv("OPENAI_KEY"))

index_name = "document-embeddings"
environment = "gcp-starter"

# Function to convert text to vector using OpenAI Embeddings


def text_to_vector(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    res = client.embeddings.create(input=[text], model=model)
    return res.data[0].embedding


# Initialize Pinecone
pinecone.init(api_key=pinecone_key, environment=environment)
index = pinecone.Index(index_name)

# Langchain Setup
# vector_db = Pinecone.from_existing_index(index_name=index_name, embedding=OpenAIEmbeddings(openai_api_key=openai.api_key))
# vector_db_retriever = vector_db.as_retriever()
ans_template = """
    Context: The following API reference information has been retrieved based on the user's question. Pay attention to function names, parameters, and any mentioned errors. Use this information to provide a technically accurate answer.

    Instructions: ONLY OUTPUT A ONE PARAGRAPH ANSWER.

    Retrieved API Information: {context}
    
    QUESTION: {question}
    
    ANSWER:
"""
# prompt_for_chain = PromptTemplate(template=ans_template, input_variables=["context", "question"])
custom_llm_model = CustomLLMChatModel(ft_model, tokenizer)
# assistant = RetrievalQA.from_chain_type(llm=custom_llm_model,
#                                         retriever=vector_db_retriever,
#                                         chain_type="stuff",
#                                         chain_type_kwargs={"prompt": prompt_for_chain})

# Read CSV File
test_df = pd.read_csv(
    '06_Data/Capstone_Data/documentation_qa_datasets/Final_FILTERED_TEST_Question_Answer_Pairs.csv')

# Select a subset of the data for testing, e.g., 10%
subset_percentage = 0.1
test_subset = test_df.sample(frac=subset_percentage)

for idx, row in test_subset.iterrows():
    question = row['Question']

    # Convert question to vector
    question_vector = text_to_vector(question)

    # Query Pinecone index with the vector
    query_results = index.query(
        vector=question_vector,
        top_k=4,  # Adjust 'top_k' as needed
        include_metadata=True  # Ensure metadata is included in the results
    )

    # Extracting text from query results
    context = ' '.join([match['metadata']['text']
                       for match in query_results['matches']])

    # Generate prompt with context and question using the ans_template
    full_prompt = ans_template.format(context=context, question=question)

    # Generate answer with LLM
    llm_answer = custom_llm_model._call(full_prompt)
    test_df.loc[idx, 'llm_answer'] = llm_answer

# Save Results to New CSV
test_df.to_csv(
    '06_Data/Capstone_Data/llm_testing_results/lora_plus_rag_testing_output.csv', index=False)
