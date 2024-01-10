import time

from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
from langchain.document_loaders import PyPDFium2Loader # pypdfium2
import torch

loader = PyPDFium2Loader("data/article/C4RA15675G.pdf")
data = loader.load()
print(type(data), len(data))
context = []
for cxt in data:
    context.append(cxt.page_content)
context = "".join(context)
print("len document: ", len(context))
print(len(context))


tokenizer = AutoTokenizer.from_pretrained("model/falcon-40b-tokenizer",  trust_remote_code=True, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained("model/falcon-40b",  trust_remote_code=True, local_files_only=True)
start = time.time()
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
)
print("SUCCESS PIPE: ", time.time() - start)

size = "4.1"
formula = "CoFe2O4"
question = f'What is needed for synthesis of {size} nm or other size {formula} NPs? NPs means nanoparticles. Please indicate in the response all the parameters of the experiment specified in the article, including equipment and reagents and mmols. If the article does not say anything about synthesis, then answer it. Answer as fully as possible, try to take the maximum of the original text from the article. Your answer should consist of several paragraphs and be quite voluminous, while preserving the source text as much as possible'
prompt = f"""
I will give you a question and a scientific article. Please give me an answer to the question using the text from the article.
Question: {question}

Article text: {context}.
"""
# Article text: {context}.
start = time.time()
sequences = pipeline(
    prompt,
    max_length=2000,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
)
print("SUCCESS SEQ: ", time.time() - start)

for seq in sequences:
    print(f"Result: {seq['generated_text']}")
