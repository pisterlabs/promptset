from langchain.llms import OpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from utils import check_and_exit, get_api_key
check_and_exit("OPENAI_API_KEY")
print("OPENAI_API_KEY found")
openai_api_key=get_api_key()
llm = OpenAI(temperature=0, openai_api_key=openai_api_key)

with open('data/good.txt', 'r') as file:
    text = file.read()
print(text[:280])

num_tokens = llm.get_num_tokens(text)
print (f"There are {num_tokens} tokens in your file")


text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n"], chunk_size=5000, chunk_overlap=350)
docs = text_splitter.create_documents([text])

print (f"You now have {len(docs)} docs intead of 1 piece of text")

# Get your chain ready to use
chain = load_summarize_chain(llm=llm, chain_type='map_reduce') 
output = chain.run(docs)
print (output)
