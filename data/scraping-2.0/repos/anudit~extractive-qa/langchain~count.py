from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.llms.openai import OpenAI
from tqdm import tqdm

from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import DeepLake
from langchain.document_loaders import TextLoader


print('Loading directory')
loader = DirectoryLoader('../census/census_csvs', glob="**/*.csv", loader_cls=CSVLoader)
print('Loading data')
data = loader.load()
print('found', len(data))

# data[0].metadata.update({"description": 'new!'})

openai_llm = OpenAI()

print('Calculating tokens')
total_tokens = 0
for doc in tqdm(data):
    total_tokens+= openai_llm.get_num_tokens(doc.page_content)

print('total_tokens:', total_tokens)

# cost per 1k tokens = $0.4/1M
# 
# found 34,041,237 docs
# 9,613,526,181 total, 282.4 tokens avg
# 0.4 * 9,613 = 3845

# 1 => 3.642kb
# 28,000 => 100.2MB
# found 34,041,237 docs
