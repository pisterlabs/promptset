import os

from langchain.llms import OpenAI, HuggingFacePipeline
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.agents import tool, AgentType
from langchain.agents.initialize import initialize_agent
from langchain.agents.agent_toolkits import (
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo,
)

from transformers import (
    PegasusTokenizer,
    PegasusForConditionalGeneration,
    BertTokenizer,
    BertForSequenceClassification,
    pipeline
)

os.environ['OPENAI_API_KEY'] = 'YOUR_API_KEY'

### Load and store documents in VectorDB

# specify document path and metadata
doc_info = {
    'path': '../documents/financial_annual_report_2022.pdf',
    'name': 'financial_annual_report_2022',
    'description': "a finance company's annual report documentation for the year 2022"
}

# load and split PDF document
loader = PyPDFLoader(doc_info['path'])
pages = loader.load_and_split()
print(f'num_pages: {len(pages)}')

# use only the first 10 pages to save compute
pages = pages[:10]

# add document to vector database
faiss_index = FAISS.from_documents(pages, OpenAIEmbeddings())
vectorstore_info = VectorStoreInfo(name=doc_info['name'],
                                   description=doc_info['description'],
                                   vectorstore=faiss_index)
toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info)

### Initialize models

# initialize base LLM
llm = OpenAI(temperature=0)

# initialize financial summarizer model
summary_model_name = "human-centered-summarization/financial-summarization-pegasus"
summary_tokenizer = PegasusTokenizer.from_pretrained(summary_model_name)
summary_model = PegasusForConditionalGeneration.from_pretrained(summary_model_name)
summary_hf = HuggingFacePipeline(pipeline=pipeline('summarization',
                                 model=summary_model,
                                 tokenizer=summary_tokenizer))

# initialize financial sentiment analysis model
sentiment_model_name = 'yiyanghkust/finbert-tone'
finbert = BertForSequenceClassification.from_pretrained(sentiment_model_name,
                                                        num_labels=3)
finbert_tokenizer = BertTokenizer.from_pretrained(sentiment_model_name)
finbert_pipe = pipeline("sentiment-analysis", model=finbert,
                        tokenizer=finbert_tokenizer)

### Create custom tools

# define custom langchain tools
@tool
def financial_summary(text: str) -> str:
    """Returns the financial summary of the input text, \
    use this for any queries related to finding the summary \
    of the input. However, remember to extract the relevant snippet \
    from `financial_annual_report_2022` first, before using this function.
    The input should always be a string related \
    to financial information, and this function always returns \
    the summary of this financial information."""
    return summary_hf.predict(text)

@tool
def financial_sentiment(text: str) -> str:
    """Returns the financial sentiment of the input text, \
    use this for any queries related to finding the sentiment \
    of the input. However, remember to extract the relevant snippet \
    from `financial_annual_report_2022` first, before using this function.
    The input should always be a string related \
    to financial information, and this function always returns \
    the sentiment of this financial information."""
    return finbert_pipe(text)

### Initialize agents

# create retrieval agent
retrieval_agent = create_vectorstore_agent(llm=llm,
                                           toolkit=toolkit,
                                           verbose=True)

# create retrieval + summary agent
summary_agent = initialize_agent(toolkit.get_tools() + [financial_summary],
                                 llm,
                                 agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
                                 handle_parsing_errors=True,
                                 verbose=True)

# create retrieval + summary + sentiment analysis agent
sentiment_agent = initialize_agent(toolkit.get_tools() + [financial_summary,
                                                          financial_sentiment],
                                  llm,
                                  agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
                                  handle_parsing_errors=True,
                                  verbose=True)

### Results

#### Retrieval-only agent
query = "How much was Capital One's loan growth and revenue in 2022 compared to 2021?"
retrieval_agent.run(query)

#### Retrieval + summary agent
summary_agent.run(f'{query} After retrieving, summarize the information.')

#### Retrieval + summary + sentiment analysis agent
sentiment_agent.run(f'{query} After retrieving, provide the finanical sentiment.')


