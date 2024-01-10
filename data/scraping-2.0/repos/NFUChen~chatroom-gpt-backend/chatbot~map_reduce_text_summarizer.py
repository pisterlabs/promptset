from langchain import OpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import PromptTemplate

class MapReduceTextSummarizer:
    def __init__(self, openai_api_key: str,question: str,text: str, text_splitter: RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n"], chunk_size=2000, chunk_overlap=100)) -> None:
        map_prompt = """
        Write a concise summary under 1000 words for the following text without losing the details of answering the following question: "{question}"
        "{text}"
        CONCISE SUMMARY:
        """.format(question = question, text= "{text}")
        combine_prompt = """
        Please provide a comprehensive summary, up to 400 words in length (feel free to maintain this length for detail preservation), for the given text while addressing the following question: "{question}" enclosed in triple backquotes.
        Present your response in bullet points, highlighting the text's essential aspects.
        ```{text}```
        BULLET POINT SUMMARY:
        """.format(question=question, text="{text}")

        self.text_splitter = text_splitter
        self.question = question
        self.text = text
        self.map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text"])
        self.combine_prompt_template = PromptTemplate(template=combine_prompt, input_variables=["text"])
        llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
        self.summary_chain = load_summarize_chain(llm=llm,
            chain_type='map_reduce',
            map_prompt=self.map_prompt_template,
            combine_prompt=self.combine_prompt_template,
            verbose=True
        )
    def summarize(self) -> str:
        docs = self.text_splitter.create_documents([self.text])
        return self.summary_chain.run(docs)
    