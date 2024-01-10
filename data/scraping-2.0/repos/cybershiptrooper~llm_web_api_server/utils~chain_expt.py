from langchain.document_loaders import PyMuPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import SentenceTransformersTokenTextSplitter
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.chains.summarize import load_summarize_chain
from . import *
from utils.nlp_trainers import LDATrainer
import asyncio

class ContextBasedGenerator:
    def __init__(self, pdf_paths=None, k=5) -> None:
        prompt_template = """You are a document creator that creates html sections based on prompts and context, which shall provide details required for the job. 
        Context shall be provided in chunks: 'ctx number', 'total ctx' provide the current chunk number, and the total chunks to be recieved respectively. 'ctx summary' provides a short summary of all the context you are to recieve. This shall be useful for any part of the document that needs summarising. You need to create valid, logical and visually pleasing html sections that will be later combined inside <html></html> tags(externally provided) to form a complete html document. For ctx number = 1, you will need to add an introductory section before anything, and you must add a heading for the document. For ctx number = total ctx, you will need to add a conclusion section. Important: For all other ctx numbers, you cannot add these sections.
        The context and the context summary are based on *people's* views on various topics: you must rephrase them as a new person's view. Do not copy them as-is. 
        You may include css, and up to 1 image in the html script. The image "alt" tag will be used as description for an image generation model to generate an image. "src" tag should be an empty string and description should be in English. Add images only if necessary or asked by prompt. Now create a document based on the context and prompt given below:
        Context: {context}
        Prompt: {prompt}
        ctx number: {context_number}
        total ctx: {total_context}
        ctx summary: {context_summary}
        html:"""
        self.k = 5
        self.PROMPT = PromptTemplate(
        template=prompt_template, input_variables=
            ["context", "prompt", "context_summary", "context_number", "total_context" ]
        )
        self.llm = OpenAI(model_name="text-davinci-003", max_tokens=2950, temperature=0.0)
        self.chain = LLMChain(llm=self.llm, prompt=self.PROMPT)
        self.summary_chain = load_summarize_chain(self.llm, chain_type="map_reduce")
        self.text_splitter = SentenceTransformersTokenTextSplitter(chunk_size=1024, chunk_overlap=50)
        if(pdf_paths is not None):
            self.generate_db_from_pdf(pdf_paths)
    
    def generate_db_from_pdf(self, pdf_paths):
        texts = []
        titles = []
        for pdf_path in pdf_paths:
            loader = PyMuPDFLoader(pdf_path)
            document = loader.load()
            titles.append(document[0])
            texts+=self.text_splitter.split_documents(document)
        self.max_search_len = len(texts)
        self.texts = texts
        self.titles = titles

    @property
    def db(self):
        vectordb = Chroma.from_documents(documents=self.texts, 
                                        embedding=OpenAIEmbeddings())
        return vectordb

    def generate_chain_response(self, prompt):
        docs = self.get_top_k_documents(prompt)
        with open("docs.log", "w") as f:
            for doc in docs:
                f.write(doc.page_content)
                f.write("\n===========\n")
        # raise Exception("stop")
        print("Summarising documents")
        summary = self.summary_chain.run(docs)
        print("Summarised documents: ", summary, "========", sep="\n")
        inputs = [
            {
                "context": doc.page_content, 
                "prompt": prompt,
                "context_summary": summary,
                "context_number": idx,
                "total_context": len(docs)
            } 
                  for doc, idx in zip(docs, range(1, len(docs)+1))]
        gpt_response = self.chain.apply(inputs)
        output = ""
        with open("gpt.log", "w") as f:
            for resp in gpt_response:
                f.write(resp["text"])
                f.write("\n============\n")
                output += resp["text"]
        output = [resp["text"] for resp in gpt_response]
        output = "<html>\n" + "\n".join(output) + "\n</html>"
        return output

    def get_top_k_documents(self, prompt):
        assert self.db is not None, "Database not initialized"
        k = self.k
        prompt_result = self.db.similarity_search_with_score(prompt, k=k)
        
        docs = []
        for result in prompt_result:
            score = result[1]
            if(score >= 0.5):
                break
            docs.append(result[0])
        
        if(len(docs) < 1):
            print("No documents found with similarity score less than 0.5. Looking for generic results.")
            docs = self.get_generic_results()
        else:
            print("Found documents with similarity score less than 0.5: returning")
        
        return docs
    
    def get_generic_results(self):
        # TODO: optimize this
        k=min(self.k, self.max_search_len)
        # docs = self.db.max_marginal_relevance_search(
        #           ' ', k=k, lambda_mult=0.0)
        print("Creating prompts based on LDA keywords")
        text_list = [text.page_content for text in self.titles]
        lda = LDATrainer(k, text_list, passes=10)
        smart_queries = lda.make_smart_queries()
        print("Queries: ", smart_queries, sep="\n")
        docs = []
        for query in smart_queries:
            prompt_result = self.db.similarity_search_with_score(query, k=1)
            for result in prompt_result:
                score = result[1]
                if(score >= 0.5):
                    break
                docs.append(result[0])
        print("returning generic results")
        return docs
    
    async def summarise(self, texts):
        return await self.summary_chain.arun(texts)
    
    async def _generate_chain_response_from_inputs(self, inputs):
        return await self.chain.aapply(inputs)
    