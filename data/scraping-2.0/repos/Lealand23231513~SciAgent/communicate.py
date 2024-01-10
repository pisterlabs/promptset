from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.llms import OpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.prompts import PromptTemplate

import os
os.environ["OPENAI_API_KEY"] = ""

template_1 = """Write a summary of this paper,
which should contain introduction of research process and achievements, and the innovation or breakthrough in the research field:
 
{text}
 
Summary Here:"""

template_2 = """Assistant is a large language model trained by OpenAI.
 
Assistant is designed to be able to assist with human refer to paper, from reading paper uploaded and answering questions about these papers to help human have a general idea of them. 
    
The task can be divided for two parts: Read the paper in parts, and answer the questions. Questions may be like: What is the research filed and interests of this paper? What is the achievement or finding in this paper? Or the significance of this research, etc.

Additionally, assistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand a few of papers, and can do some simple comparison between these papers to meet human’s requests. 

Overall, Assistant is a powerful tool for paper review. Assistant can improve the efficiency of researchers in accessing literature and obtaining reference information when conducting research.

Now, assistant will get the summary of paper and the question raised by human, assistant should answer the question.  

{history}

The summary of the paper and human's question:
{human_input}

Assistant:

"""

def communicate(path:str, question:str):

    if(path.split(".")[-1] == 'pdf'):
        loader = PyPDFLoader(path)
    elif(path.split(".")[-1] == 'docx'):
        loader = Docx2txtLoader(path)
    else:
        print("论文文件格式错误")
        os._exit(0)

    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    prompt_1 = PromptTemplate(
        template=template_1, 
        input_variables=["text"]
    )

    chain = load_summarize_chain(
        llm = OpenAI(temperature=0.2, max_tokens=1000, model="gpt-3.5-turbo-instruct"), 
        chain_type="map_reduce", 
        return_intermediate_steps=False, 
        map_prompt=prompt_1, 
        combine_prompt=prompt_1
    )
    summary = chain({"input_documents": docs}, return_only_outputs=True)["output_text"]

    prompt_2 = PromptTemplate(
        input_variables=["history", "human_input"], 
        template=template_2
    )
    
    chatgpt_chain = LLMChain(
        llm=OpenAI(temperature=0, model="gpt-3.5-turbo-instruct"), 
        prompt=prompt_2, 
        verbose=True, 
        memory=ConversationBufferWindowMemory(k=2),
    )

    question = "Please introduce the research achievement of this paper."
    input_key = "Summary:\n " + summary + "\n" + "Human's question:\n" + question
    output = chatgpt_chain.predict(human_input=input_key)

    return output

def summarizer(papers_info):
    ai_response = []
    for i,paper_info in enumerate(papers_info):
        file_path = download_arxiv_pdf(paper_info)
        papers_info[i]["path"] = file_path
        #communicate_result = communicate(file_path, "question")
        ai_response.append(f"Succesfully download <{paper_info['Title']}> into {file_path} !\n The summary result is as below:\n{summary_result}")
    return "\n".join(ai_response)
    
if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv()
    test_file = "TEST  Text Prototype Aligned Embedding to Activate LLM's Ability for Time Series.pdf"
    #communicate_result = communicate(file_path, "question")
    print(summary_result)
    #summary("C:\Pythonfiles\langchain_try\summary\\test_paper\Attention Is All You Need.pdf")