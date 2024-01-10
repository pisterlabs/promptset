from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import PyPDFLoader
from langchain import OpenAI, PromptTemplate
import glob

llm = OpenAI(temperature=0.2)


def summarize_pdfs_from_folder(pdfs_folder):
    summaries = []
    for pdf_file in glob.glob(pdfs_folder + "/*.pdf"):
        loader = PyPDFLoader(pdf_file)
        docs = loader.load_and_split()
        chain = load_summarize_chain(llm, chain_type="map_reduece")
        summary = chain.run(docs)
        print("Summary for: ", pdf_file)
        print(summary)
        print("\n")
        summaries.append(summary)

    return summaries


summaries = summarize_pdfs_from_folder("./pdfs")
