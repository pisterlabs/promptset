from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains import SimpleSequentialChain
from PyPDF2 import PdfReader

def get_mca_questions(context: str): 
   if type(context) != str:
      raise Exception("Can accept only string type as parameter to get_mca_questions()")

   if len(context.split(" ")) > 2500:
      context = " ".join(context.split(" ")[:2500])
      
   llm = OpenAI(temperature=.7)
   template = """You are a teacher preparing questions for a quiz. Given the following document, please generate 10 multiple-choice questions (MCQs) with 4 options and a corresponding answer letter based on the document.
   There should be more than one correct options

   Example question:

   Q: question here
   a. choice here
   b. choice here
   c. choice here
   d. choice here
   Correct Options: (a) or (b) or (c) or (d)

   These questions should be detailed and solely based on the information provided in the document.

   <Begin Document>
   {doc}
   <End Document>"""
   prompt = PromptTemplate(
	    input_variables=["doc", "qn_no"],
	    template=template
	)
   llm_chain = LLMChain(prompt=prompt, llm=llm)
   qs = llm_chain.run({"doc": context})
   mca_questions = [i for i in qs.split("\n\n") if i.strip() != ""]
   return mca_questions

def find_words(data):
   return len(data.split(" "))

def main():
   pages = []
   pdf_path = 'Dataset/chapter-2.pdf'
   reader = PdfReader(pdf_path)
   for page in reader.pages:
      text = page.extract_text()
      pages.append(text)
      if find_words('\n'.join(pages)) > 2500:
         break
   data = '\n'.join(pages)
   print(get_mca_questions(data))

if __name__ == "__main__":
    main()