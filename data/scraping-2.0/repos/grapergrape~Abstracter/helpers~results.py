import os
import fitz
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory

load_dotenv()

class Resulter:
    def __init__(self, window_size=1):
        self.llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k")

        self.conversation = ConversationChain(
            llm=self.llm,
            verbose=True,
            memory=ConversationBufferWindowMemory(k=window_size),
        )

    def predict(self, input):
        return self.conversation.predict(input=input)

    def get_article_text_fitz(self, pdf_file_path):
        with fitz.open(pdf_file_path) as pdf_file:
            # Initialize an empty string to hold the text of the PDF
            context = ""

            # Loop over the pages and extract the text
            for page_num in range(pdf_file.page_count):
                page = pdf_file[page_num]
                context += page.get_text("text")  
        return context

    def index_relevant_parts(self, article_text, question, chunk_size):
        results = []
        i = 0
        while i < len(article_text):
            # print chunk out of total chunks
            print(f"Chunk {i // chunk_size + 1} out of {len(article_text) // chunk_size + 1}")
            
            # Get a chunk of text
            chunk = article_text[i:i + chunk_size]
            # Check if the chunk is relevant to the question
            if self.predict(question + " " + chunk) == 'True': 
                # If it is, add it to the results
                results.append((i, chunk))
            # Move to the next chunk
            i += chunk_size
        return results

    def process_pdf_fitz(self, pdf_file_path, question, chunk_size):
        article_text = self.get_article_text_fitz(pdf_file_path)
        return self.index_relevant_parts(article_text, question, chunk_size)
    
    def summarize(self, relevant_parts, question):
        summarized_chunks = []
        
        if relevant_parts == []:
            return "No relevant parts found"
        for _, chunk in relevant_parts:
            # Prepare the prompt
            prompt = f"Summarize the findings in the following text, pretend you are a scientist writing an article: {chunk}"
            # Generate the summary
            summary = self.predict(input=prompt)
            # Append the summary to the list
            summarized_chunks.append(summary)

        # prompt the model to anwser the question with a single summary of the all the summarized chunks
        sumarized_unit = " ".join(summarized_chunks)
        print(sumarized_unit)
        final_summary_prompt = f"Anwser the query: '{question}' with a single, atleast 1500 words long, comperhensive summary of the findings in the following text: {sumarized_unit}"
        final_summary = self.predict(input = final_summary_prompt)
            

        return final_summary

def get_results(path, start_question):
    resulter = Resulter(window_size=0)
    pdf_path = path
    question = f"Does the following text contain chunks showing relevance to the question {start_question}, return a string only 'True' or 'False', no explanation for decision:"
    found_status = False
    chunk_size = 14000
    while not found_status:
        relevant_parts = resulter.process_pdf_fitz(pdf_path, question, chunk_size)
        anwser = resulter.summarize(relevant_parts, start_question)
        if len(anwser.split()) > 200:
            found_status = True
        # decrease chunk size if no relevant parts found
        chunk_size -= 1000
    return anwser

if __name__ == "__main__":
    resulter = Resulter(window_size=0)
    pdf_path = "pdf/as-74-9-1012-1.pdf"
    start_question = "What is flowcytometry and how does it preform?"
    question = f"Does the following text contain chunks showing relevance to the question {start_question}, return a string only 'True' or 'False', no explanation for decision:"
    chunk_size = 14000
    found_status = False
    while not found_status:
        relevant_parts = resulter.process_pdf_fitz(pdf_path, question, chunk_size)
        anwser = resulter.summarize(relevant_parts, start_question)
        # check if no anwser was found or if the anwser is shorter than 1500 words
        if len(anwser.split()) > 200:
            found_status = True
        # Reduce chunk size if no relevant parts found to ensure nothing was found due to cut off
        chunk_size -= 1000
    print(anwser)