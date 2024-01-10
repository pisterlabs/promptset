# def extract_text_from_pdf(pdf_path):
#     # Open the PDF file
#     pdf_document = fitz.open(pdf_path)
#     all_text = ''

#     # Iterate through all pages
#     for page_number in range(pdf_document.page_count):
#         # Get the page
#         page = pdf_document[page_number]

#         # Extract text from the page
#         text = page.get_text()
#         all_text += text

#         # Print or process the extracted text as needed
#         # print(f"Page {page_number + 1}:\n{text}\n")

#     # Close the PDF file
#     pdf_document.close()
#     return all_text


import requests

# text_splitter = TokenTextSplitter(chunk_size=10, chunk_overlap=0)

class URLtoPDF:
    def __init__(self) -> None:
          pass
          self.url = "https://drive.google.com/uc?id="
    def extract_file_id(self, url):
        try: 
            file_id = url.split("/file/d/")[1].split("/view")[0] # to get the code of google drive
            return file_id
        except IndexError:
            raise Exception("Invalid Google Drive link.")

    def download_file_from_url(self,entire_url):
        
        if entire_url.startswith(self.url):
            file_id = self.extract_file_id(entire_url)
            URL = self.url + file_id # downloading the pdf from gdrive
        URL = entire_url
        try:
            response = requests.get(URL)
            print(response)
        except Exception as e:
            print(str(e))

        if response.status_code == 200:
            return response.content# loading the file in memory
        else:
            print("exception")
            print(response.status_code)
            print(response.json())
            print("exception Done")
            raise Exception("Failed to download the file from Google Drive.")   

class PDFtoText:
    def __init__(self) -> None:
        pass
         
    def open_pdf(self, pdf):
        import fitz 
        import os
        if os.path.exists(str(pdf)) or isinstance(pdf,bytes):
                self.pdf = fitz.open('pdf',pdf)
                self.page_count = self.pdf.page_count
                return self.pdf
                # self.pdf.close()
        else:
            raise ValueError(f"PDF path is incorrect", pdf)
    def extract_all_text(self, pdf):
         # Open the PDF file
        self.pdf = self.open_pdf(pdf)
        
        all_text = ''

        # Iterate through all pages
        for page_number in range(self.page_count):
            # Get the page
            page = self.pdf[page_number]

            # Extract text from the page
            text = page.get_text()
            all_text += text

            # Print or process the extracted text as needed
            # print(f"Page {page_number + 1}:\n{text}\n")
        return all_text
    def extract_text_from_single_page(self,pdf, page_number):
        self.pdf = self.open_pdf(pdf)
        if page_number -1> self.page_count:
             raise ValueError("Invlaid pagenumber")
        else:
             return self.pdf[page_number-1].get_text()
    def extract_text_from_interval(self,pdf,page_number, interval =1):
        self.pdf = self.open_pdf(pdf)
        text = ""
        if page_number > self.page_count:
            raise ValueError("Invlaid pagenumber")
        else:
            # Calculate the start and end pages
            start_page = max(0, page_number - interval)
            end_page = min(self.page_count - 1, page_number + interval)

            for page_number in range(start_page, end_page + 1):
                text += self.extract_text_from_single_page(pdf=pdf, page_number=page_number)
        return text

class TexttoQuestions:
    def __init__(self) -> None:
          from langchain.text_splitter import TokenTextSplitter
          from langchain.embeddings.openai import OpenAIEmbeddings
          self.text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=5,  length_function=len)
          self.embeddings = OpenAIEmbeddings()
          self.prompt = """Create {n} expert level MCQ questions with 4 options and the correct answer on the following context. Strictly use the following format. Do not number questions,do not return anything else and do not include option number in answer. 
    Question: "Question_here"                         
    Options:
    a)
    b)
    c)
    d) 
    Answer: "Answer here"          
    """
    def get_data_chunks(self,text):
        return self.text_splitter.split_text(text)
    def create_knowledge_hub(self,chunks):
        
        from langchain.vectorstores import FAISS
        
        knowledge_hub = FAISS.from_texts(chunks, self.embeddings)
        return knowledge_hub
    
    def get_answer_LLM(
            self,
            prompt: str,
            knowledge_hub: str,
            chain_type: str = 'stuff',
        ) -> str:    
        if knowledge_hub == "":
            return ""
        from langchain.chains import RetrievalQA
        from langchain.llms import OpenAI   

        # chunks = get_data_chunks(data, chunk_size=chunk_size)  # create text chunks
        # knowledge_hub = create_knowledge_hub(chunks)  # create knowledge hub

        retriever = knowledge_hub.as_retriever(
            search_type="similarity", search_kwargs={"k": 3}
        )
        chain = RetrievalQA.from_chain_type(
            llm=OpenAI(temperature=0.3, model_name="gpt-4"),
            chain_type=chain_type,
            retriever=retriever,
            return_source_documents=True,
        )
        result = chain({"query": prompt})

        return result['result']
    
    def extract_questions_answers(self,response):
        # Extract the assistant's message
        # response = response.choices[0].message.content

        # Split the message into lines
        lines = response.split('\n')

        # Initialize lists to hold the questions, options, and answers
        questions = []
        options = []
        answers = []

        # Initialize a list to hold the current question's options
        current_options = []

        # Iterate over the lines
        for line in lines:
            if line.startswith('Question:'):
                # This is a question, so add it to the questions list
                questions.append((line[len('Question: '):]).strip().strip('"'))
            elif line.startswith('Options:'):
                        # This is the start of the options, so clear the current options list
                        current_options = []
            elif line.startswith('Answer:'):
                # This is an answer, so add the current options to the options list and the answer to the answers list
                options.append(current_options)
                answers.append((line[len('Answer: '):]).strip().strip('"'))
            elif line.startswith(('a)', 'b)', 'c)', 'd)')):
                # This is an option, so add it to the current options list
                current_options.append((line[len('a)'):]).strip().strip('"'))

        # Return the questions, options, and answers as a list of dictionaries
        return [{'question': q, 'options': o, 'answer': a} for q, o, a in zip(questions, options, answers)]
    def get_mcq_questions(self,n,text):
         prompt = self.prompt.format(n = n)
         knowledge_hub  = self.create_knowledge_hub(self.get_data_chunks(text))
         questions = self.get_answer_LLM(prompt=prompt, knowledge_hub=knowledge_hub)
         print(questions)
         try:
            return self.extract_questions_answers(questions)
         except Exception as e:
             raise Exception("Openai return type not a valid json", str(e))
         
         


         
        

class QuestionGenerator():
    def __init__(self) -> None:
        print("initializing generator")
        self.url_to_pdf = URLtoPDF()
        self.pdf_to_text = PDFtoText()
        self.text_to_questions = TexttoQuestions()
    def generate_mcq_questions_all_text(self,url,n):
        byte_array = self.url_to_pdf.download_file_from_url(url)
        text = self.pdf_to_text.extract_all_text(pdf = byte_array)
        questions = self.text_to_questions.get_mcq_questions(n,text)
        return(questions)
    def generate_mcq_questions_single_page(self,url,page_number, n):
        byte_array = self.url_to_pdf.download_file_from_url(url)
        text = self.pdf_to_text.extract_text_from_single_page(pdf = byte_array, page_number=page_number)
        questions = self.text_to_questions.get_mcq_questions(n,text)
        return(questions)
    def generate_mcq_questions_page_interval(self,url,page_number, interval, n):
        byte_array = self.url_to_pdf.download_file_from_url(url)
        text = self.pdf_to_text.extract_text_from_interval(pdf = byte_array, page_number = page_number, interval = interval)
        questions = self.text_to_questions.get_mcq_questions(n,text)
        return(questions)


    

        


def URLtoQuestions(url):
    uPDF = URLtoPDF()
    pdf = uPDF.download_file_from_url(url)
    pdftext = PDFtoText(pdf)
    text = pdftext.extract_all_text()
    texttoquestions  = TexttoQuestions(text)
    questions = texttoquestions.get_mcq_questions(10)
    return questions


        

          
if __name__ == '__main__':
#  questions = URLtoQuestions("https://drive.google.com/file/d/1TGEgTeDQAS2NyS36_KXv1ZyIcA0tFTvr/view?usp=drive_link")
#  print(questions)
    generator = QuestionGenerator()
    # generator.generate_mcq_questions("https://drive.google.com/file/d/1TGEgTeDQAS2NyS36_KXv1ZyIcA0tFTvr/view?usp=drive_link")
    questions = generator.generate_mcq_questions_all_text(url = "https://ncert.nic.in/textbook/pdf/kebo116.pdf", n = 10)
    print(questions)


