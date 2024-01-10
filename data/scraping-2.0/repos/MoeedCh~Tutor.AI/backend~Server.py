import os

from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from Embeddings import VectorDatabase


# class Runner:
#
#     def __init__(self):
#         self.llm = ChatOpenAI(model_name="gpt-3.5-turbo")
#         self.vdb = VectorDatabase()
#
#     def generate_markdown(self, path_to_epub, prompt):
#         # Vectorize the chapters in the book and returns the number of chapters
#         num_chapters = self.vdb.add_chapters_from_epub(path_to_epub)
#
#         os.makedirs("cache/" + os.path.basename(path_to_epub), exist_ok=True)
#
#         for i in range(num_chapters):
#             chat_history = []
#             qa = ConversationalRetrievalChain.from_llm(self.llm, retriever=self.vdb.load_index(
#                 os.path.basename(path_to_epub) + "/Chapter " + str(i)).as_retriever())
#             result = qa({"question": prompt, "chat_history": chat_history})
#             with open("cache/" + os.path.basename(path_to_epub) + "/" + str(i) + ".md", "w+") as f:
#                 f.write(str(result["answer"]))


import multiprocessing


llm = ChatOpenAI(model_name="gpt-3.5-turbo")
vdb = VectorDatabase()
def process_chapter(path_to_epub, prompt, chapter_index):
    chat_history = []
    qa = ConversationalRetrievalChain.from_llm(llm, retriever=vdb.load_index(
        os.path.basename(path_to_epub) + "/Chapter " + str(chapter_index)).as_retriever())
    result = qa({"question": prompt, "chat_history": chat_history})
    with open("cache/" + os.path.basename(path_to_epub) + "/" + str(chapter_index) + ".md", "w+") as f:
        f.write(str(result["answer"]))
        f.close()

def generate_markdown(path_to_epub, prompt):
    num_chapters = vdb.add_chapters_from_epub(path_to_epub)
    os.makedirs("cache/" + os.path.basename(path_to_epub), exist_ok=True)

    processes = []
    for i in range(num_chapters):
        p = multiprocessing.Process(target=process_chapter, args=(path_to_epub, prompt, i))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

def choosePrompt(bulletBool, exampleBool, qnaBool):
    modifier = ""
    modifiers_dict = {"bullet":" in bulleted form", "example":" provide examples (if applicable)", "qna":" and provide a thoughtful question at the end"}
    
    if (bulletBool):
        modifier += modifiers_dict['bullet']
    elif (exampleBool):
        modifier += modifiers_dict['example']
    elif (qnaBool):
        modifier += modifiers_dict['qna']

    return f"Restructure this content into its Key Concepts. Under each Key Concept, provide a detailed explanation{modifier}. Serve the response in Markdown format, use (#) to seperate the key concepts and format the rest appropriately"

def RunBackend(course_name, bulletBool, exampleBool, qnaBool):
    generate_markdown(f"../{course_name}", choosePrompt(bulletBool, exampleBool, qnaBool))