# This Recommender class is based on repo https://github.com/bhaskatripathi/pdfGPT
# Which is the main source of inspiration for this project.

import os
import tensorflow_hub as hub
import numpy as np
from utils import pdf_to_chunks, translate_text
from RecursiveSummarizer import RecursiveSummarizer
from OpenAIManager import OpenAIManager

class Recommender:
    def __init__(self, openAI_key):
        embeddings_qa_encoder = hub.load('https://tfhub.dev/google/universal-sentence-encoder/3')
        self.embeddings_q_encoder  = embeddings_qa_encoder.signatures['question_encoder']
        self.embeddings_a_encoder  = embeddings_qa_encoder.signatures['response_encoder']

        self.open_ai_manager = OpenAIManager(openAI_key)
        self.recursive_summarizer = RecursiveSummarizer(openAI_key)
    
    def get_answer_embeddings(self, chunks, batch=1000):
        embeddings = []
        for i in range(0, len(chunks), batch):
            chunks_batch = chunks[i:(i+batch)]
            emb_batch = self.embeddings_a_encoder(chunks_batch)
            embeddings.append(emb_batch)
        embeddings = np.vstack(embeddings)
        return embeddings

    def save_embeddings_for_chunks(self, chunks, batch=1000, n_neighbors=5):
        self.chunks = chunks
        self.embeddings = self.get_answer_embeddings(chunks, batch=batch)
        n_neighbors = min(n_neighbors, len(self.embeddings))
        self.nn = NearestNeighbors(n_neighbors=n_neighbors)
        self.nn.fit(self.embeddings)
    

    def search(self, text):
        inp_emb = self.embeddings_q_encoder([text])
        neighbors = self.nn.kneighbors(inp_emb, return_distance=False)[0]
        return [self.chunks[i] for i in neighbors]


    def load(self, path):
      pdf_file = os.path.basename(path)
      embeddings_file = f"{pdf_file}.npy"
      
      if os.path.isfile(embeddings_file):
          embeddings = np.load(embeddings_file)
          self.embeddings = embeddings
          return "Embeddings loaded from file"
      
      chunks = pdf_to_chunks(path, self.recursive_summarizer)
      self.save_embeddings_for_chunks(chunks)
      np.save(embeddings_file, self.embeddings)
      return 'Corpus Loaded.'


    def question_answer(self, file, question):
        if file == None:
            return '[ERROR]: PDF is empty. Provide one.'

        old_file_name = file.name
        file_name = file.name
        file_name = file_name[:-12] + file_name[-4:]
        os.rename(old_file_name, file_name)
        self.load(file_name)

        if question.strip() == '':
            return '[ERROR]: Question field is empty'

        return self.generate_answer(question)


    def get_additional_answers(self, prompt):
        prompt_for_additional_questions = (
            prompt
            + "Instructions: You are given a query and a set of search results. "
            + "It must be enough to compose a comprehensive answer to the query without any "
            + "additional knowledge using search results. Your goal is to provide 1 to 5 additional questions "
            + "to be answered in order to make search results full enough to compose answer to query. " + 
            + "Feel free to ask about language syntax, additional conditions, theory, ask for examples, etc. " + 
            + "If you provide additional questions make them short and concise. Start each question with \"Q:\"." 
        ) 
        answer = self.open_ai_manager.get_answer(prompt_for_additional_questions)
        if ('Q:' not in answer):
            return ''
        additional_questions = answer.split('Q:')[1:]
        prompt += "\nAdditional Help Questions:\n"
        for additional_question in additional_questions:
            topn_chunks = self.search(additional_question)
            top_chunk   = topn_chunks[0]
            prompt += f"Q: {additional_question}\nA: {top_chunk}\n\n"
        return prompt


    def generate_answer(self, question):
        question = translate_text(question)
        topn_chunks = self.search(question)
        prompt = ""
        prompt += 'search results:\n\n'
        for c in topn_chunks:
            prompt += c + '\n\n'

        prompt = self.get_additional_answers(prompt + f"Query: {question}\n")
            
        prompt += "Instructions: Compose a comprehensive reply to the query using the search results given. "\
                  "Cite each reference using [Page Number] notation (every result has number). "\
                  "Citation should be done at the end of each sentence. If the search results mention multiple subjects "\
                  "with the same name, create separate answers for each.  Make sure the answer is correct and don't output false. "\
                  "Ignore results which has nothing to do with the question. Only answer what is asked. "\
                  "Provide additional explanations or examples if asked by user. "\
                  "The answer should be short and concise. Answer step-by-step. \n\nQuery: {question}\nAnswer: "
        
        print(prompt)

        answer = self.open_ai_manager.get_answer(prompt)
        return answer