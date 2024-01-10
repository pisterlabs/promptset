import os
import openai
from openai.error import OpenAIError
import backoff
import tiktoken
from sentence_transformers import SentenceTransformer
from models.Embedding import Embedding
from models.Sentence import Sentence
from services.embeddings.EmbeddingService import EmbeddingService
from daos.EmbeddingDao import EmbeddingDao
from daos.StudentDao import StudentDao

openai.api_key = os.getenv('OPENAI_API_KEY')

class PromptService:
    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.embedding_dao = EmbeddingDao()
        self.student_dao = StudentDao()
        self.model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    def search_string(self, string: str, course: str, student_id: str, n_closest: int = 8) -> list[Embedding]:
        """Get all embeddings for a course and return relevant information for the string,"""
        string_embedding = self.model.encode([string])
        closest_embeddings = self.embedding_dao.get_closest_embeddings(string_embedding, n_closest)
        closest_sources = [embedding['source'] for embedding in closest_embeddings]
        self.student_dao.add_suggestions(student_id, closest_sources)
        return closest_embeddings

        

    def ask_user_prompt(self, prompt: str, searched_embeddings: list[Embedding], template_file: str) -> str:
        """Add search results to prompt and return answer to user """
        searched_sentences = [embedding['sentence'] for embedding in searched_embeddings]
        with open(template_file, 'r', encoding="utf-8") as template_file:
            template = template_file.read()
            formatted_template = template.format(searched_sentences, prompt)
            return self.call_openai(formatted_template, 3500)

    def answer_prompt(self, prompt: str, student_id: str, template_file: str, return_diagram: bool) -> list[Sentence]:
        """search for a user query, use relevant info to ask prompt and record suggestions"""
        searched_embeddings = self.search_string(prompt, "TM101", student_id, 8)
        answer =  self.ask_user_prompt(prompt, searched_embeddings, template_file)
        answer_sentences = self.embedding_service.split_into_sentences(answer)
        output_sentences = []
        for sentence in answer_sentences:
            print("sentence returned by bot:", sentence)
            embedding = self.search_string(sentence, "TM101", student_id, 1)
            output_sentences.append(Sentence(string=sentence, source=embedding[0]['source'], isDiagram=return_diagram))
        return output_sentences

    # def answer_diagram(self, prompt: str, student_id: str) -> list[Sentence]:
    #     """Ask prompt and return answer to user """
    #     with open('services/prompts/MermaidTemplate.txt', 'r', encoding="utf-8") as template_file:
    #         template = template_file.read()
    #         formatted_template = template.format(prompt)
    #         answer = self.call_openai(formatted_template, 3500)
    #         answer_sentences = self.embedding_service.split_into_sentences(answer)
    #         output_sentences = []
    #         for sentence in answer_sentences:
    #             embedding = self.search_string(sentence, "TM101", student_id, 1)
    #             output_sentences.append(Sentence(string=sentence, source=embedding[0]['source']))
    #         return output_sentences
            

    @backoff.on_exception(backoff.expo, OpenAIError, max_tries=3)
    def call_openai(self, prompt: str, max_tokens: int) -> str:
        """Call OpenAI's API with exponential backoff."""
        # print("prompt")
        # print(prompt)
        # openai_responses = openai.ChatCompletion.create(
        #     engine="text-davinci-002",
        #     prompt=prompt,
        #     max_tokens=max_tokens
        # )
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens
        )
        # print(response)
        return response.choices[0].message.content.strip()
        # print(openai_responses.choices)
        # answer = openai_responses.choices[0].text.strip()
        # return answer