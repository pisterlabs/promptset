import openai
import uuid
import datetime
from bs4 import BeautifulSoup
from app.domain.entities.chat_messages import ChatMessages
from app.domain.errors.api_exception import ApiException
from app.domain.errors.domain_errors import ChatError, ChatNotFound, HistoryOfQuestionNotFound

class ChatService:
    def __init__(self, history_question_repository, question_repository, chat_repository):
        self.history_question_repository = history_question_repository
        self.question_repository = question_repository
        self.chat_repository = chat_repository
        
    def send_message(self, user_id, question_id, message):

        chat_messages = self.chat_repository.get_by_history_question_id(question_id, user_id)
        if(chat_messages == None or len(chat_messages) <= 0):
            chat_messages = self.create_new_chat(question_id, user_id)

        user_message = ChatMessages(str(uuid.uuid4()), question_id, "user", message, datetime.datetime.utcnow(), chat_messages[-1].sequence + 1)
        chat_messages.append(user_message)
        messages = list(map(lambda x : x.to_json(), chat_messages))
        
        try:
            response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages = messages,
            temperature = 0.3
            )
            assistant_message = ChatMessages(str(uuid.uuid4()), question_id, "assistant", response.choices[0].message.content, datetime.datetime.utcnow(), chat_messages[-1].sequence + 1)
            self.chat_repository.insert_range_messages([user_message, assistant_message], user_id)
            self.chat_repository.commit()
            return response.choices[0].message.content
        except Exception as e:
            raise ApiException(ChatError())

       
        
    def create_new_chat(self,  history_question_id, user_id):
        question_id = self.history_question_repository.get_question_id(history_question_id, user_id)
        if(question_id == None):
            raise ApiException(HistoryOfQuestionNotFound())
        question = self.question_repository.get_by_id(question_id)
        messages = self.generate_initial_messages(question, history_question_id)
        self.chat_repository.insert_range_messages(messages, user_id)
        return messages

        
    def generate_initial_messages(self, question, history_question_id):
        alternatives = ""
        letter = "A"
        answer = ""
        messages = []
        statement = self.remover_tags_img(question.statement)
        for index, value in enumerate(question.alternatives):
            alternatives += f"{chr(ord(letter) + index)}){value.text}"
            if(value.id == question.answer):
                answer = f"{chr(ord(letter) + index)}){value.text}"
        chat_message_system = ChatMessages(str(uuid.uuid4()), history_question_id,  "system",  
                                    f"Você é o Vincia Bot, um professor disponível para esclarecer todas as dúvidas que o aluno possa ter em relação à seguinte questão: '{statement}', com as alternativas '{alternatives}'. A alternativa correta é a '{answer}'. Primeiramente, elabore sua própria solução para o problema. Em seguida, verifique se você chegou à alternativa correta. Caso não tenha chegado, refaça o processo até encontrar a resposta correta. Após isso, responda todas as perguntas do aluno referentes à questão.", datetime.datetime.utcnow(), 1)
        
        chat_message_assistant = ChatMessages(str(uuid.uuid4()),  history_question_id,  "assistant", "Olá, sou o Vincia Bot é irei esclarecer todas as suas dúvidas. Como posso te ajudar?",  datetime.datetime.utcnow(), 2)
        return [chat_message_system, chat_message_assistant]
    
    def remover_tags_img(self, html):
        soup = BeautifulSoup(html, 'html.parser')
        for img in soup.find_all('img'):
            img.decompose()  # Remove a tag <img> e seu conteúdo
        return str(soup)
    
