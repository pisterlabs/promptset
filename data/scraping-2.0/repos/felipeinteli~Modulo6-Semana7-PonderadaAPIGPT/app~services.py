import logging
from openai import OpenAI
import requests
from app import models
from .api_gpt import gerar_historias



class HistoryServices:

    def create_history(prompt):
        try:
            logging.info("Criando história...Service")
            response = gerar_historias(prompt)
            logging.info("Passou função chatgpt, resposta %s", response)
            history_map = {
                'prompt': prompt['prompt'],
                'resposta': response
            }
            logging.info("History_map: %s", history_map)
            try:
                logging.info("Criando história...DAO")
                with models.HistoryDAO() as dao:
                    dao.create_history(history_map)
                return True
            except (ValueError, TypeError) as e:
                return False
            except Exception as e:
                return False
        except Exception as e:
            logging.error(f"Erro ao criar na função create_history: {e}")
            return False


    def update_history(json):
        try:
            logging.info("Atualizando história...Service")
            history_map = {
                'id': json['id'],
                'prompt': json['prompt'],
                'resposta': json['resposta']
            }
            try:
                logging.info("Atualizando história...DAO")
                with models.HistoryDAO() as dao:
                    dao.update_history(history_map)
                return True
            except (ValueError, TypeError) as e:
                return False
            except Exception as e:
                return False
        except Exception as e:
            logging.error(f"Erro ao atualizar na função update_history: {e}")
            return False
        
    def delete_history(id):
        try:
            logging.info("Deletando história...Service")
            try:
                logging.info("Deletando história...DAO")
                with models.HistoryDAO() as dao:
                    dao.delete_history(id)
                return True
            except (ValueError, TypeError) as e:
                return False
            except Exception as e:
                return False
        except Exception as e:
            logging.error(f"Erro ao deletar na função delete_history: {e}")
            return False

    def find_all(self):
        try:
            with models.HistoryDAO() as dao:
                result = dao.find_all()
            return result
        except Exception as e:
            return []
        

    

    # def get_chatgpt_response(prompt):

    #     client = OpenAI(
    #         api_key = "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"
    #     )

    #     chat_completion = client.chat.completions.create(
    #         messages=[
    #             {
    #                 "role": "user",
    #                 "content": prompt
    #             }
    #         ],
    #         model="gpt-3.5-turbo",
    #     )
    #     return chat_completion.choices[0].message.content
        

    # def chat_gpt_story(json):
    #     logging.info("Entrou função chatgpt")
    #     prompt = json['prompt']
    #     logging.info("Prompt JSON: %s", prompt)
    #     
    #     prompt = f"Conte meu uma história sobre: '{prompt}'"
    #     try:
    #         response = openai.ChatCompletion.create(
    #         model="gpt-3.5-turbo",
    #         messages=[
    #             {"role": "system", "content": "You are a helpful assistant."},
    #             {"role": "user", "content": prompt},
    #         ]
    #         )
    #         return response
    #     except Exception as e:
    #         logging.error(f"Erro ao criar história na função gpt: {e}")
    #         return False

        


