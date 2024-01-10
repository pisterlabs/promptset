import json
import openai
import requests
from Utils.file_handler import read_json,read_yaml
from Env.paths import CONFIG_PATH
from tenacity import retry, wait_random_exponential, stop_after_attempt
from InformationRetriever.TodoistInformationRetriever import TodoistInformationRetriever

class GPTAssistant():
    
    def __init__(self, assistant_functions_config_file: str):
        
        app_configs = read_yaml(CONFIG_PATH + 'app_config.yaml' )
        self.model = "gpt-3.5-turbo-1106"
        self.openai_api_key = app_configs['API_KEYS']['OPENAI_API_KEY']
        self.assistant_functions_config = read_json(assistant_functions_config_file)
        

    @retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(3))
    def __chat_completion_request(self, messages, tools=None, tool_choice=None):
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + self.openai_api_key,
        }
        json_data = {"model": self.model, "messages": messages}
        if tools is not None:
            json_data.update({"tools": tools})
        if tool_choice is not None:
            json_data.update({"tool_choice": tool_choice})
        try:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=json_data,
            )
            return response
        except Exception as e:
            return e
        
    def __get_function(self, query:str):
        messages = []
        messages.append({"role": "system", "content": "Não faça suposições sobre quais valores inserir nas funções. Peça esclarecimentos se um pedido do usuário for ambíguo"})
        messages.append({"role": "user", "content": query})

        try:
            chat_response = self.__chat_completion_request(messages, tools=self.assistant_functions_config)
            if 'error' in chat_response.json().keys():
                return chat_response.json()
            assistant_message = chat_response.json()["choices"][0]["message"]
            function_args = assistant_message['tool_calls'][0]['function']['arguments']
            function_name = assistant_message['tool_calls'][0]['function']['name']
            function = {
                'name': function_name,
                'args': function_args
            }
            return function
        except Exception as e:
            print("Unable to parse ChatCompletion response")
            print(f"Exception: {e}")
            raise e
            
    
    def __formart_list_to_string(self, list):

        if len(list)==0:
            return list
        
        if isinstance(list[0], dict):
            tmp_lst = [f'- Para o projeto \'{item["name"]}\' foram concluidos {item["count"]} tarefa(s).' for item in list]
        else:
            tmp_lst = [f'- {item}' for item in list]
        str_lst = "\n".join(tmp_lst)
        return str_lst

    def get_answer(self, query: str):
        
        metric_calculator = TodoistInformationRetriever()
        
        function = self.__get_function(query)

        if 'error' in function.keys():
            return f"ocorreu o seguinte erro ao processar sua requisação: \n\n Error Type: {function['error']['type']}\n Error Message: {function['error']['message']}"

        args = json.loads(function['args'])
        function_name = function['name']

        if function_name == 'obter_quantidade_tarefas_concluidas':
            if 'data' in args.keys():
                date = args['data'] if args['data'] != 'all' and\
                    args['data'] != 'all' and  args['data'] != 'all'\
                    else None

            amount = metric_calculator.get_completed_taks_amount(date)
            tmp_str = f' no dia {date}' if date else ''
            answer = f'Foram concluidas {amount}  tarefas' + tmp_str

        elif function_name == 'obter_quantidade_tarefas_abertas':
            amount = metric_calculator.get_opened_tasks_amount()
            answer = f'Atualmente existem {amount} tarefas abertas.'
            
        elif function_name == 'obter_media_diaria_tarefas_concluidas':
            amount = metric_calculator.get_completed_taks_avg_amount()
            answer = f'Até o presente momento você tem uma média de {amount} tarefas concluídas por dia.'
            
        elif function_name == 'obter_quais_foram_as_tarefas_concluidas':
            date = args['data']
            tasks = metric_calculator.get_completed_taks(date)
            tasks = self.__formart_list_to_string(tasks)
            answer = f'No dia {date} você concluiu as seguintes tarefas:\n\n{tasks}'
            if len(tasks) == 0:
                answer = f'Você não concluiu nenhuma tarefa no dia {date}.'
            
        elif function_name == 'obter_quantidade_tarefas_concluidas_por_projeto':
            date = None
            str_msg = 'Ao todo foram concluídas as seguintes tarefas por  projeto'
            if 'data' in args.keys():
                date = args['data']
                str_msg = f'No dia {date} foram concluídas as seguintes tarefas por projeto'
                
            tasks = metric_calculator.get_projects_completed_tasks(date)
            if tasks == []:
                
                answer = f'Não foram encontradas tarefas concluídas no dia {date}'
            else:
                tasks = self.__formart_list_to_string(tasks)
                answer = f'{str_msg}:\n\n{tasks}'
                
        elif function_name == 'obter_comentarios_tarefa':
            date = None if 'data' not in args.keys() else args['data']
            task_tile = args['titulo_tarefa']
            comments = metric_calculator.get_comments(task_tile, date)
            tmp_str = 'no dia' + date  if date else ''
            
            if comments == []:
            
                answer = f'Não foram encontrados comentários para a tarefa {task_tile} ' + tmp_str
            else:
                
                comments = self.__formart_list_to_string(comments)
                answer = f'Os comentários da tarefa {task_tile} {tmp_str} foram:\n\n{comments}'
                
        elif function_name == 'obter_comentarios_por_palavra_chave':
            print(args)
            keyword = args['palavra_chave']
            comments = metric_calculator.get_comments_by_word(keyword)
            if comments == []:
                
                answer = f'Não foram encontrados comentários com a palavra chave \'{keyword}\''
            else:
                
                comments = self.__formart_list_to_string(comments)
                answer = f'Os comentários com a palavra chave {keyword} foram:\n\n{comments}'
        
        return answer