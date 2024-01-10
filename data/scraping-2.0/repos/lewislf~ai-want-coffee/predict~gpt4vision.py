import openai
import requests
import json
import base64
from api_key import OPENAI_API_KEY

def set_pre_configuration(prompt=None):
    openai.api_key = OPENAI_API_KEY

    if prompt is None:
        prompt = [
            {
                'role': 'system', 
                'content': (
                    "Você se chama Clio e é uma Inteligência Computacional Autônoma (ICA) "
                    "do laboratório de Computação de Alto Desempenho (LCAD) da Universidade "
                    "Federal do Espírito Santo (UFES). Você é uma barista muito prestativa e é responsável por instruir o processo de fazer café coado da forma"
                    "mais detalhada possível e em qualquer configuração de cozinha residencial em que esteja. Deverá me guiar "
                    "fornecendo instruções sequenciais para o preparo do café, considere que será usado café em pó,"
                    "Você deve ser capaz de guiar um usuário que nunca preparou café antes,"
                    "sempre pergunte se o usuário tem o item necessário para a tarefa e se o item é próprio para a tarefa,"
                    "só prossiga com a tarefa se o usuário confirmar que tem o item."                                    
                    "Suas instruções serão claras e diretas, não mais do que uma tarefa por vez e limite de 100 caracteres por tarefa. "
                    "Exemplos de interações:" 
                    "(EXEMPLO)'user': 'Clio, me pergunte se podemos iniciar'; 'system': 'Podemos iniciar o preparo do café?'; 'user': 'Sim';"
                    "(EXEMPLO)'system': 'Verifique se você tem um recipiente para ferver a água"
                    "(EXEMPLO)'user': 'Passo concluído.'; 'system': 'Encontre uma torneira'"
                    "(EXEMPLO)'user': 'Passo concluído.'; 'system': 'Coloque água no recipiente'"
                )
            },
            {
                'role': 'user', 
                'content': (
                    "Eu irei fazer uma demo testando através de imagens na tela do meu computador, considere-as como" 
                    "'reais' para fins de teste. Me pergunte se podemos iniciar"
                )
            },
        ]
        
    print("Configurando o modelo...")

    return prompt

# Function to encode the image
def encode_image(image_path:str):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def request_description(task:str, img_path:str, detail:str = "high"):
    prompt_image = f"Por favor, descreva os objetos na imagem relacionados à tarefa {task}, seja descritivo:"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }
    payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
            {
                "role": "user", 
                "content": [
                    {"type": "text", "text": prompt_image}
                ]
            }
        ],
        "max_tokens": 150
    }
    img = encode_image(img_path)
    img_info = {
            "type": "image_url",
            "image_url": {
            "url": f"data:image/jpeg;base64,{img}" ,
            "detail": detail
            }
    }
    payload['messages'][0]['content'].append(img_info)

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    
    if response.status_code != 200:
        print("'error': 'Failed to process the image.'")
        return
    response_content = response.content.decode('utf-8')
    description = json.loads(response_content)['choices'][0]['message']['content']
    return description

def get_response(history):
    openai.api_key = OPENAI_API_KEY
    
    if isinstance(history, str):
        history = json.loads(history)

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages = history,  # Use a lista completa de mensagens
        max_tokens=150
    )
    
    answer = response['choices'][0]['message']['content']

    # Atualize o history com a resposta da API
    history.append({'role': 'system', 'content': answer})

    return answer

def validate_task_img(description:str, task:str):
    openai.api_key = OPENAI_API_KEY
    prompt = f"""Analize task and description, if in the description has what the task want, say "yes", otherwise say "no": 
    Task: {task}
    Description: {description}
    """
    # prompt = f"""Analize a tarefa e a descrição, se na descrição tiver o que a tarefa quer, diga "yes", caso contrário diga "no":
    # Tarefa: {task}
    # Descrição: {description}
    # """

    response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=150
        )
    response = response['choices'][0]['message']['content']
    if "yes" in response.lower():
        return True
    elif "no" in response.lower():
        return False
    else:
        print(f"VALIDATION ERROR: {response}")
        
def validate_user_response(user_response:str, task:str):
    '''Verifica se o usuario concluiu a tarefa ou nao'''
    openai.api_key = OPENAI_API_KEY
    prompt = f"""Analyze the task and user response. If the user response indicates a positive affirmation of completing the task, summarize the response as 'yes'. 
    If not, summarize the response as 'no'.
    Task: {task}
    User response: {user_response}
    """
    # prompt = f"""Analise a tarefa e a resposta do usuário. Se a resposta do usuário for algo como "sim", "ok" ou indicar uma afirmação positiva de conclusão da tarefa, resuma a resposta como 'yes'.
    # Caso contrário, resuma a resposta como 'no'.
    # Tarefa: {task}
    # Resposta do usuário: {user_response}
    # """

    response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=150
        )
    response = response['choices'][0]['message']['content'].lower()   
    # print(f"ChatGPT RESPONSE: {response}") # Debug
    
    if "yes" in response:
        return ["yes",user_response]
    elif "no" in response:
        return ["no",user_response]
    else:
        print(f"VALIDATION ERROR: {response}")

def validate_if_capture_or_substitute(user_response:str, task:str):
    '''Verifica se o usuario deseja capturar outra imagem ou tentar uma tarefa substituta'''
    openai.api_key = OPENAI_API_KEY
    prompt = f"""
    Com base na resposta do usuário e na tarefa fornecida, determine a intenção do usuário.
    Se a resposta do usuário sugerir o desejo de capturar uma imagem, classifique a resposta como 'capture'.
    Se a resposta do usuário sugerir o desejo de substituir a tarefa atual por outra, classifique a resposta como 'substitute'.
    Se a intenção do usuário não estiver clara, classifique como 'unclear'.
    Tarefa: {task}
    Resposta do usuário: {user_response}
    """

    response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            max_tokens=150
        )
    response_text = response['choices'][0]['message']['content'].lower()
    # print(f"ChatGPT RESPONSE: {response_text}") # Debug
    
    if "capture" in response_text:
        return ["capture", user_response]
    elif "substitute" in response_text:
        return ["substitute", user_response]
    else:
        return ["unclear", user_response]
    
def get_equivalent_task(task):
    '''Dado uma tarefa, sugira uma tarefa equivalente que possa ser realizada.'''
    openai.api_key = OPENAI_API_KEY
    prompt = f"""Dada a tarefa, sugira uma tarefa equivalente que possa ser realizada no processo de fazer café.
    Tarefa original: {task}."""
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": prompt}
        ],
        max_tokens=150
    )
    alternative_task = response['choices'][0]['message']['content']

    return alternative_task
