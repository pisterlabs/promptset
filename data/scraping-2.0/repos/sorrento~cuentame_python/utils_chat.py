#  Los modelos disponibles son: (https://openai.com/pricing)
# https://platform.openai.com/docs/models/gpt-3-5
from conf.secrets_key import OPENAI_API_KEY
import yaml
import openai

openai.api_key = OPENAI_API_KEY

GPT_MODEL_35_4k_06 = "gpt-3.5-turbo-0613"  # $0.0015 / 1K tokens	$0.002 / 1K tokens hasta juno 23
GPT_MODEL_35_4k = "gpt-3.5-turbo"
GPT_MODEL_35_INSTRUCT = "gpt-3.5-turbo-instruct"  # $0.0015 / 1K tokens	$0.002 / 1K tokens hasta juno 23
GPT_MODEL_35_16k = 'gpt-3.5-turbo-16k'  # $0.003 / 1K tokens	$0.004 / 1K tokens
GPT_MODEL_4_8k = "gpt-4"  # $0.03 / 1K tokens	$0.06 / 1K tokens
GPT_MODEL_4_32k = "gpt-4-32k"  # $0.06 / 1K tokens	$0.12 / 1K tokens
EMBEDDING_MODEL = "text-embedding-ada-002"

FILE_PROMPTS = 'conf/prompts.yml'
# si agregamos -0613 tenemos el modelo actualizado hasta junio 13


############ REPLIS ##############################

class Repli:
    def __init__(self, repli_name, silent=False):
        self.messages = None  # lista de mensajes de la conversación
        self.repli_name = repli_name  # es el que aparce en FILE_PROMPTS

        data_all = self.load_prompts()

        if repli_name not in data_all:
            raise Exception(
                f'El nombre de repli {repli_name} no está en el fichero {FILE_PROMPTS}. Los que hay son: {data_all.keys()}')

        data = data_all[repli_name]
        if not silent:
            print(f'** Utilizando el repli {repli_name}**')
            print(f'\n**params:   {data["model"]} - max_tokens: {data["max_tokens"]} - t:{data["temperature"]}\n')

            if 'description' in data:
                print(f'   ** {data["description"]}**')
            else:
                print(f'   ** No hay descripción**')

            print(f'** prompt:\n {data["prompt"]}')

        if 'sample_qs' in data:
            qs = data['sample_qs']
            i = 0
            if not silent:
                print(f'** Hay {len(qs)} ejemplos de preguntas**')
                for f in qs:
                    print(f' [{i}] {f[:100]}')
                    i += 1
            self.samples = qs
        else:
            if not silent:
                print(f'** No hay ejemplos de preguntas**')
            self.samples = []

        if 'functions' in data:
            if not silent:
                print(f'** Hay {len(data["functions"])} funciones**')
                i = 0
                for f in data["functions"]:
                    print(f' [{i}] {f["name"]} - {f["description"]}')
                    i += 1
            self.functions = data["functions"]
        else:
            print(f'** No hay funciones**')
            self.functions = []

        self.data = data

    def load_prompts(self):
        # comprobamos si el archivo existe
        import os
        if not os.path.exists(FILE_PROMPTS):

            formato = """repli_name:
    prompt: # prompt del repli
    model: # modelo de openai
    temperature: # temperatura
    n: # número de respuestas
    max_tokens: # máximo de tokens    
    request_timeout: # timeout de openai
    description: # descripción del repli
    functions: # lista de funciones
        - name:
        description:
        parameters:
            type: object
            properties:
            required: []
    module: # módulo donde están las funciones
    sample_qs: # lista de ejemplos de preguntas
        - pregunta1
        - pregunta2

        """

            raise Exception(f'El archivo {FILE_PROMPTS} no existe. Debe tener el siguiente formato:\n{formato}')

        with open(FILE_PROMPTS, 'r', encoding='utf8') as f:
            data_all = yaml.load(f, Loader=yaml.FullLoader)
        return data_all

        # self.model = model
        # self.nombre = nombre
        # self.prompt = prompt
        # self.max_tokens = max_tokens
        # self.t = t

        # self.voces = {}  # conciencia y otras voces auxiliares
        # self.messages = [
        #     {'role': 'system', 'content': prompt}
        # ]

    def get_sample(self, index):
        if self.samples is None:
            print('No hay ejemplos de preguntas')
            return None
        return self.samples[index]

    def _get_functions(self):
        return self.data['functions']

    def responde(self, p):
        #    Si mensajes está vacío, lo inicializamos con el prompt
        if self.messages is None:
            self.messages = [{'role': 'system', 'content': self.data['prompt']}]

        self.messages.append({'role': 'user', 'content': p})

        msgs = self.messages.copy()

        params = {
            "model":       self.data['model'],
            "max_tokens":  self.data['max_tokens'],
            "temperature": self.data['temperature'],
            "messages":    msgs,
        }

        # Diccionario de valores por defecto
        default_values = {'n': 1, 'request_timeout': 30}

        # Actualizar params con las claves opcionales presentes en self.data
        for key, default_value in default_values.items():
            params[key] = self.data.get(key, default_value)

        # si 'functions' está en data, lo añadimos a params
        if 'functions' in self.data:
            params['functions'] = self.data['functions']
            module_name = self.data['module']
        else:
            module_name = None
        try:
            response = openai.ChatCompletion.create(**params)
        except Exception as e:
            # verificamos si el error es por tamaño de prompt
            if "This model's maximum context length is 4097" in str(e):
                print('** Error: el tamaño del prompt es demasiado grande probaremos con modelo 16k')
                params['model'] = GPT_MODEL_35_16k
                response = openai.ChatCompletion.create(**params)
            else:
                print(f'** Error al ejecutar openai.ChatCompletion.create: {e}')
                return None

        respuesta = process_ans(msgs, response, module_name)

        # si la respuesta es de tipo texto, agregamos la primera a la conversación
        if respuesta['tipo_respuesta'] == 'texto':
            self.messages.append({'role': 'assistant', 'content': respuesta['ans']})

        return respuesta

    def get_prompt(self):
        return self.data['prompt']

    def set_prompt_parameters(self, parameters):
        prompt = self.data['prompt']
        prompt_updated = prompt.format(**parameters)
        self.data['prompt'] = prompt_updated
        # verificamos si han quedado llaves sin reemplazar
        if '{' in prompt_updated:
            print(f'** Todavía hay llaves sin reemplazar en el prompt: {prompt_updated}')
            # mostramos los parámetros no reemplazados
            import re
            pattern = r'{([^}]+)}'
            matches = re.findall(pattern, prompt_updated)
            print(f'** Parámetros no reemplazados: {matches}. Revisa el diccionario de parámetros y agrégalo')

    def remove_last_message(self):
        msg = self.messages.pop()
        print(f'Eliminando el último mensaje: {msg}')

    def _responde_old(self, texto, t=None, max_tokens=None, model=None):
        import openai
        openai.api_key = OPENAI_API_KEY

        self.messages.append({'role': 'user', 'content': texto})

        if t is None:
            t = self.t
        if max_tokens is None:
            max_tokens = self.max_tokens
        if model is None:
            model = self.model

        response = openai.ChatCompletion.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=self.messages,
            temperature=t,
            timeout=35
        )
        res = response.choices[0].message.content
        self.messages.append({'role': 'assistant', 'content': res})

        self.html_autoload()

        return res

    def html_autoload(self):
        """
        Crea un archivo HTML con el contenido actualizado y un script para que se recargue cada 5 segundos
        :return:
        """
        body = self.get_html_body()

        contenido_js = """
            <script>
                setTimeout(function(){
                    location.reload();
                }, 5000);
            </script>
            """

        # Crear el contenido HTML completo
        contenido_html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Hora actual</title>
            </head>
            <body>
                {body}
                {contenido_js}
            </body>
            </html>
            """

        # Guardar el contenido en un archivo HTML
        nombre_archivo = "chat.html"
        print(f"Guardando el archivo '{nombre_archivo}'... html  auto reload")
        with open(nombre_archivo, "w") as archivo:
            archivo.write(contenido_html)

        print(f"Se ha actualizado el archivo '{nombre_archivo}' con la hora actual.")

    def get_md(self):
        return crea_md(self.messages)

    def get_html_body(self):
        from utils import md2html
        md = self.get_md()
        return md2html(md)

    def exporta_mensajes(self, path):
        import json
        with open(path, 'w') as fp:
            json.dump(self.messages, fp)


def responde(name, p, messages=None, reemplazos=None, silent=False, overwrite={}):
    """
    Utiliza un repli para responder a una pregunta
    name: nombre del prompt
    p: pregunta
    messages: lista de mensajes previos
    reemplazos: diccionario con los reemplazos para el prompt
    silent: si es True no imprime el prompt
    overwrite: diccionario con los parámetros que se quieren sobreescribir
    returns: respuesta, lista de mensajes, lista de respuestas
    """
    # overwrite es para sobreescribir parámetros del yaml
    msgs = messages.copy() if messages is not None else None

    c = yaml.safe_load(open(FILE_PROMPTS, encoding='utf-8'))
    params = c[name]

    if 'description' in params:
        print(f'*****  Usando {name}: {params["description"]}')
    else:
        print(f'*****  Usando {name}. NO TIENE DESCRIPCIÓN')

    if overwrite != {}:
        # sobreescribimos los parámetros que están en overwrite
        for k in overwrite:
            params[k] = overwrite[k]

    if 'n' not in params:
        n = 1  # número de respuestas
    else:
        n = params['n']

    if 'timeout' not in params:
        params['timeout'] = 30

    # si messages es un string, creamos los mensajes (es la primmera pregunta)
    if msgs is None:
        prompt = params['prompt']
        if reemplazos is not None:
            prompt = prompt.format(**reemplazos)

        if not silent:
            print(prompt + '\n-------------------\n')

        msgs = [{'role': 'assistant', 'content': prompt}]

    msgs.append({'role': 'user', 'content': p})

    full_parameters = {
        'model':           params['model'],
        'max_tokens':      params['max_tokens'],
        'temperature':     params['temperature'],
        'n':               n,
        'messages':        msgs,
        'request_timeout': params['timeout']
    }
    if 'functions' in params:
        print(f'Hay {len(params["functions"])} funciones')
        full_parameters['functions'] = params['functions']
        module_name = params['module']
        print(f'Importando el módulo {module_name}')
    else:
        module_name = None

    response = openai.ChatCompletion.create(**full_parameters)

    return process_ans(msgs, response, module_name)


def responde_interactivo(name, messages=None, reemplazos=None, silent=False, overwrite={}):
    c = yaml.safe_load(open('data_in/prompts.yml', encoding='utf-8'))
    params = c[name]
    if overwrite != {}:
        # sobreescribimos los parámetros que están en overwrite
        for k in overwrite:
            params[k] = overwrite[k]

    # si messages es un string, creamos los mensajes (es la primmera pregunta)
    if messages is None:
        prompt = params['prompt']
        if reemplazos is not None:
            prompt = prompt.format(**reemplazos)

        if not silent:
            print(prompt + '\n-------------------\n')

        messages = [{'role': 'assistant', 'content': prompt}]

    while True:
        input_text = input("User: ")
        messages.append({"role": "user", "content": input_text})
        response = openai.ChatCompletion.create(
            model=params['model'],
            max_tokens=params['max_tokens'],
            temperature=params['temperature'],
            messages=messages, timeout=25
        )
        messages.append({"role": "assistant", "content": response.choices[0].message.content})
        print("User: " + input_text)
        print("Daryl: " + response.choices[0].message.content)
        print('     ------------------ Other options: ------------------')
        for choice in response.choices[1:]:
            print('      ' + choice.message.content)
            print('               ------------------')


def replis_lista():
    import yaml
    with open('data_in/prompts.yml', 'r', encoding='utf-8') as f:
        prompts = yaml.safe_load(f)
    keys = list(prompts.keys())
    return {p: prompts[p].get('description', '') for p in prompts.keys()}

############# PRINCIPALES ############################


def haz(prompt, max_tokens=400, temp=0.3):
    # usamos el instruct model
    try:
        response = openai.Completion.create(
            model=GPT_MODEL_35_INSTRUCT,
            prompt=prompt,
            max_tokens=400,
            temperature=0.3,
        )
        res = response.choices[0].text
    except Exception as e:
        print(f'Error al ejecutar openai.Completion.create: {e}. Usaremos un modelo de 16k')
        try:
            response = openai.ChatCompletion.create(
                model=GPT_MODEL_35_16k,
                messages=[{'role': 'system', 'content': prompt},
                          {'role': 'user', 'content': 'Hazlo'}],
                max_tokens=400,
                temperature=0.3,
            )

        except Exception as e:
            # lo hacemos de nuevo pero solo con los 50k caracteres iniciales
            print(f'Cortaremos el prompt a 50k caracteres')
            prompt = prompt[:50000]
            response = openai.ChatCompletion.create(
                model=GPT_MODEL_35_16k,
                messages=[{'role': 'system', 'content': prompt},
                          {'role': 'user', 'content': 'Hazlo'}],
                max_tokens=max_tokens,
                temperature=temp
            )
        res = response.choices[0].message.content
    return res


def token_count(text):
    import tiktoken

    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    num_tokens = len(encoding.encode(text))
    return num_tokens


def token_price_calculator(model, n_tokens):
    """
    Calcula el precio de una respuesta en función del modelo y el número de tokens
    :param model: modelo
    :param n_tokens: número de tokens
    :return: precio en dólares
    """
    if model == GPT_MODEL_35_4k:
        p_1k = 0.0015
    elif model == GPT_MODEL_35_4k_06:
        p_1k = 0.015
    elif model == GPT_MODEL_35_16k:
        p_1k = 0.003
    elif model == GPT_MODEL_4_8k:
        p_1k = 0.03
    elif model == GPT_MODEL_4_32k:
        p_1k = 0.06
    else:
        raise Exception(f'El modelo {model} no está soportado')

    res = p_1k * n_tokens / 1000
    print(f'El precio por 1000 tokens es {p_1k}, por lo que el precio por {n_tokens} tokens es {res} dólares')

    return res


############## CONCURRENTES ############################

async def query_openai_api(prompt):
    import aiohttp
    url = "https://api.openai.com/v1/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }

    data = {
        "model": "gpt-3.5-turbo-instruct",
        "prompt": prompt,
        "max_tokens": 300,
        "temperature": 0.2
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=data, headers=headers) as response:
            if response.status == 200:
                print(f'tenemos la respuesta de... {prompt[-20:]}')
                return await response.json()
            else:
                print(f'Error al ejecutar openai.ChatCompletion.create: {response.status}')
                print(f'prompt: {prompt}')
                response.raise_for_status()

async def multiprompt(prompts):
    import asyncio
    # Un array con tus prompts que quieres enviar a la API
    # prompts = ["cuentame un cuento sobre barcos"]
            #    "cuentame un cuento sobre aviones",
            #    "cuentame un cuento sobre trenes",
            #    "cuentame un cuento sobre coches"]

    # Crea tareas para todas tus solicitudes API
    tasks = [query_openai_api(prompt) for prompt in prompts]

    # Ejecuta todas las tareas de manera concurrente
    responses = await asyncio.gather(*tasks)

    # Maneja las respuestas aquí (por ejemplo, imprimir la respuesta)
    # for i, response in enumerate(responses):
    #     print(f"Response {i + 1}: {response}")

    return responses
############## FUNCIONES AUXILIARES ############################


def crea_md(messages):
    # escribimos toda la conversación como un markdown donde lo que dice el user está en negrita, y lo del asistente en cursiva. Luego una línea en blanco
    str_md = ''
    for m in messages:
        content = m['content']
        if m['role'] == 'user':
            str_md += f'\n**{content}**\n\n'
        elif m['role'] == 'assistant':
            # quitamos los saltos de línea
            # content=content.replace('\n','')
            str_md += f'<em>{content}</em>\n\n\n'

    return str_md


def ejecuta_respuesta(d_fun, nombre_modulo):
    """
    Ejecuta una función con los parámetros que vienen en el diccionario d_fun, provenientes de la respuesta de openai
    :param d_fun: diccionario con los parámetros de la función y el nombre de la función
    :param repli_name:
    :return:
    """
    import importlib
    import json
    fun_name = d_fun['name']
    d_params = json.loads(d_fun['arguments'])

    try:
        modulo = importlib.import_module(nombre_modulo)
    except Exception as e:
        print(f'Error al importar el módulo {nombre_modulo}: {e}')
        return None
    # imprimimos los nombres de las funciones del módulo
    if fun_name in dir(modulo):
        print(f'Ejecutando {d_fun["name"]} con parámetros {d_params}')
        # "funcion" es el objeto función. ¿se puede ejecutar directamente? creo que no
        funcion = vars(modulo).get(fun_name)
    else:
        funcion = None
        print(
            f'error, no existe la función {d_fun["name"]} en {nombre_modulo}, asegúrate de que se ha cargado la librería')
        print(f'Las funciones disponibles son: {dir(modulo)}')

    # locals()[fun_name] = funcion #

    return funcion(**d_params)


def process_ans(msgs, response, module_name=None):
    """
    Procesa la respuesta de openai
    :param msgs: lista de mensajes
    :param name: nombre del prompt
    :param response: respuesta de openai
    :return: un diccionario con la respuesta con las posibles claves: a) si es tipo texto: ans, anss, b) si es tipo función: fun, res_fun
    En donde res_fun es la respuesta de la función
    """

    message = response.choices[0].message
    respuesta = {}
    if 'function_call' in message:
        tipo_respuesta = 'función'
        fun = message['function_call']
        respuesta['fun'] = fun
        print(f' Identificada función: {fun["name"]}\n-------------------\n')
        print(f' Argumentos:\n{fun["arguments"]}')

        res_fun = ejecuta_respuesta(fun, module_name)
        respuesta['res_fun'] = res_fun

    else:
        tipo_respuesta = 'texto'
        ans = message.content
        print(f'tokens de la respuesta: {token_count(ans)}\n-------------------\n')
        msgs.append({'role': 'assistant', 'content': ans})
        anss = [c.message.content for c in response.choices]
        for i, a in enumerate(anss):
            print(f'\n*************** ans {i}:\n {a}')
        respuesta['ans'] = ans
        respuesta['anss'] = anss
        respuesta['messages'] = msgs
    respuesta['tipo_respuesta'] = tipo_respuesta
    print('-------------------')

    return respuesta


def get_samples(repli):
    """"
    Obtiene los ejemplos de un repli que están en el propmts.yml

    """
    import yaml
    with open(FILE_PROMPTS, encoding='utf-8') as f:
        prompts = yaml.safe_load(f)
    if 'sample_qs' in prompts[repli]:
        r = prompts[repli]['sample_qs']
    else:
        r = None
        print(f'No hay ejemplos para {repli}')
    return r


def samples2md(repli, txt_total, reemplazos=None, titulo='Auscham', path_out='data_out/10_Mateo/Auscham/'):
    samples = get_samples(repli)

    respuestas = []
    i = 0
    for s in samples:
        i += 1
        print(f'Pregunta {i} de {len(samples)}')
        r = responde(repli, p=s, reemplazos={"presentaciones": txt_total})
        respuestas.append(r)

    # creamos un md con las preguntas y las respuestas
    from utils import md2htmlpdf
    txt = ''
    txt += f'# {titulo}\n\n'
    for i in range(len(samples)):
        txt += f'\n\n## {i + 1}: {samples[i]}\n\n'
        txt += f'## Respuesta\n\n'
        txt += f'{respuestas[i]["ans"]}\n\n'

    md2htmlpdf(txt, 'ejemplos', path_out=path_out)


def get_p_samples(repli, silent=False):
    """"
    Obtiene los ejemplos de un repli que están en el propmts.yml

    """
    import yaml
    with open(FILE_PROMPTS, 'r', encoding='utf-8') as f:
        prompts = yaml.load(f, Loader=yaml.FullLoader)
    if 'sample_qs' in prompts[repli]:
        qs = prompts[repli]['sample_qs']
        print(f'Hay {len(qs)} ejemplos para {repli}')
        if not silent:
            i = 0
            for q in qs:
                print(f' [{i}] {q[:100]}')
                i += 1
    else:
        qs = None
        print(f'No hay ejemplos para {repli}')
    return qs
######################## ANTIGUOS###############################################


def ans2dict(ans):
    #  en el string 'ans' obtnemos lo que está dentro de { } más externoy lo guardamos en txt
    # Encontrar el inicio y fin de las llaves más externas
    start_index = ans.find("{")
    end_index = ans.rfind("}") + 1

    # Extraer el contenido entre las llaves
    txt = ans[start_index:end_index]

    # lo convertimos en diccionario
    import json
    try:
        d = json.loads(txt)

    except Exception as e:

        print(f'** Error al convertir la respuesta en diccionario {e}')
        print(f'Lo de dentro de las llaves es: {txt}')
        d = None
    return d
