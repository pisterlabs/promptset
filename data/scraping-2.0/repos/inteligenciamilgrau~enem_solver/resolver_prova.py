import os
import json
import re
import openai  # pip install openai
import google.generativeai as genai
import time
import datetime
import requests
import base64
import PIL.Image

openai.api_key = "SUA_API_KEY_OPENAI"
GEMINI_API = "SUA_API_KEY_GEMINI"

'''
questões de número 01 a 45, relativas à área de Linguagens, Códigos e suas Tecnologias;
questões de número 46 a 90, relativas à área de Ciências Humanas e suas Tecnologias. 
questões de número 91 a 135, relativas à área de Ciências da Natureza e suas Tecnologias;
questões de número 136 a 180, relativas à área de Matemática e suas Tecnologias;
Questões de 01 a 05 (opção inglês ou espanhol)
'''

questoes = 180
selecionar_questoes = []

gabarito_filename = "./gabaritos/gabarito_unificado_azul.json"

modelo_gpt = "gpt-3.5-turbo-1106"
#modelo_gpt = "gpt-4-1106-preview"
modelo_visao_gpt = "gpt-4-vision-preview"

modelo_gemini = "gemini-pro"
modelo_visao_gemini = "gemini-pro-vision"

modelo = modelo_gemini
modelo_visao = modelo_visao_gemini

temperatura = 1.0

instrucoes = """
    Explique passo a passo a questão e depois diga a alternativa correta 
    da questão em formato json como no exemplo { "questao 03": "B" }:"""

comecou = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')

if "gemini" in modelo:
    genai.configure(api_key=GEMINI_API)
    modelo_gemini = genai.GenerativeModel('gemini-pro')
    modelo_gemini_vision = genai.GenerativeModel('gemini-pro-vision')

# Record the start time
start_time = time.time()

falhas = 0
falhas_visao = 0
sem_resposta = 0

if selecionar_questoes == []:
    number_list = [i for i in range(1, questoes + 1)]
else:
    number_list = selecionar_questoes
    questoes = len(number_list)

print("Respondendo", len(number_list), "questões.")

def perguntar_ao_chat(messages, model, temperature):
    global falhas
    if "gemini" in model:
        try:
            # for mes in messages:
            # print("messzz", messages)
            response = modelo_gemini.generate_content(messages + " " + instrucoes)

            return response.text
        except Exception as e:
            print("Erro no Gemini", e)
            print("Feedback", response.prompt_feedback, response.prompt_feedback.block_reason,
                  response.prompt_feedback.safety_ratings)
            falhas += 1
            return "{ 'questao 00': 'Erro: " + str(e) + "' }"
    elif "gpt" in model:
        try:
            response = openai.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": instrucoes},
                          {"role": "user", "content": messages}],
                temperature=temperature,
            )

            return response.choices[0].message.content
        except Exception as e:
            print("Erro", e)
            return e


def encode_image(image_path):
    if image_path.startswith("http:"):
        return base64.b64encode(requests.get(image_path).content).decode('utf-8')
    else:
        image_path = image_path.replace("\\", "\\\\")

        directory, filename = os.path.split(image_path)
        file = rf"{directory}\{filename}"

        with open(file, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')


def perguntar_ao_chat_com_imagem(question_content_img, image_file, model_vision):
    global falhas_visao
    pergunta = """
                Com o auxilio da imagem explique passo a passo a questão e depois diga a alternativa correta 
                da questão em formato json como no exemplo {'questao 03': 'B'}.\n"""

    pergunta = pergunta + question_content_img

    if "gemini" in model_vision:
        try:
            img = PIL.Image.open(image_file)
            response = modelo_gemini_vision.generate_content([pergunta, img])
            response.resolve()
            return response.text
        except Exception as e:
            print("Erro no Gemini Vision: ", e)
            #print("Feedback", response.prompt_feedback, response.prompt_feedback.block_reason,
            #      response.prompt_feedback.safety_ratings)
            falhas_visao += 1
            return "{ 'questao 00': 'Erro: " + str(e) + "' }"
    else:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openai.api_key}"
        }

        # Getting the base64 string
        base64_image = encode_image(image_file)

        payload = {
            "model": model_vision,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": pergunta
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 4_000
        }

        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        resposta = response.json()["choices"][0]['message']['content']

        return resposta


with open(gabarito_filename, 'r', encoding='utf-8') as file:
    gabarito = json.load(file)

acertos = 0
parciais = {}

assistente_intro = '# Assistente IMG para Resolução da Prova do ENEM 2023 com ChatGPT #'

print('#' * len(assistente_intro))
print(assistente_intro)
print('#' * len(assistente_intro))
print("Modelo de Linguagem:", modelo)
print("Modelo de Visão:", modelo_visao)

# Iterate through 45 questions
# for question_number in range(1, questoes + 1):
questoes_respondidas = 0
for question_number in number_list:
    questoes_respondidas += 1

    dia = "D1" if question_number <= 90 else "D2"

    # Generate the filename
    filename = f"questoes/{dia}_questao_{question_number:02d}.txt"

    # Check if a corresponding JPG file exists
    image_filename = f"questoes/questao_{question_number:02d}.PNG"
    image_exists = os.path.exists(image_filename)

    # Read and print the content of the file
    with open(filename, 'r', encoding='utf-8') as file:
        question_content = file.read()

        print(60 * "#")
        if image_exists:
            print("Enviando Questão", question_number, "- Com Imagem")
            resp_chat = perguntar_ao_chat_com_imagem(question_content, image_filename, modelo_visao)
        else:
            print("Enviando Questão", question_number, "- Sem Imagem")
            resp_chat = perguntar_ao_chat(question_content, modelo, temperatura)

        resp = resp_chat

        # print(f"Pergunta atual: {filename}:\n{question_content}")
        print(60 * "#")
        print(f"Pergunta atual: \n{question_content}")

        print("")
        print(60 * "#")
        print("Resposta do Chat:")
        print(resp)
        print(40 * "=")

        # matches = re.findall(r'["\'](quest\S+ \d+)["\']: ["\'](.*?)["\']', resp, re.IGNORECASE | re.UNICODE)
        # matches = re.findall(r'["\']((?:questao|questão)\s*\d+)["\']: ["\'](.*?)["\']', resp,
        #                     re.IGNORECASE | re.UNICODE)
        #matches = re.findall(r'["\']((?:questao|questão)_?\s*\d+)["\']: ["\'](.*?)["\']', resp,
        #                     re.IGNORECASE | re.UNICODE)
        matches = re.findall(r'["\']((?:questao|questão)_?\s*\d+)["\']\s*:\s*["\'](.*?)["\']', resp,
                           re.IGNORECASE | re.UNICODE)

        #pattern = re.compile(r'A alternativa correta é a ([A-E])\.', re.IGNORECASE)
        #matches_fora_do_padrao = pattern.findall(resp)

        pattern = re.compile(r'lternativa correta é a ([A-E])\.|lternativa correta: ([A-E])\.|lternativa correta é ([A-E])\.|lternativa correta é a letra ([A-E])\.', re.IGNORECASE)
        matches_fora_do_padrao = pattern.findall(resp)

        # Iterate over matches
        if matches:
            for match in matches:
                question_num = match[0]
                answer = match[1]
                print(f"Resposta detectada: {answer}")
        elif matches_fora_do_padrao:
            #answer = matches_fora_do_padrao[0]
            answer = next((group for match in matches_fora_do_padrao for group in match if group), None)
            print(f"Resposta detectada: {answer}")
        else:
            answer = "Sem resposta"
            sem_resposta += 1
            print("Nula")

        print('=' * 40)

    correta = gabarito[f"questao {question_number:02d}"]
    parcial = "Questao", question_number, "Gabarito", correta, "Respondida", answer, "Sem Resposta", sem_resposta, \
        "Falhas", falhas, "Falhas Visâo", falhas_visao

    print(parcial)
    if correta == answer:
        print("!!!!! ACERTOU !!!!!")
        acertos += 1
    else:
        print(">>>>>> ERROU :(((((")

    acertos_parciais = int(acertos / questoes_respondidas * 100)

    avaliacao = "Acertos: " + str(acertos) + " De: " + str(questoes_respondidas) + \
                " questoes - Percentual de Acertos: " + str(acertos_parciais) + "%"
    print(avaliacao)

    erros_e_falhas = sem_resposta + falhas + falhas_visao
    total_de_falhas = erros_e_falhas if erros_e_falhas >= 0 else 0
    acertos_ponderados_parciais = int(acertos / (questoes_respondidas - total_de_falhas) * 100) if questoes_respondidas > total_de_falhas else 0

    avaliacao_ponderada = "Acertos Ponderados: " + str(acertos) + " De: " + \
                          str(questoes_respondidas - erros_e_falhas) + " questões - "\
                          " falhas: " + str(total_de_falhas) + " questoes - Ponderado: " + \
                          str(acertos_ponderados_parciais) + "%"

    print(avaliacao_ponderada)

    save_txt = False
    if save_txt:
        # Append text to a text file
        output_text_file = f"resolucao_{comecou}_{modelo}.txt"
        with open(output_text_file, 'a', encoding='utf-8') as text_file:
            text_file.write(f"Questao: {question_content}\n")
            text_file.write(f"Resposta: {resp_chat}\n")
            text_file.write(f"Avaliacao: {avaliacao}\n")

    parciais[f"questao {question_number:02d}"] = [{"Pergunta": question_content},
                                                  {"Resposta": resp},
                                                  {"Gabarito": correta},
                                                  {"Respondida": answer},
                                                  {"Avaliacao Parcial": [{"acertos": acertos,
                                                                          "questoes": question_number,
                                                                          "acertos ponderados": acertos_ponderados_parciais,
                                                                          "percentual de acertos": acertos_parciais}]}]

    print("")

# Record the end time
end_time = time.time()
elapsed_time = end_time - start_time

# Calculate hours, minutes, and seconds
hours, remainder = divmod(elapsed_time, 3600)
minutes, seconds = divmod(remainder, 60)

erros_e_falhas = sem_resposta + falhas + falhas_visao
total_de_falhas = erros_e_falhas if erros_e_falhas >= 0 else 0
# print("questoes", questoes, total_de_falhas, erros_e_falhas)
acertos_ponderados = int(acertos / (questoes - total_de_falhas) * 100) if questoes > total_de_falhas else 0

output_text_file = f"resolucao_{modelo}_pts_{acertos_ponderados:03d}_{comecou}_tot_{questoes}.json"

with open(output_text_file, 'a', encoding='utf-8') as text_file:
    json.dump({"avaliacao": [
        {"tempo decorrido": [{"horas": int(hours), "minutos": int(minutes), "segundos": int(seconds)}],
         "total de questoes": questoes,
         "acertos": acertos,
         "erros": questoes - acertos,
         "acertos ponderados": acertos_ponderados,
         "percentual de acertos": int(acertos / questoes * 100),
         "modelo": modelo,
         "modelo visao": modelo_visao,
         "temperatura": temperatura,
         "falhas": falhas,
         "falhas_visao": falhas_visao,
         "sem_resposta": sem_resposta,
         "nome_arquivo": output_text_file
         }],
        "prova": parciais
    },
        text_file, ensure_ascii=False, indent=4)
