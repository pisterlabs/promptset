import openai
import geopy
import timezonefinder
import requests
import googletrans
import smtplib
import email.message
import intencoes_entidades_lookups
import credenciais
import json
from UncSoftwareAPI import unc_funcs as unc_funcs
from PostgreeSql import connect as banco
import autopep8
import variaveis

def gerar_resposta(messages):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", 
        messages=messages,
        max_tokens=100,
        temperature=0.5
    )
    return [response.choices[0].message.content, response.usage]

def separar_intencao_entidade(comando,intent_id,enunciados,entidades,ent_func_id,func_fonte):
    intencao = ''
    entidade = ''
    fonte_resposta = ''

    print(comando)

    enunciados = enunciados.split(',')
    entidades = entidades.split(',')
    
    possiveisIntecoes = []
    contIntencoes = 0

    print(enunciados)
    for token in enunciados:
        contIntencoes += 1
        print('token:' +str(token))
        if token.lower() in comando.lower():
            print("caiu enunciado")
            intencao += token + " "
            objIntencao = [contIntencoes,intencao]
            possiveisIntecoes.append(objIntencao)
            
    # Encontrar o elemento com o maior número na posição 0
    if len(objIntencao) > 0:
        intencao_decidida = max(possiveisIntecoes, key=lambda x: x[0])  
        print(intencao_decidida)
    if intencao_decidida != None: #Só passa para entidades se tiver intenção descoberta
        print("ent_func_id? "+str(ent_func_id)) #tem função para entidades?
        if ent_func_id == None: #Se não tiver, utiliza as inseridas manualmente
            for token in entidades:
                if token.lower() in comando.lower():
                    entidade += token + " "
        else: #Chama a função vinculada
            print("entrou ent_func")
            cursor = banco.conn.cursor()

            selectFuncoes = "SELECT func_fonte FROM funcao WHERE func_id = "+str(ent_func_id)
            print("SELECT = "+selectFuncoes)
            cursor.execute(selectFuncoes)
            ent_func_fonte = cursor.fetchall()
            print(ent_func_fonte[0][0])
            cursor.close()
            
            #Criando contexto para passar variaveis para o código a ser executado e vice versa
            contexto = {}

            #Identar código
            #codigo_identado = autopep8.fix_code(ent_func_fonte)
            #print(codigo_identado)

            #Exec e resposta do código passando o contexto
            #exec(codigo_identado,globals(),contexto)
            #resp = contexto.get("resposta")
            
            '''for token in range(len(resp[:][1])):
                print("token: "+str(resp[token][1]))
                if resp[token][1] in comando:
                    entidade += resp[token][0]
                    print("entidade encontrada")
                    break'''
            #tarefa no trello para pensar em como solucionar isso
        
        if func_fonte != "":
            fonte_resposta = func_fonte
        else:
            cursor = banco.conn.cursor()

            #selectRespostaSimples = "SELECT resp_simples FROM intecao_entidade WHERE intent_id = "+intent_id
            #cursor.execute(selectRespostaSimples)
            #resp_simples = cursor.fetchall()
            cursor.close()
                
    return intencao_decidida[1],entidade,fonte_resposta

#separar_intencao_entidade("qual o estoque do item BOBINA GALVALUME 0,50 X 1200 AZ150",["estoque,a"],["nometabela,a"],["sim"])

def varreMatrizRelacoes(command):
    intencao = ''
    entidade = ''
    #print(intencoes_entidades_lookups.tabela_relacao_intencao_entidades)
    intencoes_entidades_lookups.atualizarTabelas()
    json_results = []
    for row in intencoes_entidades_lookups.tabela_relacao_intencao_entidades:
        print(row)
        json_row = {
            "intent_id": row[0],
            "enunciados": row[1],
            "entidades": row[2],
            "ent_func_id": row[3],
            "func_fonte": row[4],
            # Adicione mais colunas conforme necessário
        }
        json_results.append(json_row)
    #print(json_results)
    for i in range(len(json_results)):
        #print("i: "+str(i)+" - "+str(len(json_results)))
        intencao,entidade,fonte_resposta = separar_intencao_entidade(command,json_results[i]['intent_id'],json_results[i]['enunciados'],json_results[i]['entidades'],json_results[i]['ent_func_id'],json_results[i]['func_fonte'])
        if intencao != '':
            break
    return intencao,entidade,fonte_resposta


def definir_fuso_horario_por_cidade(nome_cidade):
    geolocator = geopy.geocoders.Nominatim(user_agent="timezone_app")
    location = geolocator.geocode(nome_cidade, exactly_one=True)
    
    if location is None:
        return None
    
    latitude, longitude = location.latitude, location.longitude
    tf = timezonefinder.TimezoneFinder()
    tz_target = tf.timezone_at(lng=longitude, lat=latitude)
    
    return tz_target


def consultar_cotacoes(moeda):
    url = "https://economia.awesomeapi.com.br/last/"+moeda+"-BRL"
    response = requests.get(url)
    data = response.json()

    valor = data[moeda+'BRL']['high']
    #print(str(valor))
    return str(valor)


def consultar_noticias(periodo):
    translator = googletrans.Translator()
    
    url = "https://api.nytimes.com/svc/mostpopular/v2/viewed/"+str(periodo)+".json?api-key="+credenciais.APIKEY_NWTIMES
    print(url)
    response = requests.get(url)
    data = response.json()
    quantidade_noticias = len(data['results'])

    noticias = []
    for i in range(quantidade_noticias):
        titulo = data['results'][i]['title']
        titulo_traduzido = translator.translate(titulo,dest='pt')
        resumo = data['results'][i]['abstract']
        resumo_traduzido = translator.translate(resumo,dest='pt')
        novaNoticia = [titulo_traduzido.text,resumo_traduzido.text]
        #print(novaNoticia)
        noticias.append(novaNoticia)
    
    print('Noticias encontradas no período de '+str(periodo)+' dias: '+str(quantidade_noticias))
    return noticias

def enviar_email(emaildestino,assunto,corpo):
    email_address = 'pedrocarrijo95@gmail.com'
    email_password = 'rvjtpakdwqaywlrp'

    msg = email.message.EmailMessage()
    msg['From'] = email_address
    msg['To'] = emaildestino
    msg['subject'] = assunto
    msg.set_content(corpo)

    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
        smtp.login(email_address, email_password)
        smtp.send_message(msg)

def substituir_strings(string_principal, array_strings, substituto):
    for palavra in array_strings:
        string_principal = string_principal.replace(palavra, substituto)
    return string_principal
        
        
def tratarEmail(resposta_emaildestino):
    resposta_emaildestino = resposta_emaildestino.replace(" ", "")
    resposta_emaildestino = resposta_emaildestino.replace("ponto", ".")
    resposta_emaildestino = resposta_emaildestino.replace("underline", ".")
    resposta_emaildestino = resposta_emaildestino.replace("traço", "-")
    resposta_emaildestino = resposta_emaildestino.replace("hífen", "-")
    resposta_emaildestino = resposta_emaildestino.replace("arroba", "@")
    return resposta_emaildestino


def chamarAPI(endpoint,headers,body,method,codigo_resposta,entidade):
    # Faz a requisição - json.dumps
    response = requests.request(method, endpoint, headers=json.loads(headers), data=body)

    # Processa a resposta baseado no código configurado
    contexto = {'response': response,'entidade': entidade}
    print(codigo_resposta)
    exec(codigo_resposta,globals(),contexto)
    return contexto.get("resposta")

#chamarAPI('https://servicodados.ibge.gov.br/api/v1/localidades/distritos','{"Content-Type": "application/json"}','{}','GET','for i in range(5):\n resposta = "Olá os distritos B são os seguintes: ";\n resposta += "->"+str(response.json()[i]["nome"]);','')

def run_assistente(comando):

    #comando =  request.args.get('comando')
    #print(comando)

    intencao,entidade,fonte_resposta = varreMatrizRelacoes(comando)
    print('Inteção detectada: '+intencao)
    print('Entidade detectada: '+entidade)
    print('Fonte Resposta detectada: '+fonte_resposta)
    
    #resposta = ''
    #Detecta se o nome do assistente foi dito (default = true)
    if not variaveis.ASSISTENTE_DESCANSAR or variaveis.PALAVRA_ASSISTENTE:
        #Criando contexto para passar variaveis para o código a ser executado
        contexto = {'entidade': entidade}

        #Identar código
        codigo_identado = autopep8.fix_code(fonte_resposta)
        print("Identado: "+str(codigo_identado))

        #Exec e resposta do código passando o contexto
        exec(codigo_identado,globals(),contexto)
        resp = str(contexto.get("resposta"))

        #Coleta da resposta
        if resp != None:
            return resp
        return 'reposta = resposta não encontrada'
        #exemplo codigo = "resposta = unc.getEstoque('0000000001')" 

def getJsonIntencoes():
    lista_de_dicionarios = []

    for row in intencoes_entidades_lookups.tabela_intencoes_enunciados:
        dicionario = {
            "id": row[0],
            "int_descricao": row[1].split(','),
            "int_enunciado": row[2].split(',')
        }
        lista_de_dicionarios.append(dicionario)
    return lista_de_dicionarios

def getJsonEntidades():
    lista_de_dicionarios = []

    for row in intencoes_entidades_lookups.tabela_entidades:
        dicionario = {
            "ent_id": row[0],
            "ent_descricao": row[1],
            "ent_entidade": row[2].split(','),
            "func_id": row[3]
        }
        lista_de_dicionarios.append(dicionario)
    return lista_de_dicionarios

def getJsonFuncoes():
    lista_de_dicionarios = []

    for row in intencoes_entidades_lookups.tabela_funcoes:
        dicionario = {
            "func_id": row[0],
            "func_nome": row[1],
            "func_fonte": row[2],
            "func_tipo": row[3],
        }
        lista_de_dicionarios.append(dicionario)
    return lista_de_dicionarios

def getJsonRelacoes():
    lista_de_dicionarios = []

    for row in intencoes_entidades_lookups.tabela_relacao_intencao_entidades:
        dicionario = {
            "intent_id": row[0],
            "enunciados": row[1].split(','),
            "entidades": row[2].split(','),
            "ent_func_id": row[3],
            "func_fonte": row[4],
        }
        lista_de_dicionarios.append(dicionario)
    return lista_de_dicionarios

def getJsonAPIs():
    lista_de_dicionarios = []

    for row in intencoes_entidades_lookups.tabela_apis:
        dicionario = {
            "api_id": row[0],
            "api_json": row[1],
        }
        lista_de_dicionarios.append(dicionario)
    return lista_de_dicionarios