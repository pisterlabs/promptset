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

def gerar_resposta(messages):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", 
        #model="gpt-3.5-turbo-0301", ##até 1 junho 2023
        messages=messages,
        max_tokens=100,
        temperature=0.5
    )
    return [response.choices[0].message.content, response.usage]
'''
def separar_intencao_entidade(comando,frases_intencao,entidades):
    intencao = ''
    entidade = ''

    for token in frases_intencao:
        if token in comando:
            intencao += token + " "

    for token in entidades:
        if token in comando:
            entidade += token + " "

    return intencao,entidade
'''

def separar_intencao_entidade(comando,utterances,entidades):
    intencao = ''
    entidade = ''
    print(utterances)
    print(entidades)
    #ut = eval(utterances)
    #et = eval(entidades)
    #print('ut: '+str(ut))
    #print('ent: '+str(et))
    utterances = utterances[0].split(',')
    entidades = entidades[0].split(',')
    for token in utterances:
        print('token:' +token)
        if token in comando:
            intencao += token + " "
            break

    for token in entidades:
        if token in comando:
            entidade += token + " "

    return intencao,entidade

#separar_intencao_entidade("que horas sao","['teste']","['ent']")

def varreMatrizRelacoes(command):
    intencao = ''
    entidade = ''
    for i in range(len(intencoes_entidades_lookups.tabela_relacao_intencao_entidades)):
        intencao,entidade = separar_intencao_entidade(command,intencoes_entidades_lookups.tabela_relacao_intencao_entidades[i][0],intencoes_entidades_lookups.tabela_relacao_intencao_entidades[i][1])
        if intencao != '':
            break
    return intencao,entidade
'''
def varreMatrizRelacoes(command):
    intencao = ''
    entidade = ''
    for i in range(len(intencoes_entidades_lookups.relacoes_int_ent)):
        intencao,entidade = separar_intencao_entidade(command,intencoes_entidades_lookups.relacoes_int_ent[i][0],intencoes_entidades_lookups.relacoes_int_ent[i][1])
        if intencao != '':
            break
    return intencao,entidade
'''


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
    '''print(endpoint)
    print(headers)
    print(body)
    print(method)
    print(codigo_resposta)'''
    response = requests.request(method, endpoint, headers=json.loads(headers), data=body)

    # Processa a resposta baseado no código configurado
    contexto = {'response': response,'entidade': entidade}
    print(codigo_resposta)
    exec(codigo_resposta,globals(),contexto)
    return contexto.get("resposta")



#chamarAPI('https://servicodados.ibge.gov.br/api/v1/localidades/distritos','{"Content-Type": "application/json"}','{}','GET','for i in range(5):\n resposta = "Olá os distritos B são os seguintes: ";\n resposta += "->"+str(response.json()[i]["nome"]);','')