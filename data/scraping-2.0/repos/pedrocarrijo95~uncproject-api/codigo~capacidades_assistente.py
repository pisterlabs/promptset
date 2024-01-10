import pywhatkit
import datetime
import pytz
import requests
import random
import openai


import funcoes
import voz
import perfil
import credenciais
import falas_assistente
import intencoes_entidades_lookups
import variaveis
from TuyaSmartAPI import controle_tuya



#Capacidades do Assistente
list_capacidades = [
    'Saudação',
    'Tocar música',
    'Horas',
    'Previsão Tempo',
    'Alterar Cidade do perfil',
    'Gratidão',
    'Descansar',
    'Dúvidas com GPT',
    'Garoto Bonito',
    'Garota Bonita',
    'Cotação de moedas',
    'Notícias',
    'Email'
]

numero_aleatorio = random.randint(0, 4)
print('Resposta aleatória gerada: '+str(numero_aleatorio))
def acordar():
    variaveis.ASSISTENTE_DESCANSAR = False
    variaveis.BEEP = True

    return falas_assistente.text_abertura[numero_aleatorio]
    #voz.talk(falas_assistente.text_abertura[numero_aleatorio])

def tocar(command,intencao): #frases_tocar
    song = funcoes.substituir_strings(command,intencao,'')
    voz.talk('tocando ' + song)
    pywhatkit.playonyt(song)
    variaveis.ASSISTENTE_DESCANSAR = True
    variaveis.BEEP = False

def horario(entidade): 
    periodo = ''
    horas_txt = ''
    min_txt = ''
    datatempo_atual = datetime.datetime.now()
    fuso_horario = ''
    if entidade != '':
        fuso_horario = funcoes.definir_fuso_horario_por_cidade(entidade)
        fuso = pytz.timezone(fuso_horario)
        if fuso != None:
            datatempo_atual = datatempo_atual.astimezone(fuso)
            entidade = 'em '+entidade
    if datatempo_atual.hour < 2:
        horas_txt = 'hora'
    else:
        horas_txt = 'horas'    

    if datatempo_atual.minute < 2:
        min_txt = 'minuto'
    else:
        min_txt = 'minutos'                
        
    if datatempo_atual.hour < 13:
        periodo = ' da manhã'
    elif datatempo_atual.hour >=  13 and datatempo_atual.hour < 19:
        periodo = ' da tarde'
    elif datatempo_atual.hour >= 19:
        periodo = ' da noite'
    else:
        periodo = ' '
    return 'Agora são exatamente ' + str(datatempo_atual.hour) + ' ' + horas_txt + ' e ' + str(datatempo_atual.minute) + ' ' + min_txt + periodo + ' ' +entidade
    #voz.talk('Agora são exatamente ' + str(datatempo_atual.hour) + ' ' + horas_txt + ' e ' + str(datatempo_atual.minute) + ' ' + min_txt + periodo + ' ' +entidade)

def previsaoTempo(entidade):
    cidadeaux = ''
    if entidade != '':
        cidadeaux = perfil.CIDADE
        perfil.CIDADE = entidade
    link = "https://api.openweathermap.org/data/2.5/weather?q="+perfil.CIDADE+"&appid="+credenciais.APIKEY_WEATHER+"&lang=pt_br"
    requisicao = requests.get(link)
    requisicao_dic = requisicao.json()
    descricao = requisicao_dic['weather'][0]['description']
    temperatura = round(requisicao_dic['main']['temp'] - 273.15)
    voz.talk('Tempo em '+perfil.CIDADE+' se encontra '+descricao+' e com '+str(temperatura)+' º célsius')
    perfil.CIDADE = cidadeaux


def alterar_cidade(entidade):
    if entidade != '':
        perfil.CIDADE = entidade
    else:
        voz.talk('Tudo bem ! Me diga qual sua nova cidade.')
        perfil.CIDADE = voz.resposta_pergunta_assistent();
    voz.talk('Perfeito ! Agora sua cidade foi alterada para '+perfil.CIDADE)
    voz.talk('Posso te ajudar em mais alguma coisa?')
    
    
def gratidao():
    variaveis.ASSISTENTE_DESCANSAR = True
    variaveis.PALAVRA_ASSISTENTE = False
    variaveis.BEEP = False
    voz.talk(falas_assistente.text_agradecimento)
    
def descansar():
    variaveis.ASSISTENTE_DESCANSAR = True
    variaveis.PALAVRA_ASSISTENTE = False
    variaveis.BEEP = False
    voz.talk(falas_assistente.text_descanso)
    
def duvidas():
    variaveis.ASSISTENTE_DESCANSAR = True
    variaveis.PALAVRA_ASSISTENTE = False
    variaveis.GPT = True
    variaveis.CONTGPT = 0
    voz.talk(falas_assistente.text_duvidas)
    
def garota_bonita():
    voz.talk(falas_assistente.text_garota_bonita)
    voz.talk('Posso te ajudar em mais alguma coisa?')

def garoto_bonito():
    voz.talk(falas_assistente.text_garoto_bonito)
    voz.talk('Posso te ajudar em mais alguma coisa?')

def cotacao_moedas(entidade):
    sigla = 'BRL'
    if entidade != '':
        for i,ent in enumerate(intencoes_entidades_lookups.lookup_cotacao):
            if ent[0] in entidade: 
                sigla = ent[1]
                break
    print('Sigla detectada: '+sigla)
    valor_cotacao = round(float(funcoes.consultar_cotacoes(sigla)),2)    
    print(valor_cotacao)
    voz.talk('O valor atual da moeda '+entidade+' é igual a '+str(valor_cotacao)+' reais')
    voz.talk('Posso te ajudar em mais alguma coisa?')
    
def noticias(entidade):
    periodo = 1
    if entidade != '':
        for i,ent in enumerate(intencoes_entidades_lookups.lookup_noticias):
            if ent[0] == entidade: 
                periodo = ent[1]
                break
    voz.talk('Certo, estou buscando as últimas notícias mundiais e preciso traduzi-las para seu idioma, aguarde alguns minutos...')
    noticias = funcoes.consultar_noticias(periodo)
    for i in range(len(noticias)):
        print('Titulo: '+noticias[i][0])
        print('Resumo: '+noticias[i][1])
        voz.talk('Notícia em alta no mundo é intitulada como... '+noticias[i][0]+' e um resumo seria o seguinte... '+noticias[i][1])
        voz.talk('Tenho mais '+str(len(noticias) - (i+1))+' notícias, você deseja que eu continue passando por elas?')
        resposta = voz.resposta_pergunta_assistent()
        if 'sim' in resposta:
            pass
        else:
            voz.talk('Tudo bem ! Vamos deixar as notícias por enquanto então')
            voz.talk('Posso te ajudar em mais alguma coisa?')
            break

def email():
    voz.talk('Vamos lá ! Primeiro precisa me dizer para qual o email destinatário quer enviar')
    resposta_emaildestino = voz.resposta_pergunta_assistent()
    resposta_emaildestino = funcoes.tratarEmail(resposta_emaildestino)

    email_confirmado = False
    assunto_confirmado = False
    corpo_confirmado = False

    print('Email destinatário: '+resposta_emaildestino)

    while not email_confirmado:
        voz.talk('O E-mail ficou o seguinte: '+resposta_emaildestino+', está correto?')
        resposta__confirma_destinatario = voz.resposta_pergunta_assistent()
        resposta_emaildestino = funcoes.tratarEmail(resposta_emaildestino)
        if 'não' in resposta__confirma_destinatario: 
            voz.talk('Então repita por favor')
            resposta_emaildestino = voz.resposta_pergunta_assistent()
            resposta_emaildestino = funcoes.tratarEmail(resposta_emaildestino)
            print('Email destinatário: '+resposta_emaildestino)
        elif 'sim' in resposta__confirma_destinatario:
            email_confirmado = True

    voz.talk('Agora me diga o assunto que deseja ter neste email...')
    resposta_assunto = voz.resposta_pergunta_assistent()
    print('Email assunto: '+resposta_assunto)
    while not assunto_confirmado:
        voz.talk('Seu assunto de email ficou o seguinte: '+resposta_assunto+', está correto?')
        resposta__confirma_assunto = voz.resposta_pergunta_assistent()
        if 'não' in resposta__confirma_assunto: 
            voz.talk('Então repita por favor')
            resposta_assunto = voz.resposta_pergunta_assistent()
            print('Email assunto: '+resposta_assunto)
        elif 'sim' in resposta__confirma_assunto:
            assunto_confirmado = True

    variaveis.LIMITE_TEMPO_FRASE = 40
    voz.talk('Agora me diga o corpo do email, você tem um espaço de 40 segundos...')
    resposta_corpo = voz.resposta_pergunta_assistent()
    print('Email corpo: '+resposta_corpo)
    while not corpo_confirmado:
        voz.talk('Seu assunto de email ficou o seguinte: '+resposta_corpo+', está correto?')
        variaveis.LIMITE_TEMPO_FRASE = 10
        resposta__confirma_corpo = voz.resposta_pergunta_assistent()
        if 'não' in resposta__confirma_corpo: 
            voz.talk('Então repita por favor')
            variaveis.LIMITE_TEMPO_FRASE = 40
            resposta_corpo = voz.resposta_pergunta_assistent()
            print('Email corpo: '+resposta_corpo)
        elif 'sim' in resposta__confirma_corpo:
            corpo_confirmado = True
            variaveis.LIMITE_TEMPO_FRASE = 10
            

    funcoes.enviar_email(resposta_emaildestino,resposta_assunto,resposta_corpo)
    voz.talk('Email enviado com sucesso! Precisa de algo mais?')
    
def capacidades():
    voz.talk('Ainda estou em desenvolvimento, mas já sei fazer algumas coisas que são:')
    for i in range(len(list_capacidades)):
        voz.talk(list_capacidades[i])
    voz.talk('Posso te ajudar em mais alguma coisa?')
    pass
    
def erro_commando():
    voz.talk(falas_assistente.text_erro_comando)


openai.api_key = credenciais.OPENAI_APIKEY #expirou
def chamarGPT(command):
    if variaveis.CONTGPT == 0:
        voz.talk(falas_assistente.text_gpt)
        variaveis.CONTGPT += 1
        return

    pergunta_gpt = command

    if any(ext in pergunta_gpt for ext in intencoes_entidades_lookups.frases_despedida_gpt): #frases_despedida_gpt
        variaveis.GPT = False
        variaveis.ASSISTENTE_DESCANSAR = False
        voz.talk(falas_assistente.text_voltar_assistente)
        return
    elif pergunta_gpt == '':
        voz.talk('GPT Não conseguiu detectar seu comando, repita por favor.')
    else:
        variaveis.MENSAGENS_GPT.append({"role": "user", "content": str(pergunta_gpt)})

        answer = funcoes.gerar_resposta(variaveis.MENSAGENS_GPT)
        #print("ChatGPT:", answer[0], "\nCusto:\n", answer[1])
        voz.talk(answer[0])
        variaveis.MENSAGENS_GPT.append({"role": "assistant", "content": answer[0]})
        

    debugar = False
    if debugar:
        print("variaveis.MENSAGENS_GPT", variaveis.MENSAGENS_GPT, type(variaveis.MENSAGENS_GPT))
        
        
def controlar_tuya(intencao,entidade):
    intencao_valor = ''
    entidade_valor = ''
    cor_hsv = []
    if intencao != '':
        for i,int in enumerate(intencoes_entidades_lookups.lookup_tuya_intencoes):
            if int[0] in intencao: 
                intencao_valor = int[1]
                break
    if entidade != '':
        for i,ent in enumerate(intencoes_entidades_lookups.lookup_tuya_entidades):
            if ent[0] in entidade: 
                entidade_valor = ent[1]
                break
        for i,ent in enumerate(intencoes_entidades_lookups.lookup_tuya_cores):
            if ent[0] in entidade: 
                cor_hsv = [ent[1],ent[2],ent[3]]
                break
        
    branca = False        
    if 'branca' in entidade or 'normal' in entidade:
        branca = True
            
    print('Lookup intenção: '+intencao_valor)
    print('Lookup entidade: '+entidade_valor)        
    if 'ligar' in intencao_valor:
        controle_tuya.ligar_desligar(True,entidade_valor,)
    elif 'apagar' in intencao_valor:
        controle_tuya.ligar_desligar(False,entidade_valor)
    elif 'mudar' in intencao_valor:
        controle_tuya.mudar_cor(cor_hsv,entidade_valor,branca)
                
    voz.talk('Tudo bem, está feito. Posso ajudar em algo mais?')
    

