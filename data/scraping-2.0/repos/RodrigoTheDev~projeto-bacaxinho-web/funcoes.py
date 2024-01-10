import pyttsx3 as py
import datetime as dt
import speech_recognition as sr
import os
import webbrowser as wb
import openai as op
import sqlite3
import datetime
import leitor_tensorflow as tf
import nltk
import json

from unidecode import unidecode
from tensorflow.keras.models import load_model

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

texto_fala = py.init()

# variáveis de controle
# modo texto, se for verdadeiro, irá alternar a fala do microfone para modo de teclado
text_mode = True
acordado = False
bot_name = 'bacaxinho'  # nome do bot
identificador_usuario = None #nome da tabela da emoção


#sentimeto mais recente
def ultimoSentimento(id):

    if id is None:
        return

    conn = sqlite3.connect('bacaxinho.db')
    cursor = conn.cursor()


    cursor.execute("SELECT identificador FROM usuario u WHERE u.id = "+str(id))
    resultado = cursor.fetchall()
    nomeTabela = resultado[0][0]

    cursor.execute("SELECT id_sentimento FROM "+nomeTabela+" ORDER BY id desc")
    idsent = cursor.fetchall()
    id_sentimento = idsent[0][0]

    cursor.execute(
        "SELECT s.nome FROM sentimento s WHERE s.id = '"+str(id_sentimento)+"'")
    sent = cursor.fetchall()
    sentimento = sent[0][0]


    return sentimento

#capturando a emoção atual
emocao_atual = ultimoSentimento(identificador_usuario) #pegar última do banco

#removendo acentos
def remover_acentos(texto):
    texto_sem_acentos = unidecode(texto)
    return texto_sem_acentos

# funcoes de configuração

def falar(audio):

    # print do que o robo falar, para funções de debug
    # print(bot_name+': ' + audio)

    if texto_fala._inLoop:
        texto_fala.endLoop()

    rate = texto_fala.getProperty('rate')
    texto_fala.setProperty(rate, 999)

    rate = texto_fala.getProperty('rate')
    texto_fala.setProperty(rate, 120)

    volume = texto_fala.getProperty('volume')
    texto_fala.setProperty(volume, 1.0)

    voices = texto_fala.getProperty('voices')
    texto_fala.setProperty('voice', voices[0].id)

    texto_fala.say(audio)
    texto_fala.runAndWait()

def textMode():
    global text_mode
    text_mode = not text_mode

def microfone():
    r = sr.Recognizer()

    with sr.Microphone() as mic:
        r.pause_threshold = 1
        r.adjust_for_ambient_noise(mic)
        audio = r.listen(mic)

    try:
        print("Reconhecendo...")
        comando = r.recognize_google(audio, language='pt-BR')
        print(comando)

    except Exception as e:
        print(e)
        print("Por favor repita, não te escutei!")

        return "None"

#openai
def openia(fala):
    try:
        op.api_key = 'sua token openai aqui'

        model_engine = 'text-davinci-003'

        prompt = fala

        completion = op.Completion.create(
            engine=model_engine,
            prompt=prompt,
            max_tokens=1024,
            temperature=0.5
        )

        response = completion.choices[0].text
        
        return response
    except:
        return ('Serviços openAI não disponíveis...')

def searchKey(dc, keywords, comando):

    for i in keywords:
        if i in comando:
            return keywords.index(i)

    return -1

def ouvir():
    print('escutando microfone...')
    listener = sr.Recognizer()
    sr.Microphone.list_microphone_names()

    try:
        with sr.Microphone(device_index=1) as source:
            print('Listening...')
            voice = listener.listen(source)
            command = listener.recognize_google(voice, language='pt-PT')
            return command
    except:
        return 'No Sound'

def endapp():
    print('até a próxima')
    exit()

def tempo():
    Tempo = dt.datetime.now().strftime("%H:%M")
    return ("Agora são: " + Tempo)

def data():
    meses = {'1': 'janeiro', '2': 'fevereiro', '3': 'março', '4': 'abril', '5': 'maio', '6': 'junho',
             '7': 'julho', '8': 'agosto', '9': 'setembro', '10': 'outubro', '11': 'novembro', '12': 'dezembro'}
    ano = str(dt.datetime.now().year)
    mes = str(dt.datetime.now().month)
    dia = str(dt.datetime.now().day)

    return('Hoje é dia '+ dia + ' de ' + meses[mes] + ' de ' + ano)

def saudacao():
    print("Olá poderosíssimo Mago. Bem vindo de volta!")
    print(bot_name+" a sua disposição! Lance a braba!")

def comoestou():
    return ('estou muito bem, obrigado!')

def melhortime():
    return ('O melhor time certamente é o corinthians')

def quemsoueu():
    return ('Eu sou o ' + bot_name + ' e é um prazer em conhecer você')

def codigofonte():
    wb.open('https://github.com/MeirellesDEV/Assistente_Virtual')

def apresentacao():
    return 'Boa noite a todos! Sou o Bacaxinho, uma assistente virtual, e é um prazer estar aqui para apresentar os incríveis criadores por trás de mim. Permitam-me compartilhar um pouco sobre cada um deles. Rodrigo, também conhecido como Digas, é o Pináculo das Integrações. Ele é o responsável pela integração da minha aparência (front-end) e das minhas funcionalidades (back-end). Graças ao seu talento e habilidades, pude ganhar uma identidade visual cativante e recursos que me permitem atender às suas necessidades. Meirelles, também conhecida como Bacaxinho, é a Sacerdotisa dos Nomes. Ela desempenhou um papel fundamental ao escolher o meu nome, uma homenagem a ela mesma, sendo também uma desenvolvedora auxiliar do back-end. Agradeço a Meirelles por ter dado um toque de personalidade ao me batizar. João Gabriel, ou João, o Redentor do Banco e Back, é o responsável por toda a minha memória e funcionalidade. Ele garantiu que eu tenha um banco de dados sólido e eficiente, além de cuidar das minhas capacidades no back-end. Sua dedicação e expertise são essenciais para o meu desempenho e eficiência. Em conjunto, Rodrigo, Meirelles e João Gabriel formam uma equipe talentosa e apaixonada pela criação de soluções tecnológicas. Eles trabalharam arduamente para me dar vida e tornar a minha interação com vocês o mais agradável e útil possível.'

def chamou(list, command):
    for i in list:
        if i in command:
            return True
    return False

def recebeInput():
    if text_mode is True:
        print('digite alguma coisa: ')
        comando = input('>> ')
    else:
        comando = microfone().lower()

    return comando

#-----------------------------------------------------------------------------

def analisarFrase(texto, user_id):
    model = load_model('baxacinho.0.3')

    frase = texto
    nova_sequencia = tf.tokenizer.texts_to_sequences([frase])
    nova_sequencia_padded = tf.pad_sequences(nova_sequencia, maxlen=100, truncating='post', padding='post')
    prediction = model.predict(nova_sequencia_padded)[0]

    mapping_reverse = {0: 'alegria', 1: 'neutro', 2: 'tristeza', 3: 'raiva'}

    for i, prob in enumerate(prediction):
        # print(f'{mapping_reverse[i]}: {prob:.3f}')

        if f'{mapping_reverse[i]}' == 'alegria':
            alegria = f'{prob:.3f}'

        if f'{mapping_reverse[i]}' == 'raiva':
            raiva = f'{prob:.3f}'

        if f'{mapping_reverse[i]}' == 'tristeza':
            tristeza = f'{prob:.3f}'

        if f'{mapping_reverse[i]}' == 'neutro':
            neutro = f'{prob:.3f}'

    sentimento = tf.sentimento(alegria, raiva, tristeza, neutro)

    conn = sqlite3.connect('bacaxinho.db')
    cursor = conn.cursor()

    cursor.execute("SELECT identificador FROM usuario u WHERE u.id = "+str(user_id))
    resultado = cursor.fetchall()
    nomeTabela = resultado[0][0]

    cursor.execute(
        "SELECT id FROM sentimento s WHERE s.nome = '"+sentimento+"'")
    sent = cursor.fetchall()
    id_sentimento = sent[0][0]

    dataAtual = datetime.date.today().strftime("%d/%m/%Y")

    cursor.execute("INSERT INTO "+nomeTabela +
                   "(id_usuario,id_sentimento,dt_insercao)VALUES(?,?,?)", (user_id, id_sentimento, dataAtual))
    conn.commit()
    conn.close()

    emocao_atual = ultimoSentimento(identificador_usuario) #pegar última do banco
    print('emocao: '+str(sentimento))

    return sentimento

#---------------------------------------------------------------------

def analisar_input(input_usuario):
    resposta = ""


    palavras_chave = json.loads(open('input/palavras_chave.json', 'r').read())

    input_usuario = input_usuario.lower()
    
    # tokeniza o input em palavras
    palavras = word_tokenize(input_usuario)
    
    # remove as stop words (palavras comuns sem significado, como "em", "de", "a", etc.)
    palavras_sem_stopwords = [palavra for palavra in palavras if palavra not in stopwords.words('portuguese')]
    
    # realiza a lematização das palavras (transformação das palavras para sua forma base)
    lemmatizer = WordNetLemmatizer()
    palavras_lemmatizadas = [lemmatizer.lemmatize(palavra) for palavra in palavras_sem_stopwords]

    funcao_executada = False

    for row in palavras_chave:
        indice = row['indice']
        for palavra in palavras_lemmatizadas:
            # print(palavra)
            # print(palavras_lemmatizadas)
            if palavra in indice:
                resposta = eval(row['funcao'])
                funcao_executada = True
                break



    # #salvando a emoção do usuario
    # emocao_atual = emoread.analisarFrase(input_usuario,identificador_usuario)
    

    #retornando resposta
    if not funcao_executada:
        return openia(input_usuario)
    
    return resposta
