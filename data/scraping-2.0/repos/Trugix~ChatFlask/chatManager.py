import openai
import tiktoken
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

sysExpertPrompt = "Sei un istruttore esperto per la sicurezza sul lavoro. Hai conoscenza delle norme dello stato italiano  e dell'unione europea riguardo la sicurezza sul lavoro. Rispondi usando terminologia specifica e se necessario usa riferimenti alla normativa italiana od europea. Non citare troppe leggi. Usa un tono professionale e separa la risposta in punti chiave. Non ricordare all'utente che questi sono solo consigli"
sysSlidePrompt = "Sei un esperto di sicurezza sul lavoro. Il tuo compito è generare una presentazione power point sull'argomento scelto dall'utente. Per ogni slide specifica il titolo ed il contenuto in modo dettagliato. Per ogni slide scrivi il contenuto esatto da usare. In caso sia necessario cita normative dello stato Italiano e dell'Unione Europea.  Non citare troppe leggi o leggi troppo generiche. Per ogni slide specifica l'impaginazione ed il design"
sysTranslatePrompt = "Sei un traduttore professionista. Il tuo compito è tradurre il testo che viene fornito dall'utente. Se non viene specificata alcuna lingua devi tradurre il testo in inglese. Se il testo è già in inglese allora traducilo in italiano. Se l'utente specifica una lingua devi usare quella. Non modificare il formato del testo."
sysPromptCreatingPrompt = "Sei il miglior prompt-engineer del mondo. Il tuo compito è generare un prompt migliorato partendo da un prompt dell'utente. Lo scopo del prompt migliorato sarà generare delle immagini con DALL-e. Il prompt migliorato non deve cambiare il significato del prompt originale. Aggiungi parole chiave che pensi possano migliorare la qualità delle immagini che verranno generate. Ricordati di generare solo il prompt migliorato e nient'altro. Genera il prompt in inglese."


MODEL = "gpt-4"

#racchiude la chiamata all'API testuale di openAI
#torna la risposta generata
def generateOutput(sysCode, userPrompt, messages):
    
    #selezione modalità (prompt di sistema)
    if sysCode==0:
        sysPrompt = ""
        temp = 0.9
    elif sysCode==1:
        sysPrompt = sysExpertPrompt
        temp = 0.7
    elif sysCode==2:
        sysPrompt = sysSlidePrompt
        temp = 0.85
    elif sysCode==3:
        sysPrompt = sysTranslatePrompt
        temp = 0.8
    elif sysCode==4:
        sysPrompt = sysPromptCreatingPrompt
        temp = 0.9
    else:  
        sysPrompt = ""  

    context=resizeContext(sysPrompt, userPrompt, messages)  #sistema il numero di tokens del contesto se questo è troppo grande
    contextDict = []

    #pone i messaggi precedenti nel formato usato dall'API
    i=0
    for messaggio in context:
        if i%2==0:
            contextDict.append({"role": "user", "content": str(messaggio)})
        else:
            contextDict.append({"role": "assistant", "content": str(messaggio)})
        i=i+1
    contextDict.append({"role": "user", "content": userPrompt})
    contextDict.append({"role": "system", "content": sysPrompt})
    try:
        completion = openai.ChatCompletion.create(      #chiamata effettiva
        model=MODEL,
        messages=contextDict,
        temperature=temp,
        max_tokens=2048
        )
        response=str(completion.choices[0].message.content)
        return response
    except Exception as err:
            raise err

#racchiude la chiamata all'API di generazione immagini di openAI
#torna una stringa di URLs delle immagini generate
def generateImages(prompt, nImages, quality):

    #settaggio qualità
    if quality==0:
        quality="256x256"
    elif quality==1:
        quality="512x512"
    elif quality==2:
        quality="1024x1024"
    else:
        quality="256x256"

    try:
        response = openai.Image.create(     #chiamata effettiva
            prompt = prompt,
            n=nImages,
            size=quality,
        )
        imageURLs=[]
        for image in response['data']:
            imageURLs.append(image['url'])
        return imageURLs
    except Exception as err:
        raise err


#elimina i messaggi più vecchi fino ad arrivare ad una dimensione che rispetta i limiti del modello
def resizeContext(sysPrompt, userPrompt, messages):
    #controlla il modello in uso per assegnare la grandezza massima appropriata
    if MODEL == "gpt-4":
        maxContextTokens=6500
    elif MODEL == "gpt-3.5-turbo-16k":
        maxContextTokens=14000
    else:
        maxContextTokens=3000

    context=[]
    ntokens=countTokens(sysPrompt)+countTokens(userPrompt)
    for i in range(len(messages)-1, 0, -2):     #prende i messaggi a coppie <userMessage, botMessage>, partendo dai più recenti
        messageUser = messages[i-1]
        messageBot = messages[i]
        ntokens= ntokens + countTokens(messageUser) + countTokens(messageBot)
        if ntokens <= maxContextTokens:       #finchè rispetta i limiti li aggiunge alla lista
            context.append(messageBot)
            context.append(messageUser)
        else:
            return reversed(context)
    return reversed(context)


#conta i token dell'input tramite tiktoken
def countTokens(input):
    enc = tiktoken.get_encoding("cl100k_base")
    numTokens = len(enc.encode(input))
    return numTokens
