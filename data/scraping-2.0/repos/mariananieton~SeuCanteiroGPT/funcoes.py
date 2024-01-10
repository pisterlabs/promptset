import openai
import requests
import datetime

def cenourinha_gpt():
    openai.api_key = "Sua API Key aqui"

    messages = [{"role": "system", "content": "You are an assistant who answers questions about planting, growing, and harvesting only"}]

    print("Posso te ajudar a saber tudo sobre plantio e cultivo! Qual sua dúvida?")
    while True:
        user_response = input()
        messages.append({"role": "user", "content": user_response})
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages)
        resposta = response.choices[0].message.content
        print(resposta)
        messages.append({"role": "assistant", "content": resposta})

        print("Posso ajudar em algo mais?")
        texto = input()
        if texto and "não" in texto.lower():
            print("Ok, se precisar pode me chamar!")
            break
        else:
            print("Claro, faça mais uma pergunta!")

def get_clima():
    print("Diga a cidade que você quer saber o clima")
    cidade = input()
    api_key = "Sua API Key aqui"
    url = f"http://api.openweathermap.org/data/2.5/weather?q={cidade}&lang=pt_br&appid={api_key}&units=metric"
    response = requests.get(url)
    data = response.json()

    if data["cod"] == 200:
        clima = data["weather"][0]["description"]
        temperatura = data["main"]["temp"]
        umidade = data["main"]["humidity"]
        velocidade_vento = data["wind"]["speed"]
        result = f"O clima em {cidade} está {clima}. A temperatura é de {temperatura:.1f} graus Celsius, com umidade de {umidade}% e velocidade do vento de {velocidade_vento} metros por segundo."
    else:
        result = f"Não foi possível obter as informações meteorológicas para {cidade}."

    print(result)

def recomendacao_por_epoca():
    primavera = [9, 10, 11]
    verao = [12, 1, 2]
    inverno = [7, 8]
    outono = [3, 4, 5, 6]
    mes_atual = datetime.date.today().month

    if mes_atual in primavera:
        print("Como estamos na primavera, recomendo plantar abobrinha, agrião, alface,"
                                    " batata inglesa, beterraba, brócolis, cenoura, cebola, couve, espinafre e repolho")
    elif mes_atual in verao:
        print("Como estamos no verão, recomendo plantar abóbora, berinjela, chuchu, couve, beterraba,"
                                    " cenoura, pepino, ervilha, tomate, milho e feijão")
    elif mes_atual in inverno:
        print("Como estamos no inverno, recomendo plantar abóbora, abobrinha, agrião, alface, beterraba,"
                                    " brócolis, cenoura, chuchu, couve, couve-flor, espinafre, aipim e repolho")
    elif mes_atual in outono:
        print("Como estamos no outono, recomendo plantar abóbora, chuchu, couve, pepino, batata doce,"
                                    " aipim, beterraba e cenoura")


