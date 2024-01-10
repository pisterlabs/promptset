# bibliophiles:
import openai
import time
import pandas as pd
import aws
# ------------------------------------#
# rules:
openai.api_key = "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"
tam_request = 50
result = aws.data
# ------------------------------------#
inicio_1= time.time()
def request(messages):
    # api log:
    response = openai.ChatCompletion.create(model="gpt-3.5-turbo",
                                            messages= messages,
                                            max_tokens=1024,
                                            temperature=0.5)

    # request:
    return [response.choices[0].message.content, response.usage]

# ------------------------------------#
# process:

comentarios = []
gpt_result =  []

inicio = time.time()
for i in range(0, len(result), tam_request):

    coment = result[i:i + tam_request]

    mensagens = [{"role": "system",
                  "content": "preciso que você interplete o sentimento desses textos, retorne apenas positivo, neutro, negativo e 0 caso vc não consigua identificar, escreva o resultado em um formato de lista separado por virgula e sem ponto final "}]

    mensagens.append({"role": "user", "content": str(coment)})

    try:
        temp = request(mensagens)
        comentarios.extend(coment)
        gpt_result.append(temp[0])
        print(temp[0])  # diagnostico
        print(coment)   #comentarios
        print()



    except openai.error.RateLimitError:
        print("aguardando limite")
        fim = time.time()
        time.sleep(80 - (fim - inicio))
        continue
# # ------------------------------------#
diagnostico = []

for elemento in gpt_result:
    palavras = elemento.split(',')
    palavras = [palavra.strip('.') for palavra in palavras]
    diagnostico.extend(palavras)

print(f"{diagnostico} tamanho{len(diagnostico)}")
print(f"{comentarios} tamanho{len(comentarios)}")
# ------------------------------------#
# close:

df = pd.DataFrame({'comentarios': comentarios, 'sentimentos': diagnostico})

pd.to_csv(df)
fim_1 = time.time()
print(fim_1-inicio_1)
# # ------------------------------------#