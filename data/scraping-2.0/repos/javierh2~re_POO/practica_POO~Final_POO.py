from textblob import TextBlob


class AnalizadorDeSentimientos:
    def analizar_sentimiento(self, texto):
        analisis = TextBlob(texto)
        if analisis.sentiment.polarity > 0:
            return "positivo"
        elif analisis.sentiment.polarity == 0:
            return "neutral"
        else:
            return "negativo"


analizador = AnalizadorDeSentimientos()
resultado = analizador.analizar_sentimiento("this is a sad day for me")
print(resultado)

#import openai
#
#openai.api_base = "api-key"
#
#system_rol = '''Vas a funcionar como un analizador de sentimientos.
#                Yo te voy a entregar texto y desde su contenido me vas a dar tu devolucion sobre el estado emocional
#                la respuesta es con al menos 1 caracter y como maximo 4 caracteres
#                SOLO RESPUESTAS NUMERICAS donde -1 es negatividad maxima, 0 es neutral y 1 positividad maximas.
#                Podes ir entre esos rangos, es decir, 0,3, -0,5, etc tambien son validos
#                (Podes responder solo con ints o floats)'''
#
#mensajes = [{"role": "system", "content": system_rol}]
#
#
#class AnalizadorDeSentimientos:
#    def analizar_sentimiento(self, polaridad):
#        if polaridad > -0.6 and polaridad <= 0.3:
#            return "\x1b[1;31m" + "Negativo" + "\x1b[0;37m"
#        elif polaridad > -0.3 and polaridad < 0.1:
#            return "\x1b[1;31m" + "Algo negativo" + "\x1b[0;37m"
#        elif polaridad >= -0.1 and polaridad <= 0.1:
#            return "\x1b[1;33m" + "Neutral" + "\x1b[0;37m"
#        elif polaridad >= 0.1 and polaridad <= 0.5:
#            return "\x1b[1;32m" + "Algo positivo" + "\x1b[0;37m"
#        elif polaridad >= 0.4 and polaridad <= 0.9:
#            return "\x1b[1;32m" + "Positivo" + "\x1b[0;37m"
#        elif polaridad > 0.9:
#            return "\x1b[1;32m" + "Muy positivo" + "\x1b[0;37m"
#        else:
#            return "\x1b[1;31m" + "Muy negativo" + "\x1b[0;37m"
#
#
#analizador = AnalizadorDeSentimientos()
#
#while True:
#    user_prompt = input("Decime algo: ")
#    mensajes.append({"role": "user", "content": user_prompt})
#
#    completion = openai.ChatCompletion.create(
#        model="gpt-3.5-turbo",
#        messages=mensajes,
#        max_tokens=8
#    )
#
#    respuesta = completion.choices[0].message["content"]
#    mensajes.append({"role": "assistant", "content": respuesta})
#    sentimiento = analizador.analizar_sentimiento(float(respuesta))
#    print(sentimiento)
