# Made by Lamberto Ruiz Martínez
# GitHub: https://github.com/LambertoRM/

import openai

openai.api_key = "MI-API-KEY"

class chatGPT():
    def __init__(self, coleccion, quest, qua):
        self.coleccion = coleccion
        self.quest = quest
        self.qua = qua

    def pregunta(self):      
        self.quest = "Buenas, quiero que me digas " + self.qua + " posibles recetas con estos alimentos: "+ str(self.coleccion)
        
        print("ChatGPT request:\n" + self.quest)

        respuesta = openai.ChatCompletion.create(model="gpt-3.5-turbo",
                                                 messages=[{"role": "user", "content": self.quest}])
        #print(respuesta)
        lista_rec = (respuesta.choices[0].message.content).split("\n")
        #print(lista_rec)
        return lista_rec
    
    def pregunta_diferente(self):
        self.quest = "Buenas, quiero que me digas 10 posibles recetas muy elaboradas con estos alimentos: "+ str(self.coleccion)
        
        print("ChatGPT request:\n" + self.quest)

        respuesta = openai.ChatCompletion.create(model="gpt-3.5-turbo",
                                                 messages=[{"role": "user", "content": self.quest}])
        #print(respuesta)
        lista_rec = (respuesta.choices[0].message.content).split("\n")
        return lista_rec