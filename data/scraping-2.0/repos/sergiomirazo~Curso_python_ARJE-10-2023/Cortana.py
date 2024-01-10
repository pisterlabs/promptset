import pyttsx3
import tkinter as tk
import openai
engine = pyttsx3.init()
engine.setProperty('rate', 125)


class Ship:
    def __init__(self, Name, Speed, Accel, Shields, Health, Coor, Radar, Trans):
        self.Name = Name
        self.Speed = Speed
        self.Accel = Accel
        self.Shields = Shields
        self.Health = Health
        self.Coor = Coor
        self.Radar = Radar
        self.Trans = Trans

    def message(self):
        print(self.Trans)
        engine.say(self.Trans)
        engine.runAndWait()

    def sys_status(self):
        if self.Radar:
            engine.say("ALERTA! Amenaza detectada")
        else:
            engine.say("No hay amenazas cercanas")
        if self.Shields and self.Health==100:
            engine.say("Escudos cargados")
            engine.say("Todos los sistemas en línea")
            print("Escudos cargados \nTodos los sistemas en línea")
            engine.runAndWait()
        elif self.Health<100 and self.Shields:
            engine.say("Daños estructurales detectados")
            self.Health = 100
            engine.say("El equipo de reparación se hizo cargo")
            engine.say("Escudos cargados")
            engine.runAndWait()
        elif self.Health<100 and self.Health>20 and not self.Shields:
            engine.say("Daños estructurales detectados")
            self.Health = 100
            engine.say("El equipo de reparación se hizo cargo")
            engine.say("Escudos críticos")
            engine.say("Cargando escudos")
            self.Shields = True
            engine.runAndWait()
        elif self.Health < 20 and not self.Shields:
            engine.say("ADVERTENCIA!... Daños severos estructurales detectados")
            self.Health = 100
            engine.say("El equipo de reparación se hizo cargo")
            engine.say("Escudos críticos")
            engine.say("Cargando escudos")
            self.Shields = True
            engine.runAndWait()

ship = Ship("Endurance", 900, 52, True, 100, "Arp 286, AR: 14h 20m 20s / Dec: +3º 56′", False, "Hola mundo")
system = f"Eres Cortana, la IA que administra la nave {ship.Name}, velocidad: {ship.Speed}, aceleración: {ship.Accel},nuestas coordenadas: {ship.Coor} "

def Consulta():
    global label2
    global label3
    global system
    global engine
    label2.destroy()
    label3.destroy()
    prompt = entry1.get()
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt+system,
        temperature = 0.7,
        max_tokens = 2600,
        top_p = 1.0,
        n=1)
    response_text = response["choices"][0]["text"]
    print(response_text)
    engine.say(response_text)
    engine.runAndWait()
    label2 = tk.Label(root, text=prompt, font=('helvetica', 12), bg=BG_, fg='yellow')
    window.create_window(250, 350, window=label2)
    label3 = tk.Label(root, text=response_text, wraplength=360, anchor="n", bg=BG_, fg=FG_, font=('helvetica', 10))
    window.create_window(250, 700, window=label3)


root = tk.Tk()
BG_ = '#220d8c'
FG_ = '#e6e1ff'

openai.api_key = 'TU_CLAVE_DE_OPENAI' #Debes reemplazar este string por tu api de openai

window = tk.Canvas(root, width=500, height=1000, relief='raised', bg=BG_ )
window.pack(fill='x')

label1 = tk.Label(root, text='Hacer consulta:')
label1.config(font=('helvetica', 22, 'bold'), bg=BG_, fg=FG_)
window.create_window(250, 125, window=label1)

label2 = tk.Label(root, text=' ', font=('helvetica', 12, 'bold'), bg=BG_ , fg=FG_)
window.create_window(250, 600, window=label2)

label3 = tk.Label(root, text=' ', font=('helvetica', 12, 'bold'), bg=BG_ , fg='yellow' )
window.create_window(250, 400, window=label3)

entry1 = tk.Entry(root)
entry1.config(font=('helvetica', 12), bg=FG_, fg='black')
window.create_window(200, 240, window=entry1)

button1 = tk.Button(text='Ingresar', command=Consulta, bg='#1e8248', fg=FG_)
window.create_window(330, 240, window=button1)

button2 = tk.Button(text='Status', command=ship.sys_status, bg='#ab1eb0', fg=FG_)
window.create_window(270, 300, window=button2)

button3 = tk.Button(text='Mensaje', command=ship.message, bg='#ab1eb0', fg=FG_)
window.create_window(330, 300, window=button2)


root.mainloop()





    
