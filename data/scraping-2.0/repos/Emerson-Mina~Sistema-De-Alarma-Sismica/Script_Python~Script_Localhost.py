import serial
import mysql.connector
from datetime import date, time, datetime
import yagmail
import openai


openai.api_key = "Inserire la chiave Open IA"


conexion = mysql.connector.connect(
    host="localhost", 
    user="root",       
    password="",      
    database="test_db" 
)

if conexion.is_connected():
    print("Connessione al database riuscita")
else:
    print("Errore di connessione al database")
    exit(1)

cursor = conexion.cursor()


ser = serial.Serial('/dev/ttyUSB0', 9600)  #!Modificare la porta COM e la velocità in base alle impostazioni


email = 'Il vostro indirizzo e-mail  '     
contraseña = 'password dell applicazione gmail (Google).'      


yag = yagmail.SMTP(user=email, password=contraseña)

def enviar_correo(subject, body):
    try:
        yag.send(
            to='La posta del destinatario può essere inserita con input per cambiare il destinatario ogni volta, può essere inviata in massa',    #! Cambiare il destinatario con l'e-mail a cui si desidera inviare i dati.
            subject=subject,
            contents=body
        )
        print("Email inviata")
    except Exception as e:
        print("Errore nell'invio dell'e-mail:", str(e))


while True:
    try:
        if ser.is_open:
            
            linea = ser.readline().decode().strip()
            if linea:
                magnitud, resultado = linea.split(':')
                
                #!Una riga di dati viene letta da un dispositivo seriale. La riga viene quindi divisa in due parti utilizzando ":" come 
                #!separatore, assegnando la prima parte a "magnitudo" e la seconda a "risultato".
                fecha_actual = date.today()
                hora_actual = time(datetime.now().hour, datetime.now().minute, datetime.now().second)
                
                #!è possibile utilizzare qualsiasi database, nel mio caso ho utilizzato phpMyAdmin-Mysql.
                cursor.execute("INSERT INTO datos (magnitud, resultado, fecha, hora) VALUES (%s, %s, %s, %s)",
                (magnitud, resultado, fecha_actual, hora_actual))
                conexion.commit()
                print("Datos insertados: Magnitud={}, Resultado={}, Fecha={}, Hora={}".format(magnitud, resultado, fecha_actual, hora_actual))    
            
            
                completion = openai.Completion.create(
                engine="text-davinci-003",
                prompt=f"Un terremoto di :{resultado}. Mi date i punti chiave su cosa devo fare e quali misure devo adottare?",
                    max_tokens=2048
                #?Si accede all'API OpenAI facendo una richiesta in base ai dati acquisiti dal sensore e si invia una risposta al destinatario via e-mail.
                )

                openia = completion.choices[0].text
                
                
                
                subject = "¡Alerta SISMICA!"
               
                body = "<html><body>"
                body += "<h2 style='text-align: center; color: #F93D3D; font-size: 30px; font-weight: bold;'>Descripción de lo que ocurrió</h2>"
                body += "<table style='margin: 0 auto; width: 60%; border-collapse: separate; border-spacing: 0; border-radius: 10px; overflow: hidden;'>"
                body += "<tr><th style='background-color: #F93D3D; color: white; font-weight: bold; padding: 10px 15px; border: none; border-right: 1px solid #ddd; border-bottom: 1px solid #ddd; border-top-left-radius: 10px;'><b style='font-size: 20px;'>Magnitud</b></th>"
                body += "<th style='background-color: #F93D3D; color: white; font-weight: bold; padding: 10px 15px; border: none; border-right: 1px solid #ddd; border-bottom: 1px solid #ddd;'><b style='font-size: 20px;'>Resultado</b></th>"
                body += "<th style='background-color: #F93D3D; color: white; font-weight: bold; padding: 10px 15px; border: none; border-right: 1px solid #ddd; border-bottom: 1px solid #ddd;'><b style='font-size: 20px;'>Fecha</b></th>"
                body += "<th style='background-color: #F93D3D; color: white; font-weight: bold; padding: 10px 15px; border: none; border-bottom: 1px solid #ddd; border-top-right-radius: 10px;'><b style='font-size: 20px;'>Hora</b></th></tr>"
                body += "<tr><td style='text-align: center; padding: 10px 15px; border: none; border-right: 1px solid #ddd; border-bottom: 1px solid #ddd; font-size: 18px;'><b>{}</b></td>".format(magnitud)
                body += "<td style='text-align: center; padding: 10px 15px; border: none; border-right: 1px solid #ddd; border-bottom: 1px solid #ddd; font-size: 18px;'><b>{}</b></td>".format(resultado)
                body += "<td style='text-align: center; padding: 10px 15px; border: none; border-right: 1px solid #ddd; border-bottom: 1px solid #ddd; font-size: 18px;'><b>{}</b></td>".format(fecha_actual)
                body += "<td style='text-align: center; padding: 10px 15px; border: none; border-bottom: 1px solid #ddd; border-top-right-radius: 10px; font-size: 18px;'><b>{}</b></td></tr>".format(hora_actual)
                body += "</table>"
                body += "<br><br>"
                body += "<div style='text-align: center; font-size: 18px;'><b>{}</b></div>".format(openia)
                body += "</body></html>"
                
            enviar_correo(subject, body)         
    except KeyboardInterrupt:
        ser.close()
        cursor.close()
        conexion.close()
        print("Conexiones cerradas")
        break
