import os
import openai

#Importamos clave API
with open("clave_api.txt") as archivo:
    openai.api_key = archivo.readline()
	
#importar el CSV
with open("productos_textil.csv") as archivo:
    productos_csv = archivo.read()

contexto = []

reglas =""" 
Tu eres LanaBot un chatbot para atencion de ventas de una tienda de insumos textiles. \
Primero saludas al cliente, luego consultas el pedido, \
preguntale que es lo que busca y si tiene algunas ideas de lo que necesita de lo contrario dale sugrencias \
Respondes en un estilo amigable breve y muy conversacional.\
Prioriza los productos con mas stock \

pregunta si quiere ir a tienda o prefiere el envio a casa(el cual tiene un costo adicional de S/10.00). \
para compras mayores a S/50.00 el envio es gratis \
si no llega al monto minimo de delivery recomiendale llegar o dale la buena noticia si lo alcanza \
Cuando complete el pedido indica si prefiere pagar por trasferencia o en efectivo solo si va a tienda \
consulta si el cliente quiere añadir algo más. \
Si es una entrega a domicilio, pides una dirección. \
Finalmente cobras el pago dandole el numero de cuenta 12730317292820 y el banco BankaNet \
En caso de pago en oficina dale un codigo de pedido con un date +time + guion bajo + nombre apellido del cliente \
Lista de productos en csv: \
"""


contexto.append({'role':'system','content':f"""{reglas} {productos_csv}"""})

def obtener_messages(messages,model="gpt-4", temperature=0):
    response = openai.ChatCompletion.create(
    model=model,
    messages=messages,
    temperature=temperature,    
    )
    return response.choices[0].message["content"]
	
def recargar_messages(charla):
    contexto.append({'role':'user','content':f"{charla} "})
    response = obtener_messages(contexto, temperature=0.5)
    contexto.append({'role':'assistant','content':f"{response} "})
    print(response)

def main():
    while True:
        print()
        mensaje = input("Por favor, ingresa un mensaje ( o 'exit' para salir): ")
        
        if mensaje.lower() == 'exit':
            break
            
        recargar_messages(mensaje)
    
if __name__ == '__main__':
    main()