import openai
import getch

# Configuración de la API de OpenAI
openai.api_key = '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'

# Oraciones o frases en español para traducir
frases = [
    "Hola, ¿cómo estás?",
    "Me gustaría pedir una pizza, por favor.",
    "¿Dónde está la estación de tren?",
    "Vamos al cine esta noche.",
    "¡Feliz cumpleaños!"
]

# Función para obtener una traducción en alemán utilizando OpenAI
def obtener_traduccion(texto):
    traduccion = openai.Completion.create(
        engine="text-davinci-003",
        prompt=f"Translate the following sentence from Spanish to German:\n\n{texto}\n\nTranslation: ",
        max_tokens=100,
        temperature=0.3,
        top_p=1,
        n=1,
        stop=None
    )
    return traduccion.choices[0].text.strip()

# Función para esperar la tecla de entrada
def esperar_tecla():
    print("Presiona cualquier tecla para continuar...")
    getch.getch()

# Función para jugar el juego de traducción
def jugar():
    puntaje = 0
    respuestas_correctas = []
    respuestas_incorrectas = []
    
    # Mostrar las frases en español
    print("Frases en español:")
    for frase in frases:
        print("-", frase)
    esperar_tecla()
    print("\n")
    
    for frase in frases:
        # Traducción original
        traduccion_original = obtener_traduccion(frase)
        
        # Obtener una palabra o frase faltante
        palabras = traduccion_original.split()
        palabra_faltante = palabras.pop(len(palabras) // 2)  # Remover una palabra al azar
        
        # Mostrar la traducción con la palabra o frase faltante
        traduccion_incompleta = ' '.join(palabras)
        print(f"Traducción: {traduccion_incompleta} ______ {palabra_faltante}")
        
        # Obtener la respuesta del jugador
        esperar_tecla()
        respuesta_jugador = input("Ingresa la palabra o frase faltante en alemán: ")
        
        # Verificar la respuesta utilizando OpenAI
        if respuesta_jugador.strip().lower() in palabra_faltante.lower():
            puntaje += 1
            respuestas_correctas.append(frase)
        else:
            respuestas_incorrectas.append(frase)
        
    # Mostrar el puntaje final
    print(f"\nTu puntaje final es: {puntaje}/{len(frases)}")
    print("---------Análisis de respuestas:---------")
    print("Respuestas correctas:")
    for respuesta in respuestas_correctas:
        print("-", respuesta)
    print("---------Respuestas incorrectas:---------")
    for respuesta in respuestas_incorrectas:
        print("-", respuesta)
        
    # Conclusion de la IA
    print("\nConclusión:")
    if puntaje == len(frases):
        print("¡Felicitaciones! Has respondido todas las preguntas correctamente. Tu comprensión del alemán es impresionante.")
    elif puntaje > len(frases) // 2:
        print("Has tenido un buen desempeño en el juego. Sigue practicando para mejorar tu comprensión del alemán.")
    else:
        print("Aunque no obtuviste un puntaje alto, no te desanimes. El aprendizaje de un nuevo idioma lleva tiempo y práctica. Sigue esforzándote y verás mejoras.")

# Función principal
def main():
    print("............Bienvenido al juego de traducción ............\n")
    print("¿Qué quieres hacer?")
    print("1. Jugar")
    print("2. Salir")
    opcion = input("\nIngresa tu opción: ")
    
    if opcion == "1":
        jugar()
    elif opcion == "2":
        print("¡Gracias por jugar!")
    else:
        print("Opción inválida")
        main()

# Ejecutar la función principal
if __name__ == "__main__":
    main()
