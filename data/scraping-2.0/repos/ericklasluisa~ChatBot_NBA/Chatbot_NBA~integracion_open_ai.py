import openai

# Define tu clave de API de OpenAI
openai.api_key = "INSERT HERE THE KEY" #key de OpenAI eliminada por seguridad


def relacionado_nba(pregunta):
    nba_keywords = [
        "NBA", "anillos", "baloncesto", "basket", "jugador", "jugadores",
        "equipo", "equipos", "campeonato", "título", "trofeo",
        "anotación", "puntos", "rebote", "rebotes", "asistencia", "asistencias",
        "estadísticas", "estadística", "partido", "partidos", "torneo", "torneos",
        "conferencia", "conferencias", "playoffs", "temporada", "temporadas",
        "entrenador", "entrenadores", "entrenamiento", "canasta", "canastas",
        "tiros libres", "defensa", "ofensiva", "triple", "triples",
        "doble", "dobles", "pases", "faltas", "bloqueos", "robo", "robos",
        "estrellas", "jugador estrella", "historia NBA", "jugador histórico",
        "mejores jugadores NBA", "salón de la fama NBA", "All-Star", "MVP NBA",
        "drafteo", "jugador novato", "base", "ala-pívot", "pívot", "ala",
        "tiro de campo", "tiro de tres", "tiro libre", "tiros encestados",
        "rebotes ofensivos", "rebotes defensivos", "porcentaje de tiros",
        "cancha", "zonas", "tablero", "uniforme", "balón"
    ]
    return any(keyword in pregunta for keyword in nba_keywords)


def obtener_respuesta_API(prompt):
    if relacionado_nba(prompt):
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt="Pregunta referente a la NBA: " + prompt,
            max_tokens=200
        )
        return response.choices[0].text.strip()
    else:
        return "Lo siento, solo respondo preguntas relacionadas con la NBA."
