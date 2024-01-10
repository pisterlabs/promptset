import json
from typing import Any, Dict, List, Optional, Union
from ailearn_ai.model import openai_model
import random


class ExercisesModel:

    def create_exercises(self, subject: str, text: str, difficulty: int, number: int) -> Dict[str, Any]:

        def map_difficulty(difficulty: int) -> str:
            if difficulty <= 1:
                return "básica"
            elif difficulty == 2:
                return "media"
            elif difficulty >= 3:
                return "avanzada"
            else:
                return "media"
            if difficulty == 1:
                return "baja"
            elif difficulty == 2:
                return "media"
            elif difficulty == 3:
                return "alta"
            else:
                return "media"
            
        MODEL_NAME= "gpt-3.5-turbo"
        openAIModel = openai_model.OpenAIModel()
        llm = openAIModel.get_model(MODEL_NAME)

        MAX_RETRIES = 10
        TYPES = ["openExercise", "multipleChoiceExercise"]
        exercises = []
       
        for i in range(number):
            exerciseType = random.choice(TYPES)

            """
            Response format:

            {
                exercise: {
                    "type": str,
                    "question": str,
                    "explanation": str,
                    "options": [str, str, str, ...] (if type is open exercise, options is empty),
                    "correct_option": str (if type is open exercise, correct_option is None)
                }
            }
            
            """

            exercise = {
                "type": exerciseType,
                "options": [],
                "correct_option": None,
            }
            if exerciseType == "openExercise":
                #Open exercise
                request = f"""Estas creando un curso de {subject} de dificultad {map_difficulty(difficulty)}
                            Ten en cuenta que la información requerida en la respuesta debe aparecer en el texto de la lección
                            Devuelve un ejercicio de tipo abierto para siguiente leccion: {text}
                            Añade una pequeña aclaración (en explanation) si lo consideras estrictamente necesario, en caso contrario devuelve null""" 
                output_format = '{"question": str, "explanation": str}'
            elif exerciseType == "multipleChoiceExercise":
                #Multiple choice exercise
                request = f"""Estas creando un curso de {subject} de dificultad {map_difficulty(difficulty)}
                            Ten en cuenta que la información requerida en la respuesta debe aparecer en el texto de la lección
                            Devulve un ejercicio de multiple elección para la siguiente lección: {text} (debes proporcionar varias opciones de las cuales SOLO UNA es correcta)
                            Añade una pequeña aclaración (en explantation )si lo consideras estrictamente necesario, en caso contrario devuelve null"""
                output_format = '{"question": str, "explanation": str, "options": [str, str, str, ...], "answer": str}'

            if i > 0:
                    #Añadir a request los ejercicios anteriores
                    request += f"""
                                Ten en cuenta que previamente has añadido los siguientes ejercicios:
                                {[f"{exercise[exercise['type']]}" for exercise in exercises]}
                                Intenta que los ejercicios sean lo más variado posible
                                """
            for _ in range(MAX_RETRIES):
                try:
                    response = json.loads(openAIModel.prompt_llm(request, output_format, llm))
                    exercise["question"] = response["question"]
                    exercise["explanation"] = response["explanation"]
                    if exerciseType == "multipleChoiceExercise":
                        exercise["options"] = response["options"]
                        exercise["correct_option"] = response["answer"]
                    
                    break  # Break the loop if llm() call is successful
                except Exception as e:
                    print(f"Error: {e} create_exercises")
                    continue
            exercises.append(exercise)
        return exercises
    
    def validate_exercise(self, question: str, explanation: str, answer: str) -> Dict[str, Any]:
        
        MODEL_NAME= "gpt-3.5-turbo"
        openAIModel = openai_model.OpenAIModel()
        llm = openAIModel.get_model(MODEL_NAME)

        MAX_RETRIES = 10

        #2STEP approach:

        #1- Generate a response to the question
        request = f"""Cual sería una respuesta correcta a la siguiente pregunta: {question}
                        Debes devolver una respuesta concisa, sin entrar en explicaciones
                        Responde de la manera más breve que se te ocurra"""
        output_format = '{"response": str}'
        for _ in range(MAX_RETRIES):
            try:
                response = json.loads(openAIModel.prompt_llm(request, output_format, llm))["response"]
                break  # Break the loop if llm() call is successful
            except Exception as e:
                print(f"Error: {e} validate_exercise")
                continue

        #2- Compare the response with the user's answer
        request = f"""
                    Debes corregir una respuesta a la pregunta '{question}'. Una posible respuesta a la pregunta sería '{response}'.
                    La respuesta proporcionada por el alumno es '{answer}'. ¿Es dicha respuesta correcta?
                    Ten en cuenta que hay muchas formas correctas de escribir una respuesta: debes IGNORAR ESPACIOS EN BLANCO, faltas de ortografía sin importancia, nombres arbitrarios de variables, etc.)

                    Devuelve un booleano result con valor True en caso de que la respuesta sea correcta y False en caso contrario
                    Devuelve una pequeña explicación de porqué está mal, en caso de que la respuesta sea correcta devuelve null
                    """
        output_format = '{"response"{"result": bool, "explanation": str}}}}'
        for _ in range(MAX_RETRIES):
            try:
                result = json.loads(openAIModel.prompt_llm(request, output_format, llm))["response"]
                break  # Break the loop if llm() call is successful
            except Exception as e:
                print(f"Error: {e} validate_exercise")
                continue

        return result
