import json
from typing import Any, Dict, List, Optional, Union
import os
from ailearn_ai.model import openai_model
from ailearn_ai.model import exercises_model
from langchain.llms import OpenAI
from ailearn_ai.model import openai_model
from ailearn_ai import logger
import time

class CoursesModel:

    def create_course(self, subject: str, difficulty: int, isSupervised: bool) -> Dict[str, Any]:
        
        course = {
            "subject": subject,
            "difficulty": difficulty,
            "isSupervised": isSupervised,
            "sections": []
        }

        MODEL_NAME= "gpt-3.5-turbo"
        openAIModel = openai_model.OpenAIModel()
        llm = openAIModel.get_model(MODEL_NAME)
        MAX_RETRIES = 10

        def map_difficulty(difficulty: int) -> str:
            if difficulty <= 1:
                return "básica"
            elif difficulty == 2:
                return "media"
            elif difficulty >= 3:
                return "avanzada"
            else:
                return "media"

        # SECTION NAMES
        request = f"Devuleve una lista con los nombres de las subsecciones más importantes de un curso de {subject} de dificultad {map_difficulty(difficulty)}" 
        output_format = '{"sectionNames": [str, str, str, ...]}'
        for _ in range(MAX_RETRIES):
            try:
                sectionNames = json.loads(openAIModel.prompt_llm(request, output_format, llm))["sectionNames"]
                break  # Break the loop if llm() call is successful
            except Exception as e:
                print(f"Error: {e} sectionNames")
                continue
        #Save to course
        for sectionName in sectionNames:
            course["sections"].append({
                "title": sectionName,
                "lessons": []
            })
        
        #SECTION DESCRIPTIONS
        for section in course["sections"]:
            request = f"""Genera una muy breve descripción de la seccion {section['title']} de un curso de {subject} de dificultad {map_difficulty(difficulty)}
                        Resume el contenido de la sección en no más de 5 frases""" 
            output_format = '{"text": str}'
            for _ in range(MAX_RETRIES):
                try:
                    section["text"] = json.loads(openAIModel.prompt_llm(request, output_format, llm))["text"]
                    break  # Break the loop if llm() call is successful
                except Exception as e:
                    print(f"Error: {e} sectionText in section {section['title']}")
                    continue

        # LESSON NAMES
        for section in course["sections"]:
            request = f"Devuelve una lista con los nombres de las lecciones más importantes de la seccion {section['title']} de un curso de {subject} de dificultad {map_difficulty(difficulty)}" 
            output_format = '{"lessonNames": [str, str, str, ...]}'
            for _ in range(MAX_RETRIES):
                try:
                    lessonNames = json.loads(openAIModel.prompt_llm(request, output_format, llm))["lessonNames"]
                    break  # Break the loop if llm() call is successful
                except Exception as e:
                    print(f"Error: {e} lessonNames")
                    continue
            # Save to course
            for lessonName in lessonNames:
                section["lessons"].append({
                    "title": lessonName,
                    "text": ""
                })

        # LESSONS TEXT
        for section in course["sections"]:
            for lesson in section["lessons"]:
                request = f"""Desarrolla el tema {lesson['title']} (en el contexto de {section['title']} y {course['subject']}). Debes explicarlo en un nivel de dificultad {map_difficulty(difficulty)}
                            """ 

                output_format = '{"text": str}'
                for _ in range(MAX_RETRIES):
                    try:
                        lesson["text"] = json.loads(openAIModel.prompt_llm(request, output_format, llm))["text"]
                        break  # Break the loop if llm() call is successful
                    except Exception as e:
                        print(f"Error: {e} lessonText in lesson {lesson['title']}")
                        continue
        
        # EXERCISES
        NUMBER_OF_EXERCISES = 1
        if course["isSupervised"]:
            exercisesModel = exercises_model.ExercisesModel()
            for section in course["sections"]:
                for lesson in section["lessons"]:
                    
                    lesson["exercises"] = exercisesModel.create_exercises(subject=course["subject"], text=lesson["text"], difficulty=course["difficulty"], number=NUMBER_OF_EXERCISES)

        logger.info(f"Course created: {course}")
        return course
    
    