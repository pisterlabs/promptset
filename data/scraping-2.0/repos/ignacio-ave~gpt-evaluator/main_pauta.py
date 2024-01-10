import json
import pandas as pd
import openpyxl
import hashlib
import os

import openai
from langchain.output_parsers import PydanticOutputParser

from utils.file_utils import save_to_file, load_from_file, get_processed_prompts_count, save_processed_prompts_count, improved_save_to_file
from utils.dataframe_utils import reorganize_dataframe
from utils.json_utils import extract_from_json_adjusted
from utils.openai_utils import init_openai
from utils.settings_utils import ensure_settings_exists, load_settings, save_settings
from models.evaluation import EvaluationJSON, QuestionInfo

def save_processed_prompts(processed_prompts_dict):
    with open("processed_prompts.json", "w") as file:
        json.dump(processed_prompts_dict, file)

def load_processed_prompts():
    try:
        with open("processed_prompts.json", "r") as file:
            return json.load(file)
    except:
        return {}
            

def load_json_data():
    """ Carga archivo json con las respuestas y preguntas e incluye IDs únicos """
    file_path = input("Por favor, ingresa la ruta completa del archivo JSON (incluyendo el nombre del archivo): ")
    
    with open(file_path, "r") as file:
        data = json.load(file)
    
    students_data = data[0]
    all_students_data_adjusted = extract_from_json_adjusted(students_data)
    max_columns = max(len(student) for student in all_students_data_adjusted)
    
    for student in all_students_data_adjusted:
        while len(student) < max_columns:
            student.append(None)
    
    num_questions = (max_columns - 6) // 2
    adjusted_columns = ["Nombre", "Email", "Estado", "Inicio", "Fin", "Tiempo"] + [f"Q{i//2+1}" if i % 2 == 0 else f"A{i//2+1}" for i in range(num_questions * 2)]
    df_all_students_final_adjusted = pd.DataFrame(all_students_data_adjusted, columns=adjusted_columns)
    

    df_all_students_final_adjusted['Student_ID'] = df_all_students_final_adjusted['Email'].apply(generate_student_id)
    # df_all_students_final_adjusted['Evaluation_ID'] = generate_evaluation_id()
    
    output_path = input("Por favor, ingresa la ruta donde deseas guardar el archivo Excel (sin incluir el nombre del archivo): ")
    output_file_name = input("Por favor, ingresa el nombre que deseas para el archivo Excel (por ejemplo, 'mi_archivo.xlsx'): ")
    df_all_students_final_adjusted.to_excel(output_path + '/' + output_file_name, index=False)
    print(f"¡El archivo ha sido guardado en {output_path}/{output_file_name}!")

    df = df_all_students_final_adjusted
    return df



def load_questions_from_excel(filepath):
    """ Carga la pauta """
    pauta_df = pd.read_excel(filepath)
    questions = []
    
    for index, row in pauta_df.iterrows():
        question_id_hash = hashlib.sha256(row["Pregunta"].encode()).hexdigest()
        
        question_info = {
            'question_id': question_id_hash,
            'student': "Sample Student",
            'topic': row["Tópico"],
            'question': row["Pregunta"],
            'total_points': row["Total Puntos"],
            'criteria': row["Scoring Guideline"],
            'scoring_guideline': row["Scoring Guideline"],
            'good_response': row["Response GPT-4"],
            'bad_response': row["Bad Response GPT-4"]
        }
        questions.append(question_info)
    return questions

def generate_student_id(email):
    """Genera un ID único para un estudiante basado en su correo electrónico."""
    return hashlib.sha256(email.encode()).hexdigest()


def get_responses_by_student_id(student_id, responses, parsed_evaluations):
    """Retorna datos basado en ID"""
    student_responses = [response for response in responses if response["student_id"] == student_id]
    student_evaluations = [evaluation for evaluation in parsed_evaluations if evaluation["student_id"] == student_id]
    
    return student_responses, student_evaluations

def save_responses_to_excel(df, responses, parsed_evaluations):
    """Guardar respuestas y evaluaciones en un archivo Excel."""
    
    # Crear un DataFrame para las respuestas y evaluaciones
    data = []
    for response, evaluation in zip(responses, parsed_evaluations):
        student_name = df.loc[df['Student_ID'] == response["student_id"], 'Nombre'].iloc[0]
        student_email = df.loc[df['Student_ID'] == response["student_id"], 'Email'].iloc[0]
        # Asumiendo que la respuesta contiene la pregunta
        question = response["response"].split("Pregunta: ")[1].split("Respuesta Proporcionada: ")[0].strip()
        # Extraer las puntuaciones y razonamientos de la evaluación
        scores = evaluation["evaluation"].scores
        reasons = evaluation["evaluation"].reasons
        total_evaluation = "; ".join([f"{key}: {value} ({reasons[key]})" for key, value in scores.items()])

        data.append([student_name, student_email, question, response["response"], total_evaluation])

    # Convertir la lista a un DataFrame
    columns = ["Nombre", "Email", "Pregunta", "Respuesta", "Evaluación"]
    responses_df = pd.DataFrame(data, columns=columns)

    # Guardar en Excel
    output_path = input("Por favor, ingresa la ruta donde deseas guardar el archivo Excel (sin incluir el nombre del archivo): ")
    output_file_name = input("Por favor, ingresa el nombre que deseas para el archivo Excel (por ejemplo, 'respuestas.xlsx'): ")
    responses_df.to_excel(output_path + '/' + output_file_name, index=False)
    print(f"¡El archivo ha sido guardado en {output_path}/{output_file_name}!")




def generate_prompts_from_dataframe(df, questions, format_instructions):
    '''Genera prompts con la informacion relacionada a la pregunta'''
    IGNORED_RESPONSES = ["Falso", "Verdadero", "-", "No proporcionada"]
    
    question_dict = {question: {"topic": "Desconocido", "scoring_guideline": "Desconocido", "criteria": "Desconocido", 
                                "good_response": "No proporcionada", "bad_response": "No proporcionada"} for question in questions}
    prompts = []
    
    for (student, question_text), row in df.iterrows():
        
        if row['Respuesta'] in IGNORED_RESPONSES:
            continue


        question = question_dict.get(question_text, {})
        topic = question.get('topic', "Desconocido")
        scoring_guideline = question.get('scoring_guideline', "Desconocido")
        criteria = question.get('criteria', "Desconocido")
        good_response = question.get('good_response', "No proporcionada")
        bad_response = question.get('bad_response', "No proporcionada")

        prompt_text = f'''
        Actúa como un evaluador universitario de la materia ESTRUCTURA DE DATOS. En esta materia se evalúa el conocimiento de los estudiantes del departamento de informática en estructura de datos bien conocida.
        Las respuestas serán teóricas y breves.

        La pregunta a evaluar trata sobre la estructura {topic}
        Información de la Estructura: {scoring_guideline}

        Pregunta: {question_text}
        Respuesta Proporcionada: {row['Respuesta']}

        Respuesta a evaluar con 3 puntos : {good_response}
        Respuesta a evaluar con 0 puntos : {bad_response}

        Evalua asignando un puntaje de 0 a 3 en cada criterio de evaluación. 0 es la peor calificación y 3 es la mejor calificación.

        Por favor, sigue los siguientes pasos para evaluar la respuesta según los criterios de evaluación:
        1. Analiza la estructura y la información proporcionada.
        2. Revisa la pregunta y la respuesta proporcionada.
        3. Razone sobre la adecuación de la respuesta a la pregunta, considerando los criterios específicos para la estructura en cuestión.
            - Razonamiento sobre la Correctitud de la Respuesta
            - Razonamiento sobre la Claridad de la Respuesta
            - Razonamiento sobre la Relevancia de la Respuesta
        4. Rellena el archivo JSON de evaluación adjunto, incluyendo tus razonamientos y el puntaje asignado en cada criterio.
        5. Devuelve el archivo JSON completado.

        Formato del JSON:
        {format_instructions}
        '''
        student_data = {
            "name": student,
            "email": row.get("Email", "") 
        }
        prompts.append({"student": student_data, "prompt": prompt_text})

    return prompts


unprocessed_prompts_dict = {}


def generate_responses(df, prompts, responses=None, parsed_evaluations=None):
    """Genera respuestas a partir de una lista de prompts con reintentos y guarda los IDs únicos."""
    global unprocessed_prompts_dict
    processed_prompts = load_processed_prompts()
    settings = load_settings()
    max_tokens = settings.get("max_tokens", 500)
    max_attempts = settings.get("max_attempts", 3)

    pydantic_parser = PydanticOutputParser(pydantic_object=EvaluationJSON)
    
    if responses is None:
        responses = []
    
    if parsed_evaluations is None:
        parsed_evaluations = []

    total_prompts = len(prompts)
    print(f"Total de prompts a procesar: {total_prompts}")


    for i, prompt in enumerate(prompts):
        student_id = prompt["student"]["email"]
        question_id = hashlib.sha256(prompt["prompt"].encode()).hexdigest()

        if student_id in processed_prompts and question_id in processed_prompts[student_id]:
            continue


        prompt_content = prompt["prompt"]
        messages = [{'role': 'system', 'content': prompt_content}]
        response_content = None

        for attempt in range(max_attempts):
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=messages,
                    max_tokens=max_tokens
                )
                response_content = response.choices[0].message['content']
                response_content_with_id = {
                    "student_id": student_id,
                    "response": response_content
                }
                responses.append(response_content_with_id)

                eval_parsed = pydantic_parser.parse(response_content)
                eval_parsed_with_id = {
                    "student_id": student_id,
                    "evaluation": eval_parsed.to_dict()
                }
                parsed_evaluations.append(eval_parsed_with_id)


                # Add to processed prompts
                if student_id not in processed_prompts:
                    processed_prompts[student_id] = []
                processed_prompts[student_id].append(question_id)
                save_processed_prompts(processed_prompts)
                break

            except Exception as e: 
                if attempt == max_attempts - 1:
                    print(f"Error en el intento {attempt+1} para el prompt {i+1}: {e}")
                    unprocessed_prompts_dict[student_id] = prompt_content
                    responses.append(None)
                    break

            with open("output.txt", "a") as f:
                f.write(f"###Human:{prompt}\n###Assistant:{response_content}\n\n")

        if response_content:
            print(f"Response {i+1}:\n{response_content}\n{'-'*50}")
            print(f"Prompt {i + 1}/{total_prompts} procesado correctamente.")

    return responses, parsed_evaluations



def load_questions_from_excel(filepath):
    """Carga las preguntas de un archivo excel y las retorna como una lista de QuestionInfo"""
    pauta_df = pd.read_excel(filepath)
    questions = []
    
    for index, row in pauta_df.iterrows():
        question_info = QuestionInfo(
            question_id=row["ID"],
            student="Sample Student",  # Ejemplo para la desmostracion de la pauta
            topic=row["Tópico"],
            question=row["Pregunta"],
            total_points=row["Total Puntos"],
            criteria=row["Scoring Guideline"],
            scoring_guideline=row["Scoring Guideline"],
            good_response=row["Response GPT-4"],
            bad_response=row["Bad Response GPT-4"]
        )
        questions.append(question_info)
    return questions




def load_data():
    """Cargar datos de estudiantes y preguntas."""
    df = load_json_data()
    df = reorganize_dataframe(df)
    questions_filepath = input("Por favor, ingrese el nombre del archivo 'pauta.xlsx' y su ruta: ")
    questions = load_questions_from_excel(questions_filepath)
    return df, questions

def generate_and_process(df, questions):
    """Generar prompts y obtener respuestas."""
    pydantic_parser = PydanticOutputParser(pydantic_object=EvaluationJSON)
    format_instructions = pydantic_parser.get_format_instructions()
    prompts = generate_prompts_from_dataframe(df, questions, format_instructions)
    responses, parsed_responses = generate_responses(df, prompts)
    return responses, parsed_responses

def save_data(responses, parsed_responses):
    """Guardar respuestas y parsed_responses en archivos."""
    improved_save_to_file(responses, "responses.json")
    improved_save_to_file(parsed_responses, "parsed_responses.json")

def load_previous_data():
    """Cargar datos previamente guardados."""
    try:
        responses = load_from_file("responses.json")
        parsed_responses = load_from_file("parsed_responses.json")
        return responses, parsed_responses
    except:
        return None, None

def display_results(responses):
    """Mostrar las respuestas."""
    for response in responses:
        print(response)

def main():
    global unprocessed_prompts_dict

    ensure_settings_exists()
    init_openai()
    
    pydantic_parser = PydanticOutputParser(pydantic_object=EvaluationJSON)
    format_instructions = pydantic_parser.get_format_instructions()

    option = input("¿Desea cargar datos previamente guardados? (s/n): ").strip().lower()
    if option == 's':
        responses, parsed_responses = load_previous_data()
        if not responses or not parsed_responses:
            print("Error cargando datos. Cargando nuevos datos...")
            df, questions = load_data()
            prompts = generate_prompts_from_dataframe(df, questions, format_instructions) # Generar los prompts aquí.
            responses, parsed_responses = generate_responses(df, prompts)
            save_data(responses, parsed_responses)
        else:
            display_results(responses)
            option_continue = input("¿Desea continuar procesando nuevos datos? (s/n): ").strip().lower()
            if option_continue == 's':
                df, questions = load_data()
                prompts = generate_prompts_from_dataframe(df, questions, format_instructions)
                new_responses, new_parsed_responses = generate_responses(df, prompts)
                responses.extend(new_responses)
                parsed_responses.extend(new_parsed_responses)
                save_data(responses, parsed_responses)
                display_results(new_responses)
    else:
        df, questions = load_data()
        prompts = prompts = generate_prompts_from_dataframe(df, questions, format_instructions)
        responses, parsed_responses = generate_responses(df, prompts)
        save_data(responses, parsed_responses)
        display_results(responses)

    # Si es que existen prompts que no se pudieron procesar
    if unprocessed_prompts_dict:
        unprocessed_count = len(unprocessed_prompts_dict)
        print(f"\nQuedan {unprocessed_count} prompts sin procesar.")
        user_input = input(f"¿Deseas procesar nuevamente los prompts no procesados? (s/n): ").strip().lower()
        if user_input == 's':
            unprocessed_prompts = [{"student": k, "prompt": v} for k, v in unprocessed_prompts_dict.items()]
            unprocessed_responses, unprocessed_parsed_responses = generate_responses(df, unprocessed_prompts)
            responses.extend(unprocessed_responses)
            parsed_responses.extend(unprocessed_parsed_responses)
            save_data(responses, parsed_responses)
            display_results(unprocessed_responses)

            # Limpiar el diccionario después de procesar
            unprocessed_prompts_dict.clear()


if __name__ == "__main__":
    main()
