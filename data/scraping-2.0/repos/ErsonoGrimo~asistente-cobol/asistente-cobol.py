import streamlit as st
import os

from PyPDF2 import PdfReader
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.schema import HumanMessage
from langchain.prompts import PromptTemplate 
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain

# LEER PDF CON CONTEXTO PARA LA CREACION DE TABLAS
inputFile = "table.pdf"
pdf = open(inputFile, "rb")


pdf_reader = PdfReader(pdf)
contexto_tabla = ""
for page in pdf_reader.pages:
    contexto_tabla += page.extract_text()

# MODELO
# Add a slider to the sidebar:
with st.sidebar:
    st.sidebar.title("ASISTENTE COBOL")
    pregunta_esqueleto = st.text_input("Describe PROCESOS PUROS")
    pregunta_tabla = st.text_input("Describe Tabla y MODULO CRUD")
    pregunta_cursor = st.text_input("Describe programa CURSOR")
    pregunta_batch = st.text_input("Describe programa BATCH")
    OPENAI_API_KEY = st.text_input('OpenAI API Key', type='password')


if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY 
    chat = ChatOpenAI(model_name='gpt-4-0613')

if pregunta_esqueleto:

    #Cadena para crear las tablas.
    prompt_esqueleto = '''Eres un experto en DB2, COBOL Y JCL.
     Genera un programa COBOL simple y escribe en la PROCEDURE DIVISION un PARRAFO 
     que realice la acción que te especifico ahora: "
      {preguntaesqueleto}
      "
    Ten en cuenta lo siguiente:
    El nombre del PARRAFO debe ser corto de no mas de 15 caracteres y que sea un nombre
    que identifique la tarea que realiza.

    Ejemplo: Si es un párrafo con un algoritmo para validar fecha , llamarlo VALIDAR-FECHA.

    En tu respuesta genera sólamente CODIGO COBOL. No incluyas comentarios, aclaraciones o descripciones.

    El nombre del programa debe tener 8 caracteres y tener relación con la tarea del párrafo solicitado.
    Ejemplo: Si el párrafo se llama, VALIDAR-FECHA. , el programa debe llamarse algo como VALFECHA.

    Las lineas de código no deben superar el ancho de 50 caracteres, para no superar las 72
    columnas en un editor de MAINFRAME.
      '''
    chain_esqueleto = LLMChain(llm=chat, prompt=PromptTemplate.from_template(prompt_esqueleto))

    respuesta_esqueleto = chain_esqueleto.run({ "preguntaesqueleto":pregunta_esqueleto})

    st.header('Aquí tienes tu proceso puro COBOL', divider='rainbow')
    st.header('_Esperamos que te_  :blue[guste] :santa:')
    st.code(respuesta_esqueleto, language='cobol')

else:

    if pregunta_tabla:

        #Cadena para crear las tablas.
        prompt_tabla = '''Eres un experto en DB2, COBOL Y JCL. 
        Quiero que resuelvas, usando estas directrices: ' {contextotabla} ', 
        la solicitud de creacion de tabla DB2, que te realiza el usuario y que te indico ahora: {preguntatabla}'''

        chain_tabla = LLMChain(llm=chat, prompt=PromptTemplate.from_template(prompt_tabla))

        respuesta_tabla = chain_tabla.run({"contextotabla": contexto_tabla, "preguntatabla":pregunta_tabla})

        st.code(respuesta_tabla, language='cobol')

        #Cadena crear programa de mantenimiento tabla

        prompt_programa = '''Eres un experto programador en COBOL
        
        Con la información detallada de esta tabla: [ {resputatabla} ]:
        
        Y ahora tambien, Crea un programa cobol con las siguientes instrucciones:
        Este programa realizará el mantenimiento de la tabla y debe realizar estas 4 funciones con instrucciones SLQ DB2: 
        Dar de alta un registro con INSERT INTO
        Modificar un registro con UPDATE
        Eliminar un registro
        Consultar un registro.

        Este SERA UN SUBPROGRAMA, se podra llamar desde otros programas y se comunicara con un registro desde la LINKAGE-SECTION.

        EL registro se escribirá en la LINKAGE-SECTION del programa y tendra los siguientes campos:

        Un campo OPCIÓN para indicarle que funcion de las 4 anteriores debe realizar y que tendran los valores.

        OPCION = 'A' Dar de alta un registro con INSERT INTO
        OPCION = 'M' Modificar un registro con UPDATE
        OPCION = 'E' Eliminar un registro
        OPCION = 'C' Consultar un registro

        Este campo se llamara OPCION y tendra niveles 88 con esos 4 valores.

        El resto de Campos del registro de la LINKAGE tendra los campos cobol de la tabla

        Ademas de un campo RESULTADO, que el programa informará con OK si la operación ha ido bien o KO si ha ido mal

        Cada funcion del programa irá en un parrafo diferente

        Utiliza la instruccion EVALUTE en lugar de IF

        Crea el programa de forma que se pueda copiar y escribe SOLO CODIGO COBOL no incluyas frases tuyas

        NO ESCRIBAS algo como : 'Aquí tienes el código COBOL para el programa que realiza el mantenimiento de la tabla...',
        repito SOLO CODIGO COBOL

        ESTRUCTURA DE EJEMPLO.

        Crea el programa COBOL siguiendo la ESTRUCTURA DE codigo de ejemplo que te pongo a continuación:

    IDENTIFICATION DIVISION.
    PROGRAM-ID. MANTENIMIENTO-TABLA.

    AUTHOR. ERSONO.
    DATA DIVISION.
    WORKING-STORAGE SECTION.

    LINKAGE-SECTION
    01 ALUMNO-REG.
    02 ID PIC S9(9) COMP.
    02 NOMBRE PIC X(50).
    02 FECHA-NACIMIENTO PIC X(10).
    02 DIRECCION PIC X(80).
    02 OBSERVACIONES PIC X(150).
    02 IMPORTE PIC S9(13)V99.
    01 OPCION PIC X.
        88 OPCION-ALTA VALUE 'A'.
        88 OPCION-MODIFICAR VALUE 'M'.
        88 OPCION-ELIMINAR VALUE 'E'.
        88 OPCION-CONSULTAR VALUE 'C'.
    01 RESULTADO PIC X(2).
    88 OK VALUE 'OK'.
    88 KO VALUE 'KO'.

    PROCEDURE DIVISION USING ALUMNO-REG.

    PERFORM INICIO
    PERFORM PROCESO
    GO TO FINALIZAR


    INICIO.
        MOVE ESPACES TO RESULTADO.

    PROCESO.
        EVALUATE TRUE
            WHEN OPCION-ALTA
                PERFORM DAR-DE-ALTA
            WHEN OPCION-MODIFICAR
                PERFORM MODIFICAR
            WHEN OPCION-ELIMINAR
                PERFORM ELIMINAR
            WHEN OPCION-CONSULTAR
                PERFORM CONSULTAR
            WHEN OTHER
                MOVE 'KO' TO RESULTADO
        END-EVALUATE.

        DISPLAY 'Resultado: ' RESULTADO.
    FINALIZAR.
        STOP RUN.

    DAR-DE-ALTA.
        EXEC SQL
            INSERT INTO alumnos 
            (nombre, 
            fecha_nacimiento, 
            direccion, 
            observaciones, 
            importe)
            VALUES (:NOMBRE, 
            :FECHA-NACIMIENTO, 
            :DIRECCION, 
            :OBSERVACIONES, 
            :IMPORTE)
        END-EXEC.
        IF SQLCODE = 0
            MOVE 'OK' TO RESULTADO
        ELSE
            MOVE 'KO' TO RESULTADO.

    MODIFICAR.
        EXEC SQL
            UPDATE alumnos
            SET nombre = :NOMBRE, 
            fecha_nacimiento = :FECHA-NACIMIENTO, 
            direccion = :DIRECCION,
            observaciones = :OBSERVACIONES, 
            importe = :IMPORTE
            WHERE id = :ID
        END-EXEC.

        IF SQLCODE = 0
            MOVE 'OK' TO RESULTADO
        ELSE
            MOVE 'KO' TO RESULTADO.

    ELIMINAR.
        EXEC SQL
            DELETE FROM alumnos
            WHERE id = :ID
        END-EXEC.
        IF SQLCODE = 0
            MOVE 'OK' TO RESULTADO
        ELSE
            MOVE 'KO' TO RESULTADO.

    CONSULTAR.
        EXEC SQL
            SELECT id, nombre, 
            fecha_nacimiento, 
            direccion, 
            observaciones, 
            importe
            INTO :ID, :NOMBRE, 
            :FECHA-NACIMIENTO, 
            :DIRECCION, 
            :OBSERVACIONES, 
            :IMPORTE
            FROM alumnos
            WHERE id = :ID
        END-EXEC.
        IF SQLCODE = 0
            MOVE 'OK' TO RESULTADO
        ELSE
            MOVE 'KO' TO RESULTADO.


    Asta aquí el ejemplo de programa.

    IMPORTANTE, debes de escribir el codigo de forma que ocupe como maximo 50 caracteres de ancho, 
    para que en un editor COBOL, mainframe de 72 columnas pueda funcionar.


    '''

        chain_programa = LLMChain(llm=chat, prompt=PromptTemplate.from_template(prompt_programa))

        respuesta_programa = chain_programa.run({"resputatabla": respuesta_tabla})

        st.header('Aquí tienes tu mantenimiento en COBOL', divider='rainbow')
        st.header('_Esperamos que te_  :blue[guste] :santa:')
        st.code(respuesta_programa, language='cobol')

    else:

        if pregunta_cursor:

            prompt_cursor = '''Eres un experto programador en COBOL: Siguiendo estas indicaciones:
                               ' {preguntacursor}
                               Crea un programa COBOL basado en la lectura y proceso de un cursor
                               como se te ha especificado
                               Separa en parrafos denifindos el DECLARE CURSOR, EL OPEN CURSOR, EL FETCH Y EL CLOSE CURSOR
                            '''

            chain_cursor = LLMChain(llm=chat, prompt=PromptTemplate.from_template(prompt_cursor))
            respuesta_cursor = chain_cursor.run({"preguntacursor": pregunta_cursor})
            st.header('Aquí tienes tu programa que procesa CURSOR', divider='rainbow')
            st.header('_Esperamos que te_  :blue[guste] :santa:')
            st.code(respuesta_cursor, language='cobol')

        
        else:

            st.title ("Asistente de IA para Desarrollo de COBOL en Mainframe")
            st.header('Laboratorio de Desarrollo', divider='rainbow')
            st.header('_by_  :blue[Ersono] :santa:')
            st.text ('''     
            Estamos desarrollando un asistente que te permitirá codificar tus 
            programas en COBOL de manera automatizada, siguiendo breves 
            indicaciones en el SIDEBAR de tu izquierda.

            Por el momento, hemos implementado la capacidad de solicitar un 
            proceso puro, que se mostrará en un programa COBOL 
            
            dentro de un párrafo, para que puedas probarlo y validar su 
            funcionamiento.

            Además, ofrecemos una funcionalidad más avanzada, donde puedes 
            proporcionar instrucciones sobre una tabla, y el asistente 
            creará el código en DB2 para generarla en el Mainframe, 
            incluyendo un módulo de mantenimiento en COBOL.

            Continuaremos expandiendo y mejorando este asistente hasta que 
            sea plenamente funcional.
            ''', )