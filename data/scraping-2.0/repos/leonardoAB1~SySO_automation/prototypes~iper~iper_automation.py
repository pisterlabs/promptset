import os
import openai
import pandas as pd
import numpy as np
from IPython.display import display, Markdown
import questionary
from questionary import select

openai.api_key = os.environ['SYS_OPENIA_API_KEY']

class IPER_Row:
    '''
    Fila de la Matriz de Identificación de Peligros y la Evaluación de Riesgos(IPER).
    En términos simples, IPER es una descripción organizada de las actividades, 
    controles y peligros que permitan identificar los posibles riesgos. 
    Esta permitirá evaluar, monitorear, controlar y comunicar estos peligros 
    o sucesos no deseados, pudiendo también identificar los niveles de riesgo 
    y las consecuencias de estos.

    Los valores y definiciones de los distintos indicadores se describen a continuación:
        Índice de número de personas expuestas (self.INPE): Se determina en función de la cantidad de personas expuestas, en el lugar o entorno de trabajo.
        0: No aplica (NO existe exposición de personas, sólo de equipos/infraestructura)
        1: De 1 a 10 personas      
        2: De 11 a 20 personas
        3: Más de 20 personas

        Índice de Frecuencia y Duración de la exposición (self.IFDE): Considera dos variables, “Frecuencia” y “Duración” de la exposición según la siguiente tabla:
        Frecuencia de exposición del personal Duración de la exposición Índice 
        4: DIARIA o todos los días Menos del 50% del turno de trabajo
        5: DIARIA o todos los días Más del 50% del turno de trabajo
        3: SEMANALMENTE o todas las semanas (pero no se reitera cada día) Menos del 50% del turno de trabajo
        4: SEMANALMENTE o todas las semanas (pero no se reitera cada día) Más del 50% del turno de trabajo
        2: MENSUALMENTE (pero no se reitera cada semana) a SEMESTRALMENTE Menos del 50% del turno de trabajo 
        3: MENSUALMENTE (pero no se reitera cada semana) a SEMESTRALMENTE Más del 50% del turno de trabajo
        1: Se realiza con una frecuencia mayor a la SEMESTRAL Menos del 50% del turno de trabajo
        2: Se realiza con una frecuencia mayor a la SEMESTRAL Más del 50% del turno de trabajo
        2: Se está evaluando en Condiciones de Falla/no rutinaria o Anormales ó sólo se exponen equipos, no personas 

        Índice de Controles existentes (self.ICO)
        1: Existen controles implementados → Se tienen al menos tres condiciones implementadas (A, B, C ó D) 
        6: Existen controles parciales → Se tienen dos condiciones implementadas
        10: Los controles son bajos o insuficientes o no están implementados → 
        * Se cumple sólo una condición implementada o ninguna ó 
        * se identificaron "permanentes" desvíos comportamentales de incumplimiento a las normas referidas al peligro ó 
        * Es una evaluación "IPER anticipada para actividades" 

            Condiciones 
            Condicion A.  Existen ya implementados como parte del SGI procedimientos/documentos o controles operativos específself.ICOs para controlar el Peligro (p.e. PT, EPP, bloqueo/etiquetado, IT, guías, normas, inspecciones específicas).
            Condicion B.  Para el peligro específself.ICO existen implementados medios de infraestructura o ingeniería o protección colectiva o de emergencia para el Peligro (resguardos, barandas, cubiertas, líneas de vida, barreras de protección, aislamientos, medios de detección/alarma del peligro, dispositivos de enclavamiento/corte automátself.ICO, extintores, accesorios para mejorar la ergonomía, mantenimiento preventivo).
            Condicion C.  El personal ya tiene experiencia o fue capacitado/entrenado sobre el peligro específself.ICO al realizar la actividad.
            Condicion D. Se cuenta con señalización/alerta específica in-situ implementada para el peligro o situación peligrosa.

        Criterios de Severidad de Daño:
        1: "Daño Menor"
            - Lesiones que sólo requieren primeros auxilios o atención médica de seguimiento.
            - Lesiones/enfermedades que ocasionan ausencia laboral de menos de un día o transferencia de actividad por ese período.
            - Daños a equipos con costos estimados menores a 1,000 USD.

        2: "Daño Mediano"
            - Lesiones/enfermedades que ocasionan ausencia laboral temporal de 1 día a 1 mes o transferencia de actividad por ese período.
            - Daños a equipos con costos entre 1,001 y 10,000 USD.

        3: "Daño Mayor"
            - Lesiones que ocasionan ausencia laboral temporal de 1 mes a 6 meses o transferencia de actividad por ese período.
            - Lesiones/enfermedades que ocasionan "incapacidades permanentes parciales".
            - Daños a equipos con costos entre 10,001 y 100,000 USD.

        4: "Daño Extremo"
            - Lesiones/enfermedades que ocasionan ausencia laboral temporal de 6 meses a 1 año o transferencia de actividad por ese período.
            - Lesiones o enfermedades que generan "incapacidad total" al trabajador.
            - Muerte.
            - Daños a equipos con costos estimados superiores a 100,000 USD.
    '''
    def __init__(self, sector, subsector,  problem_description):

        self.df = pd.DataFrame(columns=[
            'SECTOR / AREA / UNIDAD / PROCESO O SUB-PROCESO',
            'ACTIVIDAD / TAREA / LUGAR / EQUIPO / EVENTO',
            'CONDICIÓN DE EVALUACIÓN',
            'PELIGRO_x000D_\n_x000D_\n(Evento Peligroso, categorias de Lista Maestra OST-SGI.SI.001)',
            'CONSECUENCIAS_x000D_\nmas probables_x000D_\n_x000D_\n(Lesiones o daños mas probables)',
            'DESVIO O CAUSA QUE ORIGINA EL PELIGRO_x000D_\n¿Por qué se genera el peligro?_x000D_\n_x000D_\nCausas/Devío: (Condiciones inseguras / Factores inseguros del Trabajo / Deficiencias de seguridad/ Actos Inseguros que generan el peligro)_x000D_\n_x000D_\nElementos: (Energías, equipos, maquinarias, sustancias, etc.)',
            'DETERMINACIÓN DE LOS CONTROLES O PROTECCIÓNES EXISTENTES_x000D_\n_x000D_\n(Determine aquellos controles preventivos o de protección actualmente existentes)',
            'SEVERIDAD_x000D_\n_x000D_\nDEL DAÑO',
            'SEVERIDAD_x000D_\n_x000D_\nDEL DAÑO.1', 
            'INPE', 
            'IFDE', 
            'ICO',
            '∑', 
            'Prob.', 
            'Prob..1', 
            'NIVEL DEL RIESGO',
            'ACEPTABLE / NO ACEPTABLE', 
            'ES REQUISITO LEGAL? CUAL',
            'MEDIDAS ADICIONALES DE CONTROL O PROTECCIÓN  PROPUESTAS_x000D_\n_x000D_\n(Debe atacar las CAUSAS identificadas)_x000D_\n_x000D_\n1. Eliminacion: _x000D_\n2. Sustitucion: _x000D_\n3.Controles Ingenieria/Protección colectiva:_x000D_\n4. Señalización/Controles administrativos: _x000D_\n5. EPP:',
            'SEVERIDAD RESIDUAL _x000D_\nDEL DAÑO_x000D_\n_x000D_\n(analizar si hay disminución)',
            'SEVERIDAD RESIDUAL _x000D_\nDEL DAÑO_x000D_\n_x000D_\n(analizar si hay disminución).1',
            'INPE.1', 
            'IFDE.1', 
            'ICO.1', 
            '∑.1', 
            'Prob..2', 
            'Prob..3',
            'NUEVO NIVEL DEL RIESGO', 
            'ACEPTABLE / NO ACEPTABLE.1'])
            
        self.df.index.name = '#'

        self.opciones_peligro = {
            1: "A1. Caída de personas al mismo nivel",
            2: "A2. Caídas menores a distinto nivel (entre 0,3 y 1,8 m)",
            3: "A3. Caídas mayores a distinto nivel (mayor a 1,8 m)",
            4: "A4. Contactos eléctricos (Choque eléctrico)",
            5: "A5. Contactos con partes o elementos calientes/fríos",
            6: "A6. Proyección de partículas, fragmentos",
            7: "A7. Proyección de gases, polvo o líquidos a presión ó calientes",
            8: "A8. Atrapamientos mecánicos",
            9: "A9. Cortes, golpes, penetraciones por herramientas",
            10: "A10. Cortes, golpes, penetraciones, excoriaciones de otra clase (no por herramientas)",
            11: "A11. Caída de objetos menores (menos de 5 kg) o herramientas",
            12: "A12. Aplastamiento/Ahogamiento (entre objetos o por caída/deslizamiento de objetos mayores a 5 Kg)",
            13: "A13. Golpes por objetos/equipos móviles o atropellamiento por vehículos",
            14: "A14. Golpes por objetos inmóviles o partes salientes",
            15: "A15. Incendios",
            16: "A16. Explosiones / deflagraciones",
            17: "A17. Choques de vehículos en movimiento",
            18: "A18. Vuelcos vehiculares o de equipo",
            19: "A19. Exposición a ruido",
            20: "A20. Exposición a vibraciones",
            21: "A21. Exposición a inadecuada iluminación",
            22: "A22. Exposición a temperaturas extremas (extremadamente mayor a la normal o menor a 0°C)",
            23: "A23. Exposición a humedad extrema",
            24: "A24. Exposición a radiaciones ionizantes",
            25: "A25. Exposición a radiaciones no ionizantes",
            26: "B1. Contacto o ingestión de sólidos/líquidos peligrosos",
            27: "B2. Exposición a polvos o fibras",
            28: "B3. Exposición a gases/vapores tóxicos o asfixiantes",
            29: "B4. Derrames o fugas mayores de sustancias peligrosas",
            30: "B5. Exposición a insectos/animales peligrosos",
            31: "B6. Exposición a bacterias, virus u hongos",
            32: "C1. Ejecución de posturas inadecuadas",
            33: "C2. Ejecución de movimientos repetitivos",
            34: "C3. Ejecución de sobre esfuerzo físicos",
            35: "C4. Exposición a sobre esfuerzo visual",
            36: "C5. Exposición a sobre esfuerzo mental",
            37: "D1. Sismos",
            38: "D2. Inundaciones (por lluvias o granizadas intensas o desborde de ríos)",
            39: "D3. Tormentas eléctricas o de vientos huracanados",
            40: "D4. Deslizamientos de tierra",
            41: "D5. Incendios de plantas aledañas o forestales",
            42: "D6. Convulsión social"
        }

        self.sector = sector
        self.subsector = subsector 
        self.problem_description = problem_description 
        self.controles_implementados = self._get_controles_implementados()

        # Indices
        self.INPE = self._get_inpe()
        self.IFDE = self._get_ifde() 
        self.ICO = self._get_ico()

        self.condicion_evaluacion = self._elegir_condicion()
        self.peligro = self._seleccionar_peligro()
        self.origen_peligro = self._generar_origen_peligro()

        self.severidad_daño = None
        self.string_severidad_daño = self._get_severidad_daño()

        self.string_probabilidad = None
        self.probabilidad = self._evaluar_probabilidad(self.INPE, self.IFDE, self.ICO)
        self._set_string_probabilidad()

        self.nivel_de_riesgo = self._calcular_nivel_riesgo(self.probabilidad, self.severidad_daño)
        self.aceptabilidad = self._evaluar_aceptabilidad(self.probabilidad, self.severidad_daño)

        print('El riesgo se considera:', self.aceptabilidad)

        # Nuevos Indices 
        self.controles_adicionales = self._get_controles_adicionales(self.controles_implementados)
     
        self.nueva_INPE = self.INPE 
        self.nueva_IFDE = self.IFDE 
        self.nueva_ICO = self._get_nueva_ico()
        self.nueva_probabilidad = self._evaluar_probabilidad(self.nueva_INPE, self.nueva_IFDE, self.nueva_ICO)

        self.nueva_severidad_daño = self.severidad_daño
        self.string_nueva_severidad_daño = self.string_severidad_daño

        self.nueva_string_probabilidad = None
        self._set_string_nueva_probabilidad()
        self.nuevo_nivel_de_riesgo = self._calcular_nivel_riesgo(self.nueva_probabilidad, self.nueva_severidad_daño)
        self.nueva_aceptabilidad =self._evaluar_aceptabilidad(self.nueva_probabilidad, self.nueva_severidad_daño)

        self._fill_data()

        print('El riesgo se considera:', self.nueva_aceptabilidad)

    def _evaluar_probabilidad(self, inpe, ifde, ico):
        '''
        Evalúa la probabilidad de riesgo en función de los indicadores.

        Args:
            self.INPE (int): Índice de número de personas expuestas. (0-3)
            self.IFDE (int): Índice de Frecuencia y Duración de la exposición. (1-5)
            self.ICO (int): Índice de Controles existentes. (1, 6, 10)

        Returns:
            int: Valor de la probabilidad de riesgo (1-5) o None si no se cumple ninguna condición.
        '''
        suma = inpe + ifde + ico
        if suma <= 6:
            return 1
        elif suma <= 10:
            return 2
        elif suma <= 14:
            return 3
        elif suma <= 18:
            return 4
        elif suma > 18:
            return 5
        else:
            return 0

    def _set_string_probabilidad(self):
        match self.probabilidad:
            case 1:
                self.string_probabilidad = 'Muy baja'
            case 2:
                self.string_probabilidad = 'Baja'
            case 3:
                self.string_probabilidad = 'Media'
            case 4:
                self.string_probabilidad = 'Alta'
            case 5:
                self.string_probabilidad = 'Muy Alta'
            case _:
                self.string_probabilidad = '0'

    def _set_string_nueva_probabilidad(self):
        match self.probabilidad:
            case 1:
                self.nueva_string_probabilidad = 'Muy baja'
            case 2:
                self.nueva_string_probabilidad = 'Baja'
            case 3:
                self.nueva_string_probabilidad = 'Media'
            case 4:
                self.nueva_string_probabilidad = 'Alta'
            case 5:
                self.nueva_string_probabilidad = 'Muy Alta'
            case _:
                self.nueva_string_probabilidad = '0'

    def _calcular_nivel_riesgo(self, probabilidad, severidad_daño):
        '''
        Calcula el nivel de riesgo en función de la probabilidad y la severidad del daño.

        Args:
            self.probabilidad (int): Valor de la probabilidad de riesgo (1-5).
            self.severidad_daño (int): Valor de la severidad del daño (1-5).

        Returns:
            str: Nivel de riesgo ("Trivial", "Bajo", "Moderado", "Alto" o "Intolerable").
        '''
        resultado = probabilidad * severidad_daño
        nivel = ""
        if resultado <= 2:
            nivel = "Trivial"
        elif resultado <= 6:
            nivel = "Bajo"
        elif resultado <= 9:
            nivel = "Moderado"
        elif resultado <= 15:
            nivel = "Alto"
        elif resultado >= 16:
            nivel = "Intolerable"

        return nivel

    def _evaluar_aceptabilidad(self, probabilidad, severidad_daño):
        '''
        Evalúa la aceptabilidad del riesgo en función de la probabilidad y la severidad del daño.

        Args:
            self.probabilidad (int): Valor de la probabilidad de riesgo (1-5).
            self.severidad_daño (int): Valor de la severidad del daño (1-5).

        Returns:
            str: Aceptabilidad del riesgo ("Aceptable" o "No Aceptable").
        '''
        resultado = probabilidad * severidad_daño
        nivel = ""
        if resultado > 7:
            nivel = "No Aceptable"
        else:
            nivel = "Aceptable"

        return nivel

    def _evaluar_aceptabilidad_nueva(self):
        '''
        Evalúa la aceptabilidad del riesgo en función de la probabilidad y la severidad del daño.

        Args:
            self.probabilidad (int): Valor de la probabilidad de riesgo (1-5).
            self.severidad_daño (int): Valor de la severidad del daño (1-5).

        Returns:
            str: Aceptabilidad del riesgo ("Aceptable" o "No Aceptable").
        '''
        resultado = self.nueva_probabilidad * self.severidad_daño
        nivel = ""
        if resultado > 7:
            nivel = "No Aceptable"
        else:
            nivel = "Aceptable"

        return nivel

    def _elegir_condicion(self):

        if self.IFDE == 2:
            return 'Anormal/Emergencia/No rutinaria'
        else:
            return 'Normal/Rutinaria'
            

    def _seleccionar_peligro(self):
        # Define the prompt using self.problem_description and opciones_peligro
        prompt = f"""Selecciona el peligro más adecuado lista de opciones de peligro basado en la siguiente descripción:
        Devuelve solo el indice correspondiente, un entero.

        Descripción del problema: {self.problem_description}

        Opciones de peligro:
        {self.opciones_peligro}
        """

        # Generate a response using the ChatGPT API
        messages = [{"role": "user", "content": prompt}]
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=1,
            n=1,
            stop=None,
            temperature=0
        )

        # Extract the selected option (key) from the generated response
        selected_option = int(response.choices[0].message["content"].strip())

        return selected_option

    def _generar_origen_peligro(self):
        # Llamar a la API de OpenAI para generar un objetivo conciso para el peligro dado
        prompt = f"""
        Genera una posible causa del siguiente peligro de salud y seguridad en el trabajo de manera forta y consisa: {self.peligro}.

        No empezar el texto con saltos de linea, signos de putnuación ni similar.
        Responde en una oracion de maximo 6 palabras.
        """
        
        messages = [{"role": "user", "content": prompt}]
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=80,
            n=1,
            stop=None,
            temperature=0.7
        )
        
        origen_peligro = response.choices[0].message["content"].strip()

        return origen_peligro
    
    def _get_severidad(self):
        danio_menor = [1, 2, 5, 9, 10, 14, 19, 20, 21, 22, 23, 25, 29, 30, 31, 32, 33, 34, 35, 36, 39, 40, 41, 42]
        danio_mediano = [3, 4, 7, 8, 13, 17, 18, 24, 27, 37]
        danio_mayor = [6, 11, 16, 26, 38]
        danio_extremo = [15, 25]
        if self.peligro in danio_menor:
            return 1 
        elif self.peligro in danio_mediano:
            return 2 
        elif self.peligro in danio_mayor:
            return 3 
        elif self.peligro in danio_extremo:
            return 4
        else:
            return 0
    
    def _get_severidad_daño(self):
        daño_menor = [1, 2, 5, 9, 10, 14, 19, 20, 21, 22, 23, 25, 29, 30, 31, 32, 33, 34, 35, 36, 39, 40, 41, 42]
        daño_mediano = [3, 4, 7, 8, 13, 17, 18, 24, 27, 37]
        daño_mayor = [6, 11, 16, 26, 38]
        daño_extremo = [15, 25]

        if self.peligro in daño_menor:
            self.severidad_daño = 1
            return 'daño menor'
        elif self.peligro in daño_mediano:
            self.severidad_daño = 2
            return 'daño mediano'
        elif self.peligro in daño_mayor:
            self.severidad_daño = 3
            return 'daño mayor'
        elif self.peligro in daño_extremo:
            self.severidad_daño = 4
            return 'daño extremo'
        else:
            return 0
    
    def _get_controles_implementados(self):
        controles = [] 
        respuesta = questionary.select(
                "¿Existen medidas de control preexistentes?",
                choices=["Si", "No"]
                ).ask()
        if  respuesta == "Si":
            while True:
                control = input('Agrega una medida de control preexistente: ')
                controles.append(control)
                respuesta = questionary.select(
                    "¿Existen otras mediadas de control preexistentes?",
                    choices=["Si", "No"]
                    ).ask()
                if respuesta == "No":
                    break
        return controles

    def _get_inpe(self):
        opciones_cantidad = [
            'No aplica (NO existe exposición de personas, sólo de equipos/infraestructura',
            'De 1 a 10 personas',
            'De 11 a 20 personas',
            'Más de 20 personas'
            ]

        cantidad = select('¿Cuantas personas trabajan en estan involucradas en esta actividad?', choices=opciones_cantidad).ask()

        if cantidad == 'De 1 a 10 personas':
            return 1
        elif cantidad == 'De 11 a 20 personas':
            return 3
        elif cantidad == 'Más de 20 personas':
            return 5
        else:
            return None
        
    def _get_ifde(self):
        opciones_frecuencia = [
        'DIARIA o todos los días',
        'SEMANALMENTE o todas las semanas (pero no se reitera cada día)',
        'MENSUALMENTE (pero no se reitera cada semana) a SEMESTRALMENTE',
        'Se realiza con una frecuencia mayor a la SEMESTRAL',
        'Se está evaluando en Condiciones de Falla/no rutinaria o Anormales ó sólo se exponen equipos, no personas'
        ]

        opciones_duracion = [
            'Menos del 50% del turno de trabajo',
            'Más del 50% del turno de trabajo'
            ]

        frecuencia = select('¿Con que frecuencia se realiza la actividad?', choices=opciones_frecuencia).ask()

        if frecuencia == 'DIARIA o todos los días':
            duracion = select('¿Durante cuanto tiempo?', choices=opciones_duracion).ask()
            if duracion == 'Menos del 50% del turno de trabajo':
                return 4
            else:
                return 5
        elif frecuencia == 'SEMANALMENTE o todas las semanas (pero no se reitera cada día)':
            duracion = select('¿Durante cuanto tiempo?', choices=opciones_duracion).ask()
            if duracion == 'Menos del 50% del turno de trabajo':
                return 3
            else:
                return 4
        elif frecuencia == 'MENSUALMENTE (pero no se reitera cada semana) a SEMESTRALMENTE':
            duracion = select('¿Durante cuanto tiempo?', choices=opciones_duracion).ask()
            if duracion == 'Menos del 50% del turno de trabajo':
                return 2
            else:
                return 3
        elif frecuencia == 'Se realiza con una frecuencia mayor a la SEMESTRAL':
            duracion = select('¿Durante cuanto tiempo?', choices=opciones_duracion).ask()
            if duracion == 'Menos del 50% del turno de trabajo':
                return 1
            else:
                return 2
        elif frecuencia == 'Se está evaluando en Condiciones de Falla/no rutinaria o Anormales ó sólo se exponen equipos, no personas':
            return 2
        else:
            return None

    def _get_ico(self):
        x = len(self.controles_implementados)
        if x >= 3:
            return 1 
        elif x == 2:
            return 6
        elif x == 1:
            return 10
    
    def _get_nueva_ico(self):
        x = len(self.controles_adicionales)
        if x >= 3:
            return 1 
        elif x == 2:
            return 6
        elif x == 1:
            return 10
 
    def _get_controles_adicionales(self, controles_previos):
        new_list = []
        new_list.extend(controles_previos)

        while True:
            respuesta = questionary.select(
                "¿Deseas agregar alguna nueva medida de control para disminuir el riesgo?",
                choices=["Si", "No"]
                ).ask()
            if respuesta == "No":
                break
            control = input('Agrega una nueva medida de control: ')
            new_list.append(control)
        return " ".join(new_list)

    def _fill_data(self):
        '''
        Genera la fila del riesgo
        '''
        row = [
            self.sector,
            self.subsector,
            self.condicion_evaluacion,
            self.opciones_peligro.get(self.peligro), 
            self.problem_description,
            self.origen_peligro, #TODO
            self.controles_implementados[0],
            self.string_severidad_daño,
            self.severidad_daño,
            self.INPE,
            self.IFDE,
            self.ICO,
            self.INPE+self.IFDE+self.ICO,
            self.string_probabilidad,
            self.probabilidad,
            self.nivel_de_riesgo,
            self.aceptabilidad,
            None, #Esquema legal
            self.controles_adicionales,
            self.string_nueva_severidad_daño,
            self.nueva_severidad_daño, 
            self.nueva_INPE, 
            self.nueva_IFDE, 
            self.nueva_ICO, 
            self.nueva_INPE+self.nueva_IFDE+self.nueva_ICO,
            self.nueva_string_probabilidad,
            self.nueva_probabilidad, 
            self.nuevo_nivel_de_riesgo,
            self.nueva_aceptabilidad 
            ]
        self.df.loc[0] = row

if __name__ == "__main__":
    filas = []

    while True:
        sector = input("Define el sector: ")
        subsector = input("Define el área: ")
        problem_description = input("Define la descripción del problema: ")

        iper_row = IPER_Row(sector, subsector, problem_description).df

        filas.append(iper_row)

        respuesta = questionary.select(
            "¿Desea agregar otro riesgo?(Caso contrario se generara el archivo xlsx de la IPER)",
            choices=["Si", "No"]
        ).ask()

        if respuesta == "No":
            break
    if len(filas)>=1:
        iper = pd.concat(filas, axis=0)
    else:
        iper=filas[0]
    #Guardar la instancia en un archivo Excel
    filename = 'matriz_iper.xlsx'
    iper.to_excel(filename, index=True)
    print(f"Matriz de Identificación de Peligros y la Evaluación de Riesgos(IPER) {filename}")