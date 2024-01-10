import json

# LLMChain
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate, LLMChain
from langchain.chains import SequentialChain

import logging

LOG = logging.getLogger("LLM Extract")
logging.basicConfig(level=logging.INFO)

# Cadena de las entidades nombradas


## Prompt
TEMPLATE_NER = """
Como periodista y científico de redes, está analizando la red social que surgió de un Escándalo de corrupción en Colombia.
Su tarea es inferir y reconocer las entidades nombradas de organizaciones e individuos de la noticia entre ```.
Algunas de las entidades no son explicitas, por lo que debe inferirlas a partir de la información de la noticia.

La salida solo deberia ser en formato JSON con las siguientes claves:

"Organizaciones": Lista de Nombres de Organizaciones 
"Individuos": Lista de diccionarios con los nombres de los individuos y su rol en el escándalo:
    "Nombre"
    "Cargo"
    "rolEnEscandalo"
"Relaciones": Lista de triplas de las relaciones entre individuos y organizaciones, con la siguiente estructura:
    "Individuo"
    "Relación" // Clasificador de la relación frase verbal + frase preposicional
    "Organización"
    
    
Los clasificadores de relaciones deben tener una frase verbal como ejemplo (nacio) + frase preposicional (EnCiudad) -> nacioenCiudad:\n 

```
{article}
```
"""


def extract_entities(input):
    """
    Extrae las entidades nombradas de una noticia
    """

    PROMPT_NER = PromptTemplate(
        input_variables=["article"],
        template=TEMPLATE_NER,
        # partial_variables={"format_instructions": parserNER.get_format_instructions()},
    )

    ## Definición de la Cadena
    llm = ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0.2, max_tokens=8000)

    llm_chain_NER = LLMChain(
        llm=llm,
        prompt=PROMPT_NER,
        # output_parse_function=parse_output,
    )

    chat_completion_NER = llm_chain_NER(input, return_only_outputs=True)

    return chat_completion_NER


# Extracción de Relaciones
TEMPLATE_ER = """
Como periodista y científico de redes, está analizando la red social que surgió de un Escándalo de corrupción en Colombia.
Su tarea es inferir y reconocer las relaciones entre las entidades nombradas SOLO de los siguientes individuos.
```
{individuos}
```
La salida solo deberia ser una lista en formato JSON con las siguientes claves:

"Relaciones": Lista de triplas de las relaciones entre individuos, con la siguiente estructura:
    "Nombre_Individuo_A": Nombre de los individuos que se encuentran entre ```
    "Detalles Relación": Detalles de la relación
    "Relación": Clasificador de la relación frase verbal + frase preposicional
    "Nombre_Individuo_B": Nombre de los individuos que se encuentran entre ```

Si no hay relación entre dos individuos, no escriba nada.
Los clasificadores de relaciones deben tener una frase verbal como ejemplo (nacio) + frase preposicional (EnCiudad) -> nacioenCiudad:\n 

La tarea se realiza en base al siguiente texto de una noticia:


{article}

"""


def extract_relations(input):
    """
    Extrae las relaciones entre individuos de una noticia
    """

    PROMPT_ER = PromptTemplate(
        input_variables=["individuos", "article"], template=TEMPLATE_ER
    )
    llm = ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0.2, max_tokens=2000)

    llm_chain_ER = LLMChain(
        llm=llm,
        prompt=PROMPT_ER,
        # output_parse_function=parse_output,
    )

    chat_completion_ER = llm_chain_ER(input, return_only_outputs=True)
    return chat_completion_ER


if __name__ == "__main__":
    ## Ejemplo
    input = {
        "article": """
    Luego de una extensa investigación, la Contraloría General de la República profirió un fallo de responsabilidad fiscal por más de 2,9 billones de pesos por el daño producido a los intereses del Estado debido, según se lee en el documento, a las decisiones sobre el manejo de los recursos públicos que "conllevaron a la pérdida de valor de las mayores inversiones en el Proyecto de Ampliación y Modernización de la Refinería de Cartagena ( Reficar )". La decisión se conoció en un auto firmado este 26 de abril por la Contralora Delegada Intersectorial n.° 15 de la Unidad de Investigaciones Especiales contra la Corrupción. (Puede ser de su interés: Recursos por $ 6,6 billones están en juego en obras inconclusas) Es así como, por el mayor daño patrimonial al Estado en su historia, el órgano de control llamó a responder fiscalmente a dos presidentes y tres vicepresidentes de Reficar, siete miembros de su junta directiva (incluyendo el presidente de Ecopetrol para la época de los hechos) y cuatro multinacionales contratistas del proyecto. De acuerdo con la decisión, en la modernización de la Refinería de Cartagena hubo mayores inversiones que no le agregaron valor al proyecto en cuantía de 997 millones de dólares (aproximadamente 2,9 billones de pesos), el monto en que cuantificó finalmente la Contraloría el daño fiscal ocasionado. (Le puede interesar: Contraloría ratifica cuestionamientos a cuentas de Reficar ) Además, el órgano de control encontró que hubo gastos que no tuvieron relación con el proceso de modernización de la Refinería, hubo una baja productividad de la mano de obra directa, mayores gastos en contratación de personal y demoras en las actividades propias del proyecto causadas por acciones y omisiones, por parte de la junta directiva de Reficar, la administración de Reficar, contratistas y supervisor del proyecto. Por ejemplo, se determinó que algunas de estas mayores inversiones consistieron en retrabajos, reprocesos, sobrantes excesivos en inventarios y gastos no asociados a la construcción. Para la entidad, el daño al patrimonio fue producto de las acciones y omisiones de la junta directiva de Reficar y su administración, contratista y supervisor del proyecto, quienes, dice la Contraloría , en ejercicio de la gestión fiscal directa o indirecta, de manera antieconómica, ineficiente e inoportuna, contribuyeron a la billonaria pérdida de recursos públicos. Sanciones por el control de cambio 2 y 3 Facebook Twitter Enviar Linkedin Para la Contraloría, en la modernización de la Refinería de Cartagena hubo mayores inversiones que no le agregaron valor al proyecto en cuantía de 997 millones de dólares. Foto: Yomaira Grandett / EL TIEMPO En primer lugar, por unas adiciones en exceso por 1,3 billones de pesos en el control de cambio 2, se falló responsabilidad fiscal, en forma solidaria y a título de culpa grave  en contra de los funcionarios de Reficar: Carlos Eduardo Bustillo Lacayo , vicepresidente de Proyecto y Asesor de la Vicepresidencia de Proyecto de Reficar para la época de los hechos; Andrés Virgilio Riera Burelli, vicepresidente de proyectos de Reficar para la época; Reyes Reinoso Yanez , presidente y representante legal de Reficar para entonces; Orlando José Cabrales Martínez , también presidente y representante legal para ese momento; Magda Nancy Manosalva Cely , para entonces vicepresidenta financiera y administrativa. También fueron hallados responsables varios miembros de la junta directiva de la Refinería para ese momento: Javier Genaro Gutiérrez Pemberthy , presidente de Ecopetrol; Pedro Alfonso Rosales Navarro , en representación de Ecopetrol; Diana Constanza Calixto Hernández , en representación de Ecopetrol; Henry Medina González , miembro de la junta de Reficar y de Ecopetrol; y Hernando José Gómez Restrepo , miembro de la junta de Reficar. De otro lado, se halló fiscalmente responsables a los contratistas: Chicago Bridge & Iron Company CB&I UK Limited;  CBI Colombiana; Foster Wheeler USA Corporation y Process Consultants Inc. (Lea también: Expertos y analistas escriben sobre caso Reficar) De otro lado, por el daño patrimonial producido al aprobarse adiciones de recursos en exceso en cuantía de 1,6 billones de pesos en el control de cambio 3, se halló fiscalmente responsables, de forma solidaria y a título de culpa grave, a los funcionarios de Reficar para la época: Bustillo Lacayo, Riera Burelli, Reinoso Yanez, y Manosalva Cely. Así mismo, a los miembros de la junta directiva Gutiérrez Pemberthy, Rosales Navarro, Natalia Gutiérrez Jaramillo y Uriel Salazar Duque, ambos miembros de la junta directiva de Reficar. Los mismos cuatro contratistas fueron igualmente hallados responsables en este punto. Finalmente, se declaró como terceros civilmente responsables a las compañías de seguros Compañías Aseguradoras de Fianzas S.A. Confianza, Chubb de Colombia Compañía de Seguros S.A y AXA Colpatria Seguros S.A. Contra este fallo de responsabilidad fiscal, que es de primera instancia, se pueden interponer los recursos de reposición, ante la misma Contralora Delegada n.° 15 de la Unidad de Investigaciones Especiales Contra la Corrupción; y el de apelación ante la Sala Fiscal y Sancionatoria de la Contraloría General de la República. Cuando el fallo esté en firme, se enviará el expediente al Consejo de Estado para que ejerza el control automático e integral de legalidad previsto en el artículo 23 de la Ley 2080 de enero 25 de 2021. (Le recomendamos: Procuraduría archiva caso a exprocurador Carlos G. Arrieta por Reficar) Absueltos de responsabilidad fiscal Mientras en los controles de cambio 2 y 3 la Contraloría encontró responsabilidad fiscal, en el control de cambio 4, por 645.990 millones de pesos no fue así, por lo cual absolvió de responsabilidad fiscal a: Los miembros de la junta directiva de Reficar para la época, Astrid Martínez Ortiz, Carlos Gustavo Arrieta, Gutiérrez Pemberthy, Rosales Navarro, Salazar Duque, Reinoso Yanez, presidente de la junta. Así mismo, en favor de Riera Burelli, como vicepresidente del proyecto de expansión de Ecopetrol. También falló en favor, en este punto, de los contratistas CBI Colombiana, Chicago Bridge & Iron Company CB&I UK Limited; Foster Wheeler USA Corporation y Process Consultants Inc. Fueron absueltos de responsabilidad igualmente, por los controles de cambio 2, 3 y 4  Chicago Bridge & Iron Company (CB&I) Américas Ltd. y César Luis Barco García, director Corporativo de Proyectos de Ecopetrol. justicia@eltiempo.com En Twitter: @JusticiaET
        """
    }

    json_NER = extract_entities(input)

    individuos = [item["Nombre"] for item in json_NER["Individuos"]]
    individuos = ", ".join(individuos)
    individuos

    input.update({"individuos": individuos})

    json_ER = extract_relations(input)

    print(json_ER)
