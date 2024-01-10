import json

from argparse import ArgumentParser

from openai import OpenAI
import tiktoken

from neuroticla.oai.constants import MODEL_TOKENS, MODEL_ABBREV
from neuroticla.oai.tokenize import create_chunks


def add_args(module_name: str, parser: ArgumentParser) -> None:
    parser.add_argument(
        'model', type=str, default=list(MODEL_ABBREV.keys())[0], help='OpenAI Model',
        choices=list(MODEL_ABBREV.keys())
    )


def test_prompt_nf(arg) -> int:
    system = '''
    You are a helpful assistant. You know about social sciences and machine learning. You recently researched the framing of news articles.
    '''
    asistant = '''
    News framing refers to the way media outlets present and structure a news story, 
    which can significantly influence how audiences understand and interpret the information. 
    A frame is a perspective through which a news item is communicated, highlighting certain aspects while excluding others. 
    It is a way to structure the news narrative to provide meaning and context.
    
    Each framing dimension shapes how the audience perceives the story and can lead to different emotional and cognitive responses. 
    For instance, a news story about climate change could be framed in terms of scientific evidence (problem definition), 
    human activity as the cause (causal interpretation), a moral crisis (moral evaluation), 
    and policy change as a solution (treatment recommendation). Another frame might focus on economic implications, 
    highlight natural cycles as a cause, avoid moral judgments, and emphasize technological innovation over policy changes.
    '''
    asistant2 = '''
    You studied four framing dimensions: Economy, Labour Market, Welfare and Security in connection with migration.
    The Economy, Labour Market, Welfare and Security frame have been identified to be among the most relevant frames in migration coverage of teh news media
    
    Economy framing can be observed when economy-related aspects of migration are mentioned:
    - people referred to as being economically motivated to migrate
    - migration as being related to the economic development of the country of destination or origin
    - explicit economic (i.e., financial) consequences for the country of destination, transit or origin
    - any sort of costs for a country that arise from migration
    
    Labour Market framing can be observed when labour market-related aspects of migration are mentioned:
    - participation in the labor market, including actions/laws/plans to prevent/limit immigrant active participation in the labor market
    - mentions of non-active/passive participation or unemployment
    - labour market-related consequences for the migrants’ country of origin due to emigration
    - legal and illegal forms of labour when related to migration
    - fear of foreigners regarding labor market
    - individual migrants mentioning their working situation
    
    Welfare framing can be observed when welfare-related aspects of migration are mentioned:
    - welfare, public or social assistance, social benefits, social care, or social services regading migrations
    - education, child and family support, work related support, pension/retirement, public healthcare, state subsidies food/clothing, public housing, and housing/accommodation organization for refugees/migrants
    - EU support/aid payments for countries to handle migration
    
    Security framing can be obsrved when security-related aspects of migration include references to:
    - security and crime issues when those held responsible are migrants
    - law and order, border security, border control, border protection actions (e.g., fence enforcement, tear gas), and also to the deportation of migrants
    - human trafficking/smuggling, subjective security feeling, police dealings with migrants, (fear of) terror connected to migration
    - fear related to crime/security-related aspects of migration
    - illegal immigration
    '''
    instruct = '''
    Analyze how the following news article was framed, selecting from four possible framing dimensions: Economy, Labour Market, Welfare or Security. 
    You must:
    - Respond in JSON format, with an array of relevant sentences for each framing taken from a given article.
    - each sentence is a JSON object with the "orig" for an original sentence and "trans" for the translated sentence to English, which you should also do.
    - If there are no relevant sentences, then output an empty array.
    - Use short JSON property names: "eco" for Economy, "lab" for Labour Market, "wel" for Welfare and "sec" for Security.
    - Mark the "migration" JSON property with 1 if an article talks about immigrants and with 0 if not,
    - Use the "explain" JSON property with your explanation of why you have chosen those framing dimensions.
    None of the above framing dimensions may be present. No framing is the majority case.
    Here is an article:
    
    SILVIA AVILÉS BARCELONA. La Generalitat, el Ayuntamiento de Salt (Girona) y 
    colectivos de inmigrantes hicieron ayer un llamamiento a la tranquilidad, y 
    reclamaron calma para poder reflexionar sobre la tensión vivida en la localidad 
    durante esta última semana.
    El pasado lunes, unos 200 vecinos de Salt encendieron la llama al manifestarse
    delante del ayuntamiento pidiendo mayor seguridad. El incidente terminó con la
    suspensión del pleno municipal que se retomó el jueves con nuevos altercados. La
    policía tuvo que desalojar a dos vecinos que increparon a la alcaldesa durante
    la aprobación de los presupuestos. Mientras, a las puertas del ayuntamiento,
    inmigrantes y vecinos se peleaban entre sí, unos criticando las acusaciones
    hacia este sector, los otros quejándose de la inseguridad ciudadana. El pleno
    finalizó con la decisión de incorporar tres agentes más de policia y cámaras de
    vídeo-vigilancia en la localidad. La alcaldesa de Salt, Iolanda Pineda (PSC),
    reclamó «recursos para una situación excepcional», dado que Salt es un municipio
    de 31.000 habitantes, el 43% de los cuales son inmigrantes. Además, uno de cada
    cuatro inmigrantes está en el paro. Begoña Sánchez, miembro de SOS Racismo,
    declaró a Ep que es una situación «preocupante» agravada por la crisis. Mohamed
    Atauil, representa una asociación de miles de marroquíes que viven en esta
    localidad. lamenta que la policía sólo registre a los inmigrantes en la calle,
    ya que lo considera una discriminación que los criminaliza. El conseller de
    Interior, Joan Saura,contestó que los agentes tienen «instrucciones clarísimas
    de respeto» a esos colectivos. Pacto por la Inmigración La consellera de Acción
    Social, Carme Capdevila, convocará la comisión de seguimiento del Pacto Nacional
    por la Inmigración para analizar la situación en Cataluña tras los altercados en
    Salt.
    '''

    if arg.model not in MODEL_TOKENS:
        raise RuntimeError('Invalid model specified!')
    client = OpenAI()

    response = client.chat.completions.create(
        model=arg.model,
        # response_format={'type': 'json_object'},
        seed=2611,
        temperature=0.001,
        messages=[
            {"role": "system", "content": system},
            {"role": "assistant", "content": asistant},
            {"role": "assistant", "content": asistant2},
            {"role": "user", "content": instruct}
        ]
    )
    resp: str = response.choices[0].message.content
    obj = json.loads(resp)
    print(json.dumps(obj, indent=2))
    return 0