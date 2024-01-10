from langchain.chains import LLMChain
from src.dorahLLM.maritalkllm import MariTalkLLM
from langchain.prompts import PromptTemplate
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain.llms.fake import FakeListLLM
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType


def generate_topics_from_text(input_text: str) -> list[str]:
    if input_text == "":
        return [""]

    template = """Você faz uma lista só com os tópicos das partes principais do texto.

Texto: "Os polímeros são macromoléculas constituídas por unidades menores, os monômeros.
Um polímero pode ser de origem natural ou de origem sintética. Os polímeros naturais (ou biopolímeros) são os que ocorrem na natureza, como é o caso das proteínas e dos polissacarídeos (como amido, glicogênio e celulose). Já os polímeros sintéticos, como o nome faz alusão, são os polímeros produzidos em laboratório, sendo a maioria de origem petrolífera.
Os polímeros podem ser classificados quanto ao número de monômeros, temos o homopolímero que é o polímero derivado de apenas um tipo de monômero. Já o copolímero é polímero derivado de dois ou mais tipos de monômeros.
"

Lista: Origem natural, Origem sintética, Homopolímeros, Copolímeros

Texto:"A queda da Bastilha, em 14 de julho de 1789, é considerada o evento que deu início à Revolução Francesa. A Bastilha era uma grande prisão considerada símbolo do Antigo Regime, e sua tomada pela população de Paris espalhou a revolução por toda a França. Esse alastramento da revolta popular resultou em grandes transformações sociais e políticas no país e marcou a queda do Antigo Regime. A tomada da Bastilha foi consequência da grave crise que a sociedade francesa enfrentava no final do século XVIII.
A Revolução Francesa foi resultado da grave crise que a sociedade francesa enfrentava nas décadas de 1770 e 1780. Nesse período, a França possuía uma monarquia absolutista e era governada por Luís XVI, que concentrava todo o poder do Estado francês. Além disso, a sociedade francesa era dividida em três grandes grupos, conhecidos como estados.
 O Primeiro Estado representava o clero e os representantes da sociedade vinculados à Igreja. O Segundo Estado representava a aristocracia francesa, isto é, a nobreza, que mantinha uma série de privilégios e viviam à custa do Estado e de seus feudos. Essas duas classes, juntas, impunham uma grande opressão e exploração sobre o Terceiro Estado, o qual agrupava o povo em geral e formava cerca de 95% da população francesa|1|
"

Lista: Revolução Francesa, Transformações sociais, Transformações políticas, Monarquia absolutista

Texto: "A coesão textual é a conexão linguística que permite a amarração das ideias dentro de um texto. Um conjunto aleatório de palavras e mesmo de frases não constitui um texto. Ou seja, para que algum material linguístico possa ser reconhecido como texto e possa funcionar comunicativamente, são necessários certos critérios de organização desse material. 
O vínculo que existe entre palavras, orações e as diferentes partículas do texto por meio de um referente, isso é chamado de coesão referencial.
Nesse tipo de coesão, os elementos de coesão anunciam, ou retomam as frases, sequências e palavras que indicam conceitos e fatos.
Já a coesão sequencial é a maneira como os fatos se organizam no tempo do texto. Para isto, são utilizadas relações semânticas que ligam as orações e os parágrafos à medida que o texto é descrito.
Referências bibliográficas:
ANTUNES, I. Lutar com palavras – coesão e coerência. São Paulo: Parábola Editorial, 2005.
KOCH, I. A coesão textual. São Paulo: Contexto, 1989.
KOCH, I.; TRAVAGLIA, L. C. Texto e coerência, São Paulo: Contexto, 1991.
MARCUSCHI, L. A. Linguística de texto – como é, como se faz. Recife, Editora da UFPE, 1983.
"

Lista: Oraçôes, Conectivos, Referecia, Relações semânticas

Texto: "As etapas da Evolução Humana. Primatas: Os mais antigos viveram há cerca de 70 milhões de anos. Os pré-australopitecos.Essas primeiras espécies viveram logo após a separação do grupo que originou os hominídeos e os chimpanzés.
O registro fóssil remonta algumas das espécies desse período:
Sahelantropus tchadensis: Fóssil encontrado no continente africano, pertencente a uma espécie de primata. 
Orrorin tugenensis: Fóssil encontrado no Quênia. Também já apresentava indicações da postura bípede.
As principais características dos australopitecos eram: a postura ereta, a locomoção bípede, a dentição primitiva e a mandíbula mais semelhante a da espécie humana.
A extinção da maioria dos australopitecos possibilitou o surgimento de uma nova linhagem.
O gênero Homo se destaca pelo desenvolvimento do sistema nervoso e da inteligência. A espécie Homo neanderthalensis foi descoberta em 1856, quando fósseis, datados de 40.000 anos, foram encontrados em uma caverna da Alemanha.
O Homo sapiens sapiens é a denominação científica do homem moderno, sendo uma subespécie do Homo sapiens. o crânio do ser humano moderno apresenta 1400 cm3. 
Luzia foi considerado como a mulher mais antiga das Américas e também a primeira brasileira. Ela foi encontrada no começo dos anos 1970, por uma missão arqueológica realizada por Brasil e França e o seu fóssil
"

Lista: Pré-australopitecos, Australopitecos,  Gênero Homo, Homo sapiens

Texto: "A Geometria Analítica estuda elementos geométricos em um sistema de coordenadas num plano ou espaço. Estes objetos geométricos são determinados por sua localização e posição em relação a pontos e eixos deste sistema de orientação.
O Sistema Cartesiano Ortogonal é uma base de referência para localização de coordenadas. A Geometria Analítica surgiu com René Descartes em 1637, o mesmo matemático e filósofo que criou o sistema de coordenadas no plano cartesiano. No século XVII, Descartes relacionou a álgebra com a geometria.
Par ordenado.
Um ponto qualquer no plano possui a coordenada P(x, y). Por exemplo o ponto (1,2)
Distância entre dois pontos
A distância entre dois pontos no plano cartesiano é o comprimento do segmento que une estes dois pontos. Se tivermos como base os pontos A (xa, ya) e B (xb, yb), a distância é representada pelo segmento de reta AB (dAB). 
Coordenadas do ponto médio
Ponto médio é o ponto que divide um segmento em duas partes de mesma medida.
Coeficiente angular de uma reta
O coeficiente angular reto m de uma reta é a tangente de sua inclinação alfa em relação ao eixo x
"

Lista: Elementos geométricos, Sistema cartesiano, Ponto médio, Coeficiente angular

Texto: "{text_object}
"

Lista:"""

    prompt = PromptTemplate.from_template(template)
    model = MariTalkLLM()
    chain = LLMChain(prompt=prompt, llm=model)
    output_date = chain(inputs={"text_object": input_text})
    origin = output_date["text"]

    output_parser = CommaSeparatedListOutputParser()
    output = output_parser.parse(origin)
    if len(output) > 6:
        output = output[:6]

    print(output)
    return output


def generate_topics_from_text_test(input_text: str) -> list[str]:
    if input_text == "" or input_text == "Texto incoerente.":
        return [""]

    responses = [
        "Final Answer: Revolução Liberal do Porto, Cortes Gerais e Extraordinárias, Dom Pedro I, Autonomia das províncias, Assembleia Constituinte, Tratado de paz."
    ]
    llm = FakeListLLM(responses=responses)
    tools = load_tools(["python_repl"])
    agent = initialize_agent(
        tools=tools, llm=llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION
    )
    origin = agent.run("Você faz um resumo do texto")
    output_parser = CommaSeparatedListOutputParser()
    output = output_parser.parse(origin)
    return output
