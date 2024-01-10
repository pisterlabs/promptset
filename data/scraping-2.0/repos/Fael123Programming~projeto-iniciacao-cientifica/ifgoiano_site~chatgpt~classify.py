import openai, pandas, os, time
from decouple import config


# Generate class, uncomment the following variables.
SOURCE_PATH = 'C:\\Users\\Rafae\\OneDrive\\Documentos\\projeto_iniciacao_cientifica\\ifgoiano_site\\datasets\\clean'
TARGET_PATH = 'C:\\Users\\Rafae\\OneDrive\\Documentos\\projeto_iniciacao_cientifica\\ifgoiano_site\\datasets\\class'

# Generate inter_class, uncomment the following variables.
# SOURCE_PATH = 'C:\\Users\\Rafae\\OneDrive\\Documentos\\projeto_iniciacao_cientifica\\ifgoiano_site\\datasets\\inter'
# TARGET_PATH = 'C:\\Users\\Rafae\\OneDrive\\Documentos\\projeto_iniciacao_cientifica\\ifgoiano_site\\datasets\\inter_class'

DICIO_RAFAEL = {
    "ACOES_DE_EXTENSAO": "Ações de Extensão: A Extensão no IF Goiano é entendida como um processo educativo, cultural, científico, tecnológico, social e político que promove a interação dialógica e transformadora entre IF Goiano, instituições parceiras e sociedade, articulando o conhecimento gerado pela pesquisa, ensino e extensão com as demandas emanadas de diferentes segmentos sociais na perspectiva do desenvolvimento economicamente viável, socialmente justo e ambientalmente sustentável, considerando sempre a territorialidade.",
    "COMUNICADOS_OFICIAIS": "Comunicados Oficiais: A comunicação escrita oficial é a forma pela qual se redigem as correspondências e os atos administrativos no serviço público.",
    "CONQUISTAS_DE_MEMBROS": "Conquistas de Membros: As conquistas de membros do IF Goiano são premiações concedidas a estudantes, professores e servidores em eventos oficiais e demonstram a qualidade do ensino da Instituição.",
    "CAMPANHAS": "Campanhas: As campanhas do IF Goiano são um conjunto de ações planejadas pela instituição com o propósito de alcançar um objetivo específico relativo à mesma, como conscientização de seus membros, etc.",
    "EDITAIS": "Editais: São divulgações oficiais que, vinculadas em local público ou difundidas em jornal, contêm um anúncio de um concurso, exame de seleção, licitação ou concorrência, para o conhecimento das pessoas interessadas.",
    "PROCESSOS_SELETIVOS": "Processos seletivos: São uma iniciativa de recrutamento e seleção com o objetivo de identificar, entre candidatos para uma vaga, o que mais se enquadra ao cargo.",
    "PUBLICAÇÕES": "Publicações: São publicações de material literário por membros do IF Goiano.",
    "AVALIAÇÕES": "Avaliações: São processos de análise da qualidade do ensino oferecido no instituto pelos seus alunos e por órgãos do serviço público.",
    "QUESTION_PATTERN": "De acordo com os temas: ações de extensão, comunicados oficiais, conquistas de membros do IF Goiano (premiações), campanhas (covid-19, etc), editais, processos seletivos, publicações (obras literárias, boletins, etc) e avaliações, e baseado nas definições anteriormente feitas, escolha um para o seguinte texto (escreva apenas o tema): '{}' '{}'.",
}

DICIO_REJANE = {
    "ACOES_DE_EXTENSAO": "Ações de Extensão: as ações de Extensão abrangem os conteúdos relacionados à Extensão, sendo publicações sobre Projetos de Extensão, Programas, Arte e Cultura, Eventos, Estágio (convênios com empresas e seleção de estágio), Emprego, Egressos e parcerias com instituições, empresas e comunidade externa.",
    "COMUNICADOS_OFICIAIS": "Comunicados Oficiais: os comunicados oficiais são conteúdos de caráter informativo emitidos pela instituição sobre diversos assuntos institucionais de conhecimento público.",
    "CONQUISTAS_DE_MEMBROS": "Conquistas de membros do IF Goiano (premiações): conteúdos sobre a participação e premiação de discentes e servidores e que tenham relevância sob a perspectiva institucional.",
    "CAMPANHAS": "Campanhas: ações planejadas de divulgação sobre diversos assuntos de interesse institucional e social.",
    "EDITAIS": "Editais: ato administrativo, escrito e oficial para divulgar normas sobre processos específicos. O edital é utilizado por diversos setores do IF Goiano, inclusive para promover o conhecimento dos públicos interessados sobre processos seletivos.",
    "PROCESSOS_SELETIVOS": "Processos Seletivos: Procedimentos utilizados para ranquear e classificar candidatos em seleções de diversos tipos, com destaque para os processos de seleção de ingresso nos cursos.",
    "PUBLICAÇÕES": "Publicações (obras literárias, boletins, etc): conteúdos relacionados à publicação de produções do IF Goiano e boletins de novas aquisições da biblioteca.",
    "AVALIAÇÕES": "Avaliações: publicações de avaliações de diversos tipos (Autoavaliação Institucional, Avaliação Docente, Avaliação das Coordenações de Cursos), resultados de avaliações da Instituição, dos cursos e da comunidade discente (Avaliações do MEC, Reconhecimento de Cursos e Enade, por exemplo) e cronogramas e informes de avaliações/provas dos cursos.",
    "QUESTION_PATTERN": "De acordo com os temas: ações de extensão, comunicados oficiais, conquistas de membros do IF Goiano (premiações), campanhas (covid-19, etc), editais, processos seletivos, publicações (obras literárias, boletins, etc) e avaliações, e baseado nas definições anteriormente feitas, escolha um para o seguinte texto (escreva apenas o tema): '{}' '{}'.",
}

DICIO = DICIO_REJANE

def get_messages(title, description) -> list:
    messages = list()
    for key, value in DICIO.items():
        local_dict = dict()
        local_dict['role'] = 'user'
        if key == 'QUESTION_PATTERN':
            value = value.format(title, description)
        local_dict['content'] = value
        messages.append(local_dict)
    return messages


if __name__ == '__main__':
    key = config('API_KEY_JESMMER')
    files = os.listdir(SOURCE_PATH)
    counter = 1
    for f in files[len(files) - 1:]:
        csv = pandas.read_csv(os.path.join(SOURCE_PATH, f))
        csv = csv.reset_index()
        csv['subject'] = '-'
        print('-' * 100)
        for i in range(csv.shape[0]):
            while True:
                try:
                    title, description = csv.iloc[i][['titulo', 'descricao']]
                    response = openai.ChatCompletion.create(
                        api_key=key,
                        model='gpt-3.5-turbo',
                        messages=get_messages(title, description)
                    )
                    answer = response['choices'][0]['message']['content'].replace('.', '')
                    csv.loc[i, 'subject'] = answer
                    print(f'({counter})')
                    print(f'Title: {title}')
                    print(f'Description: {description}')
                    print(f'Subject: {answer.upper()}')
                    print('-' * 100)
                    if counter % 3000 == 0:
                        time.sleep(60)
                    counter += 1
                except (openai.error.ServiceUnavailableError, Exception) as e:
                    print(f'\'{e.__class__.__name__}\' caught with message \'{e}\'')
                    time.sleep(60)
                else:
                    break
        csv.to_csv(os.path.join(TARGET_PATH, f), encoding='utf-8', index=False)