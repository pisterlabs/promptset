import json
import os
import re
import time
import uuid
from calendar import monthrange
from datetime import date, datetime

import feedparser
import openai
import pytz
import requests
import telepot
from apscheduler.schedulers.background import BackgroundScheduler
from bs4 import BeautifulSoup
from googletrans import Translator
from telepot.loop import MessageLoop
from telepot.namedtuple import InlineKeyboardButton, InlineKeyboardMarkup


# Função para remover tags HTML
def remove_html_tags(text):
    clean = re.compile("<.*?>")
    return re.sub(clean, "", text)


# Configurações do Azure OpenAI RBLN
openai.api_type = "azure"
openai.api_base = "YOUR_AZURE_API_BASE"
openai.api_version = "YOUR_AZURE_API_VERSION"
openai.api_key = "YOUR_AZURE_API_KEY"

# Configurações do Azure OpenAI
RBLN_API_BASE = "YOUR_AZURE_API_BASE"
RBLN_API_KEY = "YOUR_AZURE_API_KEY"

# Configurações do Azure OpenAI para redundancia
RBPS_API_BASE = "YOUR_AZURE_API_BASE"
RBPS_API_KEY = "YOUR_AZURE_API_KEY"


def set_openai_config(api_base, api_key):
    openai.api_type = "azure"
    openai.api_base = api_base
    openai.api_version = "YOUR_AZURE_API_VERSION"
    openai.api_key = api_key


# API para as configurações do Telegram e LinkedIn
TELEGRAM_TOKEN = "YOUR_TELEGRAM_TOKEN"
ACCESS_TOKEN = "YOUR_ACCESS_TOKEN"

# URLs dos feeds RSS
feed_urls = {
    "/startcustomblog": "https://techcommunity.microsoft.com/plugins/custom/microsoft/o365/custom-blog-rss?tid=-177205926965371099&size=100",
    "/startinfrastructure": "https://techcommunity.microsoft.com/plugins/custom/microsoft/o365/custom-blog-rss?tid=5272649121701694560&board=CoreInfrastructureandSecurityBlog&size=25",
    "/startazureaiservices": "https://techcommunity.microsoft.com/plugins/custom/microsoft/o365/custom-blog-rss?tid=3287690017842470215&board=Azure-AI-Services-blog&size=25",
    "/startmicrosoft365": "https://techcommunity.microsoft.com/plugins/custom/microsoft/o365/custom-blog-rss?tid=-7424720648213528660&board=microsoft_365blog&size=25",
    "/startserverless": "https://serverless360.com/feed/",
    "/startgetpratical": "https://getpractical.co.uk/feed/",
    "/startEducatordeveloper": "https://techcommunity.microsoft.com/plugins/custom/microsoft/o365/custom-blog-rss?tid=-3610219301967395228&board=EducatorDeveloperBlog&size=25",
    "/startlandingpage": "https://devblogs.microsoft.com/landingpage/",
    "/startcommandline": "https://devblogs.microsoft.com/commandline/feed/",
    "/startmikefrobbins": "https://mikefrobbins.com/index.xml",
    "/azuregovernanceandmanagement": "https://techcommunity.microsoft.com/plugins/custom/microsoft/o365/custom-blog-rss?tid=-5120206136278231098&board=AzureGovernanceandManagementBlog&size=25",
    "/microsoftentra": "https://techcommunity.microsoft.com/plugins/custom/microsoft/o365/custom-blog-rss?tid=-5120206136278231098&board=Identity&size=25",
    "/infrastructuresecurity": "https://techcommunity.microsoft.com/plugins/custom/microsoft/o365/custom-blog-rss?tid=-5120206136278231098&board=CoreInfrastructureandSecurityBlog&size=25",
    "/securitycomplianceidentity": "https://techcommunity.microsoft.com/plugins/custom/microsoft/o365/custom-blog-rss?tid=-5120206136278231098&board=MicrosoftSecurityandCompliance&size=25",
    "/fasttrackforazure": "https://techcommunity.microsoft.com/plugins/custom/microsoft/o365/custom-blog-rss?tid=-5120206136278231098&board=FastTrackforAzureBlog&size=25",
    "/appsonazureblog": "https://techcommunity.microsoft.com/plugins/custom/microsoft/o365/custom-blog-rss?tid=-5120206136278231098&board=AppsonAzureBlog&size=25",
    "/windowsitpro": "https://techcommunity.microsoft.com/plugins/custom/microsoft/o365/custom-blog-rss?tid=-5120206136278231098&board=Windows-ITPro-blog&size=25",
    "/itopstalkblog": "https://techcommunity.microsoft.com/plugins/custom/microsoft/o365/custom-blog-rss?tid=-5120206136278231098&board=ITOpsTalkBlog&size=25",
    "/adamtheautomator": "https://adamtheautomator.com/feed/",
    "/thelazyadministrator": "https://www.thelazyadministrator.com/feed/",
    "/powershellcommunity": "https://devblogs.microsoft.com/powershell-community/feed/",
    "/powershellteam": "https://devblogs.microsoft.com/powershell/feed/",
}

feed_names = {
    "Custom Blog": "/startcustomblog",
    "Infrastructure": "/startinfrastructure",
    "Azure AI Services": "/startazureaiservices",
    "Microsoft 365": "/startmicrosoft365",
    "Serverless": "/startserverless",
    "GetPratical": "/startgetpratical",
    "Educator Developer": "/startEducatordeveloper",
    "Landing Page": "/startlandingpage",
    "Command Line": "/startcommandline",
    "Mike F Robbins": "/startmikefrobbins",
    "Azure Governance MGMT": "/azuregovernanceandmanagement",
    "Microsoft Entra (Azure AD)": "/microsoftentra",
    "Infrastructure Security": "/infrastructuresecurity",
    "Sec, Compliance Identity": "/securitycomplianceidentity",
    "FastTrack for Azure": "/fasttrackforazure",
    "Apps on Azure Blog": "/appsonazureblog",
    "Windows IT Pro": "/windowsitpro",
    "IT OpsTalk Blog": "/itopstalkblog",
    "Adam the Automator": "/adamtheautomator",
    "The Lazy Administrator": "/thelazyadministrator",
    "PowerShell Community": "/powershellcommunity",
    "PowerShell Team": "/powershellteam",
}



# Configuração inicial
(
    awaiting_confirmation,
    awaiting_schedule,
    selected_article,
    summary,
    article_title,
    article_url,
    global_articles,
    translated_article_title,
    selected_day,
    selected_month,
    selected_year,
) = (None, None, None, None, None, None, None, None, None, None, None)

translator = Translator()
scheduler = BackgroundScheduler()
scheduler.start()
global_jobs = {}


# Função para obter Open Graph Tags de uma URL
def get_open_graph_tags(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    meta_tags = soup.find_all("meta")
    og_tags = {}
    for tag in meta_tags:
        if tag.get("property", "").startswith("og:"):
            og_tags[tag["property"][3:]] = tag["content"]
    return og_tags


# Função para gerar resumo usando GPT-4
def generate_summary(article_summary, article_url):
    cleaned_summary = remove_html_tags(article_summary)
    messages = [
        {
            "role": "system",
            "content": f"""Act like Linkedingpt, an advanced platform and language model designed to generate viral posts for LinkedIn, which helps users increase their number of followers and likes on LinkedIn by improving the format and interaction in LinkedIn Feed posts. Use the following rules to get the most views:

            - You are also an Expert in Azure, and Microsoft products, who will post news on LinkedIn to gain organic engagement, using techniques to gain more visibility according to the rules of the LinkedIn algorithm, I will post news in an XML RSS and you summarize in detail, according to the LinkedIn algorithm to gain more visibility.

            - Pay close attention, if the news is in English, always write the summary in BR Portuguese.

            - The first two lines must be creative, engaging, charismatic and intelligent to immediately capture the reader's attention. Start each sentence on a new line and add numbering with emoji to the first two lines for better structuring.

            - The ideal length of the summary is 1,200 and 1,500 characters (never exceed 1,500 characters), written in a professional and organized manner.

            - Use a maximum of 200 characters in each paragraph. At the end of each paragraph there will be an empty line to start the next paragraph.

            - Place an emoji at the beginning of each paragraph, which is related to the written paragraph.

            - An approach with a professional, formal and information-rich tone. Citing examples, code examples, commands or scripts.

            - When you hear commands or scripts from any programming language or framework, always indent the script, command or code separately from the text so that it remains legible and clear to the reader.

            - Structured presentation format, with paragraphs separating different points, with high specificity, always explaining in detail and, if possible, citing examples and how to do it.

            - End the post with a thought-provoking question to encourage community engagement. This should come before hashtags.

            - Always include 4 to 5 Hashtags that are related to the news released at the end of the summary.

            - PAY THIS COMMENT VERY CAREFULLY. Follow ALL of these guidelines to create a post that not only informs, but also engages and inspires, following the rules of the LinkedIn algorithm""",
        },
        {"role": "user", "content": f"Summarize the following news: {cleaned_summary}"},
    ]

    # Moodelo de linguagem GPT-4-32k para gerar o resumo
    try:
        set_openai_config(RBLN_API_BASE, RBLN_API_KEY)
        response = openai.ChatCompletion.create(
            engine="gpt-4-32k", messages=messages, temperature=0.5
        )
    except Exception as e:
        print(f"Erro com API OpenAI Primaria: {e}")
        print("Alterando para API OpenAI Secundaria..")
        set_openai_config(RBPS_API_BASE, RBPS_API_KEY)
        response = openai.ChatCompletion.create(
            engine="gpt-4-32k", messages=messages, temperature=0.5
        )
    summary = response["choices"][0]["message"]["content"]
    return summary


# Obter URN da pessoa autenticada
ME_RESOURCE = f"https://api.linkedin.com/v2/me"
headers = {"Authorization": f"Bearer {ACCESS_TOKEN}"}
me_response = requests.get(ME_RESOURCE, headers=headers)

if "id" not in me_response.json():
    raise Exception(
        "Erro ao obter o perfil do LinkedIn. Verifique o se o token de acesso não está expirado."
    )


# Função para postar no LinkedIn usando a API
def post_to_linkedin(title, description, url):
    owner = "urn:li:person:" + me_response.json()["id"]
    headers = {
        "Authorization": f"Bearer {ACCESS_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "owner": owner,
        "text": {"text": description},
        "content": {
            "contentEntities": [
                {
                    "entityLocation": url,
                    "thumbnails": [
                        {"resolvedUrl": get_open_graph_tags(url).get("image", "")}
                    ],
                }
            ],
            "title": title,
        },
        "distribution": {"linkedInDistributionTarget": {}},
    }

    # Postar no LinkedIn
    response = requests.post(
        "https://api.linkedin.com/v2/shares", headers=headers, json=payload
    )
    response_data = response.json()
    return response_data


# Função para enviar um seletor de data, divida em etapas, primeiro o dia, depois o mês e depois o ano.
def send_datepicker(chat_id):
    now = datetime.now()
    year = now.year

    # Criar botões para cada dia do mês
    days = [
        InlineKeyboardButton(text=str(day), callback_data=f"day_{day}")
        for day in range(1, 32)
    ]

    # Criar botões para cada mês
    months = [
        [
            InlineKeyboardButton(text=str(month), callback_data=f"month_{month}")
            for month in range(1, 7)
        ],
        [
            InlineKeyboardButton(text=str(month), callback_data=f"month_{month}")
            for month in range(7, 13)
        ],
    ]

    # Criar botões para o ano atual e o próximo
    years = [
        InlineKeyboardButton(text=str(year + i), callback_data=f"year_{year+i}")
        for i in range(2)
    ]

    # Agrupar os botões em linhas
    keyboard = [days[i : i + 7] for i in range(0, len(days), 7)]  # Dias
    keyboard.extend(months)  # Meses
    keyboard.append(years)  # Anos

    # Enviar a mensagem com o teclado inline
    bot.sendMessage(
        chat_id,
        "Selecione o dia, mês e ano:",
        reply_markup=InlineKeyboardMarkup(inline_keyboard=keyboard),
    )


# Função para lidar com mensagens recebidas
def handle(msg):
    global awaiting_confirmation, awaiting_schedule, selected_article, summary, article_title, article_url, global_articles, selected_day, selected_month, selected_year
    content_type, chat_type, chat_id = telepot.glance(msg)

    if content_type != "text":
        bot.sendMessage(
            chat_id,
            "*Desculpe, eu só posso processar mensagens de texto.*",
            parse_mode="markdown",
        )
        return

    user_input = (
        msg["text"].strip().lower()
    )  # Converte a entrada para minúsculas para evitar problemas de case-sensitivity

    # Processando comando /start
    if user_input == "/start":
        choose_feed(chat_id)
        return

    # Verificando se o bot está aguardando uma confirmação do usuário
    if awaiting_confirmation:
        handle_confirmation(user_input, chat_id)
        return

    # Verificando se o bot está aguardando uma solicitação de agendamento
    if selected_day and selected_month and selected_year:
        handle_schedule_request(chat_id, user_input)
        return

    # Processando comandos de feed RSS
    if user_input in feed_urls:
        handle_rss_feed(user_input, chat_id)
        return

    # Processando escolha de artigo do usuário
    if user_input.isdigit():
        handle_article_choice(int(user_input), chat_id)
        return

    else:
        bot.sendMessage(
            chat_id,
            "*Desculpe, não entendi o seu pedido. Por favor, selecione um feed RSS ou digite um número de artigo.*",
            parse_mode="markdown",
        )

    if summary:
        send_post_schedule_options(chat_id)
        return


def send_post_schedule_options(chat_id):
    keyboard = [
        [InlineKeyboardButton(text="Postar", callback_data="postar")],
        [InlineKeyboardButton(text="Agendar", callback_data="agendar")],
    ]

    bot.sendMessage(
        chat_id,
        "Escolha uma opção:",
        reply_markup=InlineKeyboardMarkup(inline_keyboard=keyboard),
    )


def choose_feed(chat_id):
    bot.sendMessage(
        chat_id,
        text="Olá, sou o Linkedingpt, um bot que ajuda você a postar notícias no LinkedIn. Escolha um dos feeds RSS disponíveis:",
        reply_markup=InlineKeyboardMarkup(
            inline_keyboard=[
                [
                    InlineKeyboardButton(
                        text="Custom Blog",
                        callback_data="/startcustomblog",
                    ),
                    InlineKeyboardButton(
                        text="Infrastructure",
                        callback_data="/startinfrastructure",
                    ),
                ],
                [
                    InlineKeyboardButton(
                        text="Azure AI Services",
                        callback_data="/startazureaiservices",
                    ),
                    InlineKeyboardButton(
                        text="Microsoft 365",
                        callback_data="/startmicrosoft365",
                    ),
                ],
                [
                    InlineKeyboardButton(
                        text="Serverless",
                        callback_data="/startserverless",
                    ),
                    InlineKeyboardButton(
                        text="GetPratical",
                        callback_data="/startgetpratical",
                    ),
                ],
                [
                    InlineKeyboardButton(
                        text="Educator Developer",
                        callback_data="/startEducatordeveloper",
                    ),
                    InlineKeyboardButton(
                        text="Landing Page",
                        callback_data="/startlandingpage",
                    ),
                ],
                [
                    InlineKeyboardButton(
                        text="Command Line",
                        callback_data="/startcommandline",
                    ),
                    InlineKeyboardButton(
                        text="Mike F Robbins",
                        callback_data="/startmikefrobbins",
                    ),
                ],
                [
                    InlineKeyboardButton(
                        text="Azure Governance MGMT",
                        callback_data="/azuregovernanceandmanagement",
                    ),
                    InlineKeyboardButton(
                        text="Microsoft Entra (Azure AD)",
                        callback_data="/microsoftentra",
                    ),
                ],
                [
                    InlineKeyboardButton(
                        text="Infrastructure Security",
                        callback_data="/infrastructuresecurity",
                    ),
                    InlineKeyboardButton(
                        text="Sec, Compliance Identity",
                        callback_data="/securitycomplianceidentity",
                    ),
                ],
                [
                    InlineKeyboardButton(
                        text="FastTrack for Azure",
                        callback_data="/fasttrackforazure",
                    ),
                    InlineKeyboardButton(
                        text="Apps on Azure Blog",
                        callback_data="/appsonazureblog",
                    ),
                ],
                [
                    InlineKeyboardButton(
                        text="Windows IT Pro",
                        callback_data="/windowsitpro",
                    ),
                    InlineKeyboardButton(
                        text="IT OpsTalk Blog",
                        callback_data="/itopstalkblog",
                    ),
                
                ],
                [
                    InlineKeyboardButton(
                        text="Adam the Automator",
                        callback_data="/adamtheautomator",
                    ),
                    InlineKeyboardButton(
                        text="The Lazy Administrator",
                        callback_data="/thelazyadministrator",
                    ),                    
                ],
                [
                    InlineKeyboardButton(
                        text="PowerShell Community",
                        callback_data="/powershellcommunity",
                    ),
                    InlineKeyboardButton(
                        text="PowerShell Team",
                        callback_data="/powershellteam",
                    ),                    
                ],
                [
                    InlineKeyboardButton(
                        text="Verificar agendamentos pendentes",
                        callback_data="list_schedules",
                    ),
                ],
            ],
        ),
    )


# Função para lidar com a confirmação do usuário
def handle_confirmation(user_input, chat_id):
    global awaiting_confirmation
    if user_input.lower() == "s":
        response_data = post_to_linkedin(article_title, summary, article_url)
        if "id" in response_data:
            bot.sendMessage(
                chat_id, "*Postado com sucesso no LinkedIn.*", parse_mode="markdown"
            )
        else:
            bot.sendMessage(
                chat_id, "*Erro ao postar no LinkedIn.*", parse_mode="markdown"
            )
    elif user_input.lower() == "n":
        bot.sendMessage(chat_id, "*Postagem cancelada.*", parse_mode="markdown")
    else:
        bot.sendMessage(
            chat_id,
            "*Resposta inválida. Por favor, responda com 'S' para SIM ou 'N' para NÃO.*",
            parse_mode="markdown",
        )
        return  # Retorna para evitar redefinir awaiting_confirmation
    awaiting_confirmation = False  # Resetando o estado de awaiting_confirmation


# Função para lidar com a solicitação de agendamento do usuário
def handle_schedule_request(chat_id, message_text):
    global translated_article_title, selected_day, selected_month, selected_year

    try:
        # Asumindo que a mensagem de texto é uma string representando a hora no formato HH:MM
        schedule_time = datetime.strptime(message_text, "%H:%M").time()

        # Criar um objeto datetime com a data selecionada e a hora fornecida
        schedule_datetime = datetime.combine(
            date(selected_year, selected_month, selected_day), schedule_time
        )

        scheduler.add_job(
            post_to_linkedin,
            "date",
            run_date=schedule_datetime,
            args=[article_title, summary, article_url],
            id=translated_article_title,  # Usando o título traduzido do artigo como ID do trabalho
        )
        bot.sendMessage(
            chat_id, "*Postagem agendada com sucesso.*", parse_mode="markdown"
        )
    except ValueError:
        bot.sendMessage(
            chat_id,
            "*Formato de hora inválido. Por favor, forneça a hora no formato HH:MM.*",
            parse_mode="markdown",
        )
        return  # Retorna para evitar redefinir selected_day, selected_month e selected_year
    selected_day = (
        selected_month
    ) = selected_year = None  # Resetando o dia, mês e ano selecionados


# Função para lidar com a escolha de feed RSS do usuário
def handle_rss_feed(user_input, chat_id):
    global global_articles
    bot.sendMessage(
        chat_id,
        "*Bem-vindo! Por favor, selecione o feed RSS do qual deseja coletar as notícias.*",
        parse_mode="markdown",
    )

    # Criação de um botão para cada feed RSS
    keyboard = [
        [InlineKeyboardButton(text=key, callback_data=feed_urls[key])]
        for key in feed_urls
    ]
    bot.sendMessage(
        chat_id,
        "Escolha o feed:",
        reply_markup=InlineKeyboardMarkup(inline_keyboard=keyboard),
    )


# Enviar notificação de geração de resumo
def send_summary_generation_notification(chat_id):
    bot.sendMessage(
        chat_id,
        "*O processo de geração do resumo encontra-se em andamento. Por favor, aguarde enquanto o sistema procede com a produção do resumo.*",
        parse_mode="markdown",
    )


# Função para lidar com a escolha de artigo do usuário
def handle_article_choice(article_index, chat_id):
    global selected_article, article_url, article_title, summary, awaiting_confirmation, awaiting_schedule, translated_article_title

    # Verifica se global_articles foi definido
    if global_articles is None:
        bot.sendMessage(
            chat_id,
            "*Nenhuma notícia foi listada ainda. Por favor, selecione um feed RSS primeiro.*",
            parse_mode="markdown",
        )
        return

    if not 0 < article_index <= len(global_articles):
        bot.sendMessage(
            chat_id,
            "*Número de artigo inválido. Por favor, tente novamente.*",
            parse_mode="markdown",
        )
        return

    # Resetando o estado do bot
    if isinstance(global_articles, list):
        selected_article = global_articles[article_index - 1]
        article_url = selected_article.link
        article_title = selected_article.title
        translated_article_title = translator.translate(article_title, dest="pt").text
    else:
        bot.sendMessage(
            chat_id,
            "*Nenhuma notícia foi listada ainda. Por favor, selecione um feed RSS primeiro.*",
            parse_mode="markdown",
        )
        return

    # Enviar notificação de geração de resumo
    send_summary_generation_notification(chat_id)

    # Gerar resumo
    summary = generate_summary(selected_article.summary, article_url)
    bot.sendMessage(chat_id, f"*Resumo: {summary} *", parse_mode="markdown")

    # Enviar opções de postagem
    keyboard = [
        [
            InlineKeyboardButton(text="Postar", callback_data="postar"),
            InlineKeyboardButton(text="Agendar", callback_data="agendar"),
        ],
        [
            InlineKeyboardButton(
                text="Gerar novo texto", callback_data="gerar_novo_texto"
            )
        ],
        [
            InlineKeyboardButton(
                text="Escolher outra notícia", callback_data="escolher_outra_noticia"
            )
        ],
        [InlineKeyboardButton(text="Cancelar", callback_data="cancelar")],
    ]
    bot.sendMessage(
        chat_id,
        "Escolha uma opção:",
        reply_markup=InlineKeyboardMarkup(inline_keyboard=keyboard),
    )


# Função para listar os agendamentos pendentes
def list_schedules(chat_id):
    jobs = scheduler.get_jobs()
    if jobs:
        bot.sendMessage(chat_id, "*Agendamentos pendentes:*", parse_mode="markdown")
        keyboard = []
        for job in jobs:
            # Formata a data e a hora para o formato dd/mm/yy HH:MM
            formatted_time = job.next_run_time.strftime("%d/%m/%y %H:%M")
            bot.sendMessage(
                chat_id,
                f"- {job.id} agendado para {formatted_time}",
                parse_mode="markdown",
            )
            cancel_id = str(
                uuid.uuid4()
            )  # Cria um identificador único para o botão de cancelamento
            global_jobs[
                cancel_id
            ] = job.id  # Adiciona o trabalho ao dicionário global_jobs
            keyboard.append(
                [
                    InlineKeyboardButton(
                        text=f"Cancelar {job.id}", callback_data=f"cancel_{cancel_id}"
                    )
                ]
            )
        keyboard.append(
            [InlineKeyboardButton(text="Cancelar todos", callback_data="cancel_all")]
        )
        bot.sendMessage(
            chat_id,
            "Selecione um agendamento para cancelar:",
            reply_markup=InlineKeyboardMarkup(inline_keyboard=keyboard),
        )
    else:
        bot.sendMessage(
            chat_id, "*Não há agendamentos pendentes.*", parse_mode="markdown"
        )


# Função para lidar com o retorno da escolha do usuário
# Função para lidar com o retorno da escolha do usuário
def on_callback_query(msg):
    global global_articles, selected_day, selected_month, selected_year, article_title, summary, article_url, translated_article_title, selected_article
    query_id, from_id, query_data = telepot.glance(msg, flavor="callback_query")

    # Verifica se o usuário clicou no botão "Gerar novo texto"
    if query_data == "gerar_novo_texto":
        if selected_article is not None:
            # Gerar novo resumo
            summary = generate_summary(selected_article.summary, article_url)
            bot.sendMessage(
                from_id, f"*Novo resumo: {summary} *", parse_mode="markdown"
            )
            # Apresentar as opções novamente
            keyboard = [
                [
                    InlineKeyboardButton(text="Postar", callback_data="postar"),
                    InlineKeyboardButton(text="Agendar", callback_data="agendar"),
                ],
                [
                    InlineKeyboardButton(
                        text="Gerar novo texto", callback_data="gerar_novo_texto"
                    )
                ],
                [
                    InlineKeyboardButton(
                        text="Escolher outra notícia",
                        callback_data="escolher_outra_noticia",
                    )
                ],
                [InlineKeyboardButton(text="Cancelar", callback_data="cancelar")],
            ]
            bot.sendMessage(
                from_id,
                "Escolha uma opção:",
                reply_markup=InlineKeyboardMarkup(inline_keyboard=keyboard),
            )
        else:
            bot.sendMessage(from_id, "Nenhuma notícia selecionada.")
        return

    # Verifica se o usuário clicou no botão "Escolher outra notícia"
    if query_data == "escolher_outra_noticia":
        for i, article in enumerate(global_articles):
            translated_title = translator.translate(article.title, dest="pt").text
            bot.sendMessage(from_id, f"{i+1}. {translated_title}")
        bot.sendMessage(
            from_id,
            "*Por favor, indique a notícia que deseja resumir informando o número correspondente.*",
            parse_mode="markdown",
        )
        return

    # Verifica se o usuário clicou no botão "Cancelar"
    if query_data == "cancelar":
        bot.sendMessage(from_id, "*Operação cancelada.*", parse_mode="markdown")
        choose_feed(from_id)
        return

    # Verifica se o usuário clicou no botão "Agendar"
    if query_data == "agendar":
        bot.answerCallbackQuery(query_id, text="Agendamento selecionado.")
        send_datepicker(from_id)
        return

    # Verifica se o usuário clicou no botão "Postar"
    if query_data == "postar":
        bot.answerCallbackQuery(query_id, text="Postagem selecionada.")
        response_data = post_to_linkedin(article_title, summary, article_url)
        if "id" in response_data:
            bot.sendMessage(
                from_id, "*Postado com sucesso no LinkedIn.*", parse_mode="markdown"
            )
        else:
            bot.sendMessage(
                from_id, "*Erro ao postar no LinkedIn.*", parse_mode="markdown"
            )
        return

    if query_data.startswith("cancel_"):
        if query_data == "cancel_all":
            scheduler.remove_all_jobs()
            bot.answerCallbackQuery(
                query_id, text="Todos os agendamentos foram cancelados."
            )
        else:
            cancel_id = query_data[len("cancel_") :]
            job_id = global_jobs.get(cancel_id)
            if job_id is not None:
                scheduler.remove_job(job_id)
                bot.answerCallbackQuery(
                    query_id, text=f'O agendamento "{job_id}" foi cancelado.'
                )
        return

    # Verifica se o usuário clicou no botão "Verificar agendamentos pendentes"
    if query_data == "list_schedules":
        list_schedules(from_id)
        return

    # Verifica se o usuário selecionou um dia
    if query_data.startswith("day_"):
        selected_day = int(query_data[len("day_") :])
        bot.answerCallbackQuery(query_id, text=f"Dia {selected_day} selecionado.")
        if selected_month and selected_year:
            bot.sendMessage(from_id, "Por favor, digite a hora no formato HH:MM")
        return

    # Verifica se o usuário selecionou um mês
    if query_data.startswith("month_"):
        selected_month = int(query_data[len("month_") :])
        bot.answerCallbackQuery(query_id, text=f"Mês {selected_month} selecionado.")
        if selected_day and selected_year:
            bot.sendMessage(from_id, "Por favor, digite a hora no formato HH:MM")
        return

    # Verifica se o usuário selecionou um ano
    if query_data.startswith("year_"):
        selected_year = int(query_data[len("year_") :])
        bot.answerCallbackQuery(query_id, text=f"Ano {selected_year} selecionado.")
        if selected_day and selected_month:
            bot.sendMessage(from_id, "Por favor, digite a hora no formato HH:MM")
        return

    # Verifica se o dado retornado é um feed RSS válido
    if query_data in feed_urls:
        feed_url = feed_urls[query_data]
        bot.answerCallbackQuery(query_id, text="Processando feed RSS...")
        feed = feedparser.parse(feed_url)
        articles = feed.entries

        # Limita o número de notícias para 25 se o feed escolhido for startmikefrobbins
        if query_data == "/startmikefrobbins" and len(articles) > 25:
            articles = articles[:25]

        global_articles = articles  # Agora isto está definindo a variável global

        # Envia uma mensagem informando que o feed está sendo traduzido
        feed_name = [
            name for name, command in feed_names.items() if command == query_data
        ][0]
        bot.sendMessage(
            from_id,
            f"*Realizando a tradução dos títulos das notícias do Feed {feed_name}, por favor, aguarde...*",
            parse_mode="markdown",
        )

        for i, article in enumerate(articles):
            try:
                # Traduzindo o título
                translated_title = translator.translate(article.title, dest="pt").text

                # Verificando e formatando a data de publicação
                if hasattr(article, "published_parsed"):
                    publication_date = time.strftime(
                        "%d/%m/%Y", article.published_parsed
                    )
                    message = (
                        f"{i+1}. {translated_title} (Publicado em: {publication_date})"
                    )
                else:
                    # Se a data de publicação não estiver disponível
                    message = f"{i+1}. {translated_title}"

                bot.sendMessage(from_id, message)

            except TypeError:
                bot.sendMessage(from_id, f"{i+1}. Erro na tradução do título")

        bot.sendMessage(
            from_id,
            "*Por favor, indique a notícia que deseja resumir informando o número correspondente.*",
            parse_mode="markdown",
        )

    else:
        bot.answerCallbackQuery(query_id, text="Erro ao processar feed RSS.")


# Inicializar o bot
bot = telepot.Bot(TELEGRAM_TOKEN)

# Configurar o loop de mensagens
bot.message_loop({"chat": handle, "callback_query": on_callback_query})

# Manter o programa em execução
while True:
    pass
