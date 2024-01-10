"""Set of default prompts."""

from llama_index.prompts.base import PromptTemplate
from llama_index.prompts.prompt_type import PromptType

############################################
# Tree
############################################

DEFAULT_SUMMARY_PROMPT_TMPL = (
    "Напишите краткое изложение следующего. Старайтесь использовать только "
    "информация предоставлена."
    "Постарайтесь включить как можно больше ключевых деталей.\n"
    "\n"
    "\n"
    "{context_str}\n"
    "\n"
    "\n"
    'КРАТКОЕ ОПИСАНИЕ:"""\n'
)

DEFAULT_SUMMARY_PROMPT = PromptTemplate(
    DEFAULT_SUMMARY_PROMPT_TMPL, prompt_type=PromptType.SUMMARY
)

# insert prompts
DEFAULT_INSERT_PROMPT_TMPL = (
    "Контекстная информация приведена ниже. Он представлен в виде пронумерованного списка "
    "(от 1 до {num_chunks}),"
    "где каждому элементу в списке соответствует краткое описание.\n"
    "---------------------\ n"
    "{context_list}"
    "---------------------\ n"
    "Учитывая контекстную информацию, вот новый фрагмент "
    "информация: {new_chunk_text}\n"
    "Ответьте номером, соответствующим резюме, которое следует обновить."
    "Ответом должно быть число, соответствующее "
    "краткое изложение, наиболее относящееся к вопросу.\n"
)
DEFAULT_INSERT_PROMPT = PromptTemplate(
    DEFAULT_INSERT_PROMPT_TMPL, prompt_type=PromptType.TREE_INSERT
)


# # single choice
DEFAULT_QUERY_PROMPT_TMPL = (
    "Ниже приведены некоторые варианты. Он представлен в виде пронумерованного списка "
    "(от 1 до {num_chunks}),"
    "где каждому элементу в списке соответствует краткое описание.\n"
    "---------------------\ n"
    "{context_list}"
    "\n---------------------\ n"
    "Используя только указанные выше варианты, а не предварительные знания, верните "
    "выбор, который наиболее релевантен для вопроса: '{query_str}'\n"
    "Предоставьте выбор в следующем формате: 'ОТВЕТ: <число>' и объясните почему "
    "это краткое изложение было выбрано в связи с вопросом.\n"
)
DEFAULT_QUERY_PROMPT = PromptTemplate(
    DEFAULT_QUERY_PROMPT_TMPL, prompt_type=PromptType.TREE_SELECT
)

# multiple choice
DEFAULT_QUERY_PROMPT_MULTIPLE_TMPL = (

    "Ниже приведены некоторые варианты. Он представлен в пронумерованном "
    "список (от 1 до {num_chunks})",
    "где каждому элементу в списке соответствует краткое описание.\n"
    "---------------------\ n"
    "{context_list}"
    "\n---------------------\ n"
    "Используя только приведенные выше варианты, а не предварительные знания, верните лучшие варианты "
    "(не более {branching_factor}, ранжированный от наиболее релевантного к наименьшему), что "
    "наиболее релевантны для вопроса: '{query_str}'\n"
    "Предоставьте варианты в следующем формате: 'ОТВЕТ: <цифры>' и объясните, почему "
    "эти резюме были отобраны в связи с вопросом.\n"
)
DEFAULT_QUERY_PROMPT_MULTIPLE = PromptTemplate(
    DEFAULT_QUERY_PROMPT_MULTIPLE_TMPL, prompt_type=PromptType.TREE_SELECT_MULTIPLE
)


DEFAULT_REFINE_PROMPT_TMPL = (
    "Исходный запрос выглядит следующим образом: {query_str}\n"
    "Мы предоставили существующий ответ: {existing_answer}\n"
    "У нас есть возможность уточнить существующий ответ "
    "(только при необходимости) с дополнительным контекстом ниже.\n"
    "------------\ n"
    "{context_msg}\n"
    "------------\ n"
    "Учитывая новый контекст, уточните первоначальный ответ, чтобы он стал лучше "
    "ответь на запрос."
    "Если контекст бесполезен, верните исходный ответ.\n"
    "Уточненный ответ: "
)



DEFAULT_REFINE_PROMPT = PromptTemplate(
    DEFAULT_REFINE_PROMPT_TMPL, prompt_type=PromptType.REFINE
)


DEFAULT_TEXT_QA_PROMPT_TMPL = (

    "Контекстная информация приведена ниже.\n"
"---------------------\ n"
"{context_str}\n"
"---------------------\ n"
"Учитывая контекстную информацию, а не предварительные знания",
"ответьте на запрос.\n"
"Запрос: {query_str}\n"
"Ответ: "
)
DEFAULT_TEXT_QA_PROMPT = PromptTemplate(
    DEFAULT_TEXT_QA_PROMPT_TMPL, prompt_type=PromptType.QUESTION_ANSWER
)

DEFAULT_TREE_SUMMARIZE_TMPL = (

    "Answer: "
    "Ниже приведена контекстная информация из нескольких источников.\n"
    "---------------------\ n"
    "{context_str}\n"
    "---------------------\ n"
    "Учитывая информацию из нескольких источников, а не предварительные знания",
    "ответьте на запрос.\n"
    "Запрос: {query_str}\n"
    "Ответ: "
)
DEFAULT_TREE_SUMMARIZE_PROMPT = PromptTemplate(
    DEFAULT_TREE_SUMMARIZE_TMPL, prompt_type=PromptType.SUMMARY
)


############################################
# Keyword Table
############################################

DEFAULT_KEYWORD_EXTRACT_TEMPLATE_TMPL = (
    "Ниже приведен некоторый текст. Учитывая текст, извлеките до {max_keywords} "
    "ключевые слова из текста. Избегайте стоп-слов."
    "---------------------\ n"
    "{text}\n"
    "---------------------\ n"
    "Укажите ключевые слова в следующем формате, разделенном запятыми: 'КЛЮЧЕВЫЕ слова: <ключевые слова>'\n"
)
DEFAULT_KEYWORD_EXTRACT_TEMPLATE = PromptTemplate(
    DEFAULT_KEYWORD_EXTRACT_TEMPLATE_TMPL, prompt_type=PromptType.KEYWORD_EXTRACT
)


# NOTE: the keyword extraction for queries can be the same as
# the one used to build the index, but here we tune it to see if performance is better.
DEFAULT_QUERY_KEYWORD_EXTRACT_TEMPLATE_TMPL = (
    "Ниже приведен вопрос. Учитывая вопрос, извлеките до {max_keywords} "
    "ключевые слова из текста. Сосредоточьтесь на извлечении ключевых слов, которые мы можем использовать "
    "чтобы лучше всего найти ответы на этот вопрос. Избегайте стоп-слов.\n"
    "---------------------\ n"
    "{question}\n"
    "---------------------\ n"
    "Укажите ключевые слова в следующем формате, разделенном запятыми: 'КЛЮЧЕВЫЕ слова: <ключевые слова>'\n"
)
DEFAULT_QUERY_KEYWORD_EXTRACT_TEMPLATE = PromptTemplate(
    DEFAULT_QUERY_KEYWORD_EXTRACT_TEMPLATE_TMPL,
    prompt_type=PromptType.QUERY_KEYWORD_EXTRACT,
)


############################################
# Structured Store
############################################

DEFAULT_SCHEMA_EXTRACT_TMPL = (
   
    "Мы хотим извлечь соответствующие поля из неструктурированного текстового фрагмента в "
    "- структурированная схема. Сначала мы предоставляем неструктурированный текст, а затем "
    "мы предоставляем схему, которую хотим извлечь."
    "----------- текст-----------\ n"
    "{text}\n"
    "----------- схема-----------\ n"
    "{schema}\n"
    "---------------------\ n"
    "Учитывая текст и схему, извлеките соответствующие поля из текста в "
    "следующий формат: "
    "поле 1: <значение>\поле 2: <значение>\n...\n\n"
    "Если какое-либо поле отсутствует в тексте, не включайте его в выходные данные."
    "Если в тексте нет полей, верните пустую строку.\n"
    "Поля: "
)
DEFAULT_SCHEMA_EXTRACT_PROMPT = PromptTemplate(
    DEFAULT_SCHEMA_EXTRACT_TMPL, prompt_type=PromptType.SCHEMA_EXTRACT
)

# NOTE: taken from langchain and adapted
# https://tinyurl.com/b772sd77
DEFAULT_TEXT_TO_SQL_TMPL = (
    "SQLQuery: "
    "Учитывая входной вопрос, сначала создайте синтаксически правильный {dialect} "
    "запрос для запуска, затем просмотрите результаты запроса и верните ответ."
    "Вы можете упорядочить результаты по соответствующему столбцу, чтобы вернуть наибольшее количество "
    "интересные примеры в базе данных.\n"
    "Никогда не запрашивайте все столбцы из определенной таблицы, запрашивайте только "
    "несколько соответствующих столбцов с учетом вопроса.\n"
    "Обратите внимание на использование только тех имен столбцов, которые вы можете видеть в схеме "
    "описание."
    "Будьте осторожны и не запрашивайте столбцы, которые не существуют."
    "Обратите внимание, какой столбец находится в какой таблице."
    "Кроме того, при необходимости уточняйте имена столбцов именем таблицы.\n"
    "Используйте следующий формат:\n"
    "Вопрос: Вопрос здесь\n"
    "SQL-запрос: SQL-запрос для запуска\n"
    "SQLResult: результат SQL-запроса\n"
    "Ответ: Окончательный ответ здесь\n"
    "Используйте только таблицы, перечисленные ниже.\n"
    "{schema}\n"
    "Вопрос: {query_str}\n"
    "SqlQuery: "
)

DEFAULT_TEXT_TO_SQL_PROMPT = PromptTemplate(
    DEFAULT_TEXT_TO_SQL_TMPL,
    prompt_type=PromptType.TEXT_TO_SQL,
)


# NOTE: by partially filling schema, we can reduce to a QuestionAnswer prompt
# that we can feed to ur table
DEFAULT_TABLE_CONTEXT_TMPL = (

    "Мы предоставили схему таблицы ниже."
    "---------------------\ n"
    "{schema}\n"
    "---------------------\ n"
    "Мы также предоставили контекстную информацию ниже."
    "{context_str}\n"
    "---------------------\ n"
    "Учитывая контекстную информацию и схему таблицы",
    "дайте ответ на следующую задачу: {query_str}"
)

DEFAULT_TABLE_CONTEXT_QUERY = (
    "Предоставьте высокоуровневое описание таблицы",
    "а также описание каждого столбца в таблице."
    "Предоставьте ответы в следующем формате:\n"
    "Описание таблицы: <описание>\n"
    "Описание столбца 1: <описание>\n"
    "Описание столбца 2: <описание>\n"
    "...\n\n"
)

DEFAULT_TABLE_CONTEXT_PROMPT = PromptTemplate(
    DEFAULT_TABLE_CONTEXT_TMPL, prompt_type=PromptType.TABLE_CONTEXT
)

# NOTE: by partially filling schema, we can reduce to a refine prompt
# that we can feed to ur table
DEFAULT_REFINE_TABLE_CONTEXT_TMPL = (

    "Мы предоставили схему таблицы ниже."
    "---------------------\ n"
    "{schema}\n"
    "---------------------\ n"
    "Мы также предоставили некоторую контекстную информацию ниже."
    "{context_msg}\n"
    "---------------------\ n"
    "Учитывая контекстную информацию и схему таблицы",
    "дайте ответ на следующую задачу: {query_str}\n"
    "Мы предоставили существующий ответ: {existing_answer}\n"
    "Учитывая новый контекст, уточните первоначальный ответ, чтобы он стал лучше "
    "- отвечай на вопрос."
    "Если контекст бесполезен, верните исходный ответ."
)
DEFAULT_REFINE_TABLE_CONTEXT_PROMPT = PromptTemplate(
    DEFAULT_REFINE_TABLE_CONTEXT_TMPL, prompt_type=PromptType.TABLE_CONTEXT
)


############################################
# Knowledge-Graph Table
############################################

DEFAULT_KG_TRIPLET_EXTRACT_TMPL = (
    "Ниже приведен некоторый текст. Учитывая текст, извлеките до "
    "{max_knowledge_triplets} "
    "тройки знаний в форме (субъект, предикат, объект). Избегайте стоп-слов.\n"
    "---------------------\ n"
    "Пример:"
    "Сообщение: Элис - мать Боба."
    "Тройняшки:\n(Алиса, мать Боба)\n"
    "Текст: Philz - кофейня, основанная в Беркли в 1982 году.\n"
    "Тройняшки:\n"
    "(Филз, ис, кофейня)\n"
    "(Philz, основана в Беркли)\n"
    "(Philz, основана в 1982 году)\n"
    "---------------------\ n"
    "Текст: {text}\n"
    "Тройняшки:\n"
)
DEFAULT_KG_TRIPLET_EXTRACT_PROMPT = PromptTemplate(
    DEFAULT_KG_TRIPLET_EXTRACT_TMPL, prompt_type=PromptType.KNOWLEDGE_TRIPLET_EXTRACT
)

############################################
# HYDE
##############################################

HYDE_TMPL = (

    "Пожалуйста, напишите отрывок, чтобы ответить на вопрос\n"
    "Постарайтесь включить как можно больше ключевых деталей.\n"
    "\n"
    "\n"
    "{context_str}\n"
    "\n"
    "\n"
    'Отрывок:"""\n'
)

DEFAULT_HYDE_PROMPT = PromptTemplate(HYDE_TMPL, prompt_type=PromptType.SUMMARY)


############################################
# Simple Input
############################################

DEFAULT_SIMPLE_INPUT_TMPL = "{query_str}"
DEFAULT_SIMPLE_INPUT_PROMPT = PromptTemplate(
    DEFAULT_SIMPLE_INPUT_TMPL, prompt_type=PromptType.SIMPLE_INPUT
)


############################################
# Pandas
############################################

DEFAULT_PANDAS_TMPL = (

    "Вы работаете с фреймворком данных pandas в Python.\n"
    "Имя dataframe - `df`.\n"
    "Это результат `print(df.head())`:\n"
    "{df_str}\n\n"
    "Вот входной запрос: {query_str}.\n"
    "Учитывая информацию df и входной запрос, пожалуйста, следуйте "
    "эти инструкции:\n"
    "{instruction_str}"
    "Вывод:\n"
)

DEFAULT_PANDAS_PROMPT = PromptTemplate(
    DEFAULT_PANDAS_TMPL, prompt_type=PromptType.PANDAS
)


############################################
# JSON Path
############################################

DEFAULT_JSON_PATH_TMPL = (

    "Мы предоставили схему JSON ниже:\n"
    "{schema}\n"
    "Учитывая задачу, ответьте запросом пути в формате JSON, который "
    "может извлекать данные из значения JSON, соответствующего схеме.\n"
    "Задача: {query_str}\n"
    "JSONPath: "
)

DEFAULT_JSON_PATH_PROMPT = PromptTemplate(
    DEFAULT_JSON_PATH_TMPL, prompt_type=PromptType.JSON_PATH
)


############################################
# Choice Select
############################################

DEFAULT_CHOICE_SELECT_PROMPT_TMPL = (

    "Список документов приведен ниже. Рядом с каждым документом проставлен номер. "
    "с кратким изложением документа. Также предусмотрен вопрос. \n"
    "Ответьте номерами документов "
    "вам также следует проконсультироваться, чтобы ответить на вопрос, в порядке значимости."
    "в качестве показателя релевантности. Оценка релевантности - это число от 1 до 10, основанное на "
    "насколько, по вашему мнению, этот документ имеет отношение к данному вопросу.\n"
    "Не включайте никаких документов, которые не имеют отношения к данному вопросу. \n"
    "Пример формата: \n"
    "Документ 1:\n<краткое содержание документа 1>\n\n"
    "Документ 2:\n<краткое содержание документа 2>\n\n"
    "...\n\n"
    "Документ 10:\n<краткое содержание документа 10>\n\n"
    "Вопрос: <вопрос>\n"
    "Ответ:\n"
    "Документ: 9, Актуальность: 7\n"
    "Документ: 3, Актуальность: 4\n"
    "Документ: 7, Актуальность: 3\n\n"
    "Давайте попробуем это сейчас: \n\n"
    "{context_str}\n"
    "Вопрос: {query_str}\n"
    "Ответ:\n"
)
DEFAULT_CHOICE_SELECT_PROMPT = PromptTemplate(
    DEFAULT_CHOICE_SELECT_PROMPT_TMPL, prompt_type=PromptType.CHOICE_SELECT
)
