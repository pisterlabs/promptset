"""Set of default prompts."""

from llama_index.prompts.base import PromptTemplate
from llama_index.prompts.prompt_type import PromptType

############################################
# Tree
############################################

# DEFAULT_SUMMARY_PROMPT_TMPL = (
#     "Write a summary of the following. Try to use only the "
#     "information provided. "
#     "Try to include as many key details as possible.\n"
#     "\n"
#     "\n"
#     "{context_str}\n"
#     "\n"
#     "\n"
#     'SUMMARY:"""\n'
# )
DEFAULT_SUMMARY_PROMPT_TMPL = (
    "写一个关于以下内容的摘要。尽量只使用所提供的信息。"
    "尽量包含尽可能多的关键细节。\n"
    "\n"
    "\n"
    "{context_str}\n"
    "\n"
    "\n"
    '摘要："""\n'
)

DEFAULT_SUMMARY_PROMPT = PromptTemplate(
    DEFAULT_SUMMARY_PROMPT_TMPL, prompt_type=PromptType.SUMMARY
)

# insert prompts
# DEFAULT_INSERT_PROMPT_TMPL = (
#     "Context information is below. It is provided in a numbered list "
#     "(1 to {num_chunks}),"
#     "where each item in the list corresponds to a summary.\n"
#     "---------------------\n"
#     "{context_list}"
#     "---------------------\n"
#     "Given the context information, here is a new piece of "
#     "information: {new_chunk_text}\n"
#     "Answer with the number corresponding to the summary that should be updated. "
#     "The answer should be the number corresponding to the "
#     "summary that is most relevant to the question.\n"
# )
DEFAULT_INSERT_PROMPT_TMPL = (
    "下面提供了上下文信息，以编号列表形式提供（从1到{num_chunks}），"
    "其中列表中的每一项对应一个摘要。\n"
    "---------------------\n"
    "{context_list}"
    "---------------------\n"
    "根据上下文信息，这是一个新的信息片段：{new_chunk_text}\n"
    "答案为应更新的摘要的编号。答案应为与问题最相关的摘要对应的编号。\n"
)

DEFAULT_INSERT_PROMPT = PromptTemplate(
    DEFAULT_INSERT_PROMPT_TMPL, prompt_type=PromptType.TREE_INSERT
)


# # single choice
# DEFAULT_QUERY_PROMPT_TMPL = (
#     "Some choices are given below. It is provided in a numbered list "
#     "(1 to {num_chunks}),"
#     "where each item in the list corresponds to a summary.\n"
#     "---------------------\n"
#     "{context_list}"
#     "\n---------------------\n"
#     "Using only the choices above and not prior knowledge, return "
#     "the choice that is most relevant to the question: '{query_str}'\n"
#     "Provide choice in the following format: 'ANSWER: <number>' and explain why "
#     "this summary was selected in relation to the question.\n"
# )
DEFAULT_QUERY_PROMPT_TMPL = (
    "以下是一些选择项，它们以编号列表的形式呈现（从1到{num_chunks}），"
    "其中列表中的每个项目对应一个摘要。\n"
    "---------------------\n"
    "{context_list}"
    "\n---------------------\n"
    "仅使用上述选择，不使用先前的知识，找出与问题 '{query_str}' 最相关的选择。\n"
    "请以以下格式提供答案：'ANSWER: <编号>'，并解释为什么选择这个摘要与问题相关。\n"
)

DEFAULT_QUERY_PROMPT = PromptTemplate(
    DEFAULT_QUERY_PROMPT_TMPL, prompt_type=PromptType.TREE_SELECT
)

# multiple choice
# DEFAULT_QUERY_PROMPT_MULTIPLE_TMPL = (
#     "Some choices are given below. It is provided in a numbered "
#     "list (1 to {num_chunks}), "
#     "where each item in the list corresponds to a summary.\n"
#     "---------------------\n"
#     "{context_list}"
#     "\n---------------------\n"
#     "Using only the choices above and not prior knowledge, return the top choices "
#     "(no more than {branching_factor}, ranked by most relevant to least) that "
#     "are most relevant to the question: '{query_str}'\n"
#     "Provide choices in the following format: 'ANSWER: <numbers>' and explain why "
#     "these summaries were selected in relation to the question.\n"
# )
DEFAULT_QUERY_PROMPT_MULTIPLE_TMPL = (
    "下面列出了一些选择项，它们以编号列表的形式呈现（从1到{num_chunks}），"
    "列表中的每个项目对应一个摘要。\n"
    "---------------------\n"
    "{context_list}"
    "\n---------------------\n"
    "仅使用上述选择，不使用先前的知识，返回与问题 '{query_str}' 最相关的前若干选择项 "
    "（不超过{branching_factor}个），按从最相关到最不相关的顺序排列。\n"
    "请以以下格式提供选择：'ANSWER: <编号>'，并解释为什么选择这些摘要与问题相关。\n"
)

DEFAULT_QUERY_PROMPT_MULTIPLE = PromptTemplate(
    DEFAULT_QUERY_PROMPT_MULTIPLE_TMPL, prompt_type=PromptType.TREE_SELECT_MULTIPLE
)


# DEFAULT_REFINE_PROMPT_TMPL = (
#     "The original query is as follows: {query_str}\n"
#     "We have provided an existing answer: {existing_answer}\n"
#     "We have the opportunity to refine the existing answer "
#     "(only if needed) with some more context below.\n"
#     "------------\n"
#     "{context_msg}\n"
#     "------------\n"
#     "Given the new context, refine the original answer to better "
#     "answer the query. "
#     "If the context isn't useful, return the original answer.\n"
#     "Refined Answer: "
# )
DEFAULT_REFINE_PROMPT_TMPL = (
    "原始查询如下：{query_str}\n"
    "我们已经提供了一个现有答案：{existing_answer}\n"
    "我们有机会通过以下一些更多的上下文来完善现有答案（仅在需要时）。 \n"
    "------------\n"
    "{context_msg}\n"
    "------------\n"
    "在新的上下文基础上，完善原始答案以更好地回答查询。"
    "如果上下文对于完善答案没有帮助，那么返回原始答案。\n"
    "完善后的答案："
)

DEFAULT_REFINE_PROMPT = PromptTemplate(
    DEFAULT_REFINE_PROMPT_TMPL, prompt_type=PromptType.REFINE
)


# DEFAULT_TEXT_QA_PROMPT_TMPL = (
#     "Context information is below.\n"
#     "---------------------\n"
#     "{context_str}\n"
#     "---------------------\n"
#     "Given the context information and not prior knowledge, "
#     "answer the query.\n"
#     "Query: {query_str}\n"
#     "Answer: "
# )
DEFAULT_TEXT_QA_PROMPT_TMPL = (
    "下面是上下文信息。\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "根据上下文信息回答问题。\n"
    "问题：{query_str}，详细说说\n"
    "答案："
)

DEFAULT_TEXT_QA_PROMPT = PromptTemplate(
    DEFAULT_TEXT_QA_PROMPT_TMPL, prompt_type=PromptType.QUESTION_ANSWER
)

# DEFAULT_TREE_SUMMARIZE_TMPL = (
#     "Context information from multiple sources is below.\n"
#     "---------------------\n"
#     "{context_str}\n"
#     "---------------------\n"
#     "Given the information from multiple sources and not prior knowledge, "
#     "answer the query.\n"
#     "Query: {query_str}\n"
#     "Answer: "
# )
DEFAULT_TREE_SUMMARIZE_TMPL = (
    "下面是来自多个来源的上下文信息。\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "根据来自多个来源的信息而不是先前的知识，回答查询。\n"
    "查询：{query_str}\n"
    "答案："
)

DEFAULT_TREE_SUMMARIZE_PROMPT = PromptTemplate(
    DEFAULT_TREE_SUMMARIZE_TMPL, prompt_type=PromptType.SUMMARY
)


############################################
# Keyword Table
############################################

# DEFAULT_KEYWORD_EXTRACT_TEMPLATE_TMPL = (
#     "Some text is provided below. Given the text, extract up to {max_keywords} "
#     "keywords from the text. Avoid stopwords."
#     "---------------------\n"
#     "{text}\n"
#     "---------------------\n"
#     "Provide keywords in the following comma-separated format: 'KEYWORDS: <keywords>'\n"
# )
DEFAULT_KEYWORD_EXTRACT_TEMPLATE_TMPL = (
    "下面提供了一些文本。根据文本，从中提取最多 {max_keywords} 个关键词。避免使用停用词。"
    "---------------------\n"
    "{text}\n"
    "---------------------\n"
    "请以以下逗号分隔的格式提供关键词：'KEYWORDS: <关键词>'\n"
)

DEFAULT_KEYWORD_EXTRACT_TEMPLATE = PromptTemplate(
    DEFAULT_KEYWORD_EXTRACT_TEMPLATE_TMPL, prompt_type=PromptType.KEYWORD_EXTRACT
)


# NOTE: the keyword extraction for queries can be the same as
# the one used to build the index, but here we tune it to see if performance is better.
# DEFAULT_QUERY_KEYWORD_EXTRACT_TEMPLATE_TMPL = (
#     "A question is provided below. Given the question, extract up to {max_keywords} "
#     "keywords from the text. Focus on extracting the keywords that we can use "
#     "to best lookup answers to the question. Avoid stopwords.\n"
#     "---------------------\n"
#     "{question}\n"
#     "---------------------\n"
#     "Provide keywords in the following comma-separated format: 'KEYWORDS: <keywords>'\n"
# )
DEFAULT_QUERY_KEYWORD_EXTRACT_TEMPLATE_TMPL = (
    "下面提供了一个问题。根据问题，从中提取最多 {max_keywords} 个关键词。专注于提取我们可以用来最佳查找答案的关键词。避免使用停用词。\n"
    "---------------------\n"
    "示例："
    "问题：公司中层在杭州的住宿费是多少？\n"
    "关键词：公司中层，杭州，住宿费\n"
    "---------------------\n"
    "问题：{question}\n"
    "关键词：\n"
)

DEFAULT_QUERY_KEYWORD_EXTRACT_TEMPLATE = PromptTemplate(
    DEFAULT_QUERY_KEYWORD_EXTRACT_TEMPLATE_TMPL,
    prompt_type=PromptType.QUERY_KEYWORD_EXTRACT,
)


############################################
# Structured Store
############################################

# DEFAULT_SCHEMA_EXTRACT_TMPL = (
#     "We wish to extract relevant fields from an unstructured text chunk into "
#     "a structured schema. We first provide the unstructured text, and then "
#     "we provide the schema that we wish to extract. "
#     "-----------text-----------\n"
#     "{text}\n"
#     "-----------schema-----------\n"
#     "{schema}\n"
#     "---------------------\n"
#     "Given the text and schema, extract the relevant fields from the text in "
#     "the following format: "
#     "field1: <value>\nfield2: <value>\n...\n\n"
#     "If a field is not present in the text, don't include it in the output."
#     "If no fields are present in the text, return a blank string.\n"
#     "Fields: "
# )
DEFAULT_SCHEMA_EXTRACT_TMPL = (
    "我们希望从非结构化的文本块中提取相关字段，生成一个结构化模式。"
    "我们首先提供非结构化文本，然后提供我们希望提取的模式。"
    "-----------文本-----------\n"
    "{text}\n"
    "-----------模式-----------\n"
    "{schema}\n"
    "---------------------\n"
    "根据给定的文本和模式，在以下格式中从文本中提取相关字段："
    "字段1: <值>\n字段2: <值>\n...\n\n"
    "如果文本中没有某个字段，请不要在输出中包含它。"
    "如果文本中没有任何字段，请返回一个空字符串。\n"
    "字段："
)

DEFAULT_SCHEMA_EXTRACT_PROMPT = PromptTemplate(
    DEFAULT_SCHEMA_EXTRACT_TMPL, prompt_type=PromptType.SCHEMA_EXTRACT
)

# NOTE: taken from langchain and adapted
# https://tinyurl.com/b772sd77
# DEFAULT_TEXT_TO_SQL_TMPL = (
#     "Given an input question, first create a syntactically correct {dialect} "
#     "query to run, then look at the results of the query and return the answer. "
#     "You can order the results by a relevant column to return the most "
#     "interesting examples in the database.\n"
#     "Never query for all the columns from a specific table, only ask for a "
#     "few relevant columns given the question.\n"
#     "Pay attention to use only the column names that you can see in the schema "
#     "description. "
#     "Be careful to not query for columns that do not exist. "
#     "Pay attention to which column is in which table. "
#     "Also, qualify column names with the table name when needed.\n"
#     "Use the following format:\n"
#     "Question: Question here\n"
#     "SQLQuery: SQL Query to run\n"
#     "SQLResult: Result of the SQLQuery\n"
#     "Answer: Final answer here\n"
#     "Only use the tables listed below.\n"
#     "{schema}\n"
#     "Question: {query_str}\n"
#     "SQLQuery: "
# )
DEFAULT_TEXT_TO_SQL_TMPL = (
    "给定一个输入问题，首先创建一个符合语法的{dialect}查询以运行，然后查看查询结果并返回答案。"
    "您可以通过相关列对结果进行排序，以返回数据库中最有趣的示例。"
    "永远不要查询特定表中的所有列，只询问与问题相关的少数列。"
    "注意仅使用在模式描述中可见的列名。"
    "小心不要查询不存在的列。"
    "注意哪个列位于哪个表中。"
    "在需要时，也要用表名限定列名。\n"
    "使用以下格式：\n"
    "问题：在这里提出问题\n"
    "SQL查询：要运行的SQL查询\n"
    "SQL结果：SQL查询结果\n"
    "答案：在这里给出最终答案\n"
    "仅使用下面列出的表。\n"
    "{schema}\n"
    "问题：{query_str}\n"
    "SQL查询："
)

DEFAULT_TEXT_TO_SQL_PROMPT = PromptTemplate(
    DEFAULT_TEXT_TO_SQL_TMPL,
    prompt_type=PromptType.TEXT_TO_SQL,
)


# NOTE: by partially filling schema, we can reduce to a QuestionAnswer prompt
# that we can feed to ur table
# DEFAULT_TABLE_CONTEXT_TMPL = (
#     "We have provided a table schema below. "
#     "---------------------\n"
#     "{schema}\n"
#     "---------------------\n"
#     "We have also provided context information below. "
#     "{context_str}\n"
#     "---------------------\n"
#     "Given the context information and the table schema, "
#     "give a response to the following task: {query_str}"
# )
DEFAULT_TABLE_CONTEXT_TMPL = (
    "我们在下面提供了一个表结构。"
    "---------------------\n"
    "{schema}\n"
    "---------------------\n"
    "我们还在下面提供了一些上下文信息。"
    "{context_str}\n"
    "---------------------\n"
    "根据上下文信息和表结构，"
    "针对以下任务给出一个回答：{query_str}"
)

# DEFAULT_TABLE_CONTEXT_QUERY = (
#     "Provide a high-level description of the table, "
#     "as well as a description of each column in the table. "
#     "Provide answers in the following format:\n"
#     "TableDescription: <description>\n"
#     "Column1Description: <description>\n"
#     "Column2Description: <description>\n"
#     "...\n\n"
# )
DEFAULT_TABLE_CONTEXT_QUERY = (
    "提供一个关于表的高级描述，以及表中每个列的描述。"
    "请按以下格式提供答案：\n"
    "表描述： <描述>\n"
    "列1描述： <描述>\n"
    "列2描述： <描述>\n"
    "...\n\n"
)

DEFAULT_TABLE_CONTEXT_PROMPT = PromptTemplate(
    DEFAULT_TABLE_CONTEXT_TMPL, prompt_type=PromptType.TABLE_CONTEXT
)

# NOTE: by partially filling schema, we can reduce to a RefinePrompt
# that we can feed to ur table
# DEFAULT_REFINE_TABLE_CONTEXT_TMPL = (
#     "We have provided a table schema below. "
#     "---------------------\n"
#     "{schema}\n"
#     "---------------------\n"
#     "We have also provided some context information below. "
#     "{context_msg}\n"
#     "---------------------\n"
#     "Given the context information and the table schema, "
#     "give a response to the following task: {query_str}\n"
#     "We have provided an existing answer: {existing_answer}\n"
#     "Given the new context, refine the original answer to better "
#     "answer the question. "
#     "If the context isn't useful, return the original answer."
# )
DEFAULT_REFINE_TABLE_CONTEXT_TMPL = (
    "我们在下面提供了一个表结构。"
    "---------------------\n"
    "{schema}\n"
    "---------------------\n"
    "我们还在下面提供了一些上下文信息。"
    "{context_msg}\n"
    "---------------------\n"
    "根据上下文信息和表结构，"
    "针对以下任务给出一个回答：{query_str}\n"
    "我们已经提供了一个现有答案：{existing_answer}\n"
    "根据新的上下文，优化原始答案以更好地回答问题。"
    "如果上下文无用，请保持原始答案。"
)

DEFAULT_REFINE_TABLE_CONTEXT_PROMPT = PromptTemplate(
    DEFAULT_REFINE_TABLE_CONTEXT_TMPL, prompt_type=PromptType.TABLE_CONTEXT
)


############################################
# Knowledge-Graph Table
############################################

DEFAULT_KG_TRIPLET_EXTRACT_TMPL = (
    "Some text is provided below. Given the text, extract up to "
    "{max_knowledge_triplets} "
    "knowledge triplets in the form of (subject, predicate, object). Avoid stopwords.\n"
    "---------------------\n"
    "Example:"
    "Text: Alice is Bob's mother."
    "Triplets:\n(Alice, is mother of, Bob)\n"
    "Text: Philz is a coffee shop founded in Berkeley in 1982.\n"
    "Triplets:\n"
    "(Philz, is, coffee shop)\n"
    "(Philz, founded in, Berkeley)\n"
    "(Philz, founded in, 1982)\n"
    "---------------------\n"
    "Text: {text}\n"
    "Triplets:\n"
)
# DEFAULT_KG_TRIPLET_EXTRACT_TMPL = (
#     "下面提供了一些文本。根据文本，提取最多 {max_knowledge_triplets} 个知识三元组，"
#     "形式为（主语，谓语，宾语）。避免使用停用词。\n"
#     "---------------------\n"
#     "示例："
#     "文本：Alice是Bob的母亲。"
#     "三元组：\n（Alice，是...的母亲，Bob）\n"
#     "文本：Philz是于1982年在伯克利创立的咖啡店。\n"
#     "三元组：\n"
#     "(Philz，是，咖啡店)\n"
#     "(Philz，创立于，伯克利)\n"
#     "(Philz，创立于，1982年)\n"
#     "---------------------\n"
#     "文本：{text}\n"
#     "三元组：\n"
# )

DEFAULT_KG_TRIPLET_EXTRACT_PROMPT = PromptTemplate(
    DEFAULT_KG_TRIPLET_EXTRACT_TMPL, prompt_type=PromptType.KNOWLEDGE_TRIPLET_EXTRACT
)

############################################
# HYDE
##############################################

# HYDE_TMPL = (
#     "Please write a passage to answer the question\n"
#     "Try to include as many key details as possible.\n"
#     "\n"
#     "\n"
#     "{context_str}\n"
#     "\n"
#     "\n"
#     'Passage:"""\n'
# )
HYDE_TMPL = (
    "请撰写一个段落来回答问题\n"
    "尽量包含尽可能多的关键细节。\n"
    "\n"
    "\n"
    "{context_str}\n"
    "\n"
    "\n"
    '段落："""\n'
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

# DEFAULT_PANDAS_TMPL = (
#     "You are working with a pandas dataframe in Python.\n"
#     "The name of the dataframe is `df`.\n"
#     "This is the result of `print(df.head())`:\n"
#     "{df_str}\n\n"
#     "Here is the input query: {query_str}.\n"
#     "Given the df information and the input query, please follow "
#     "these instructions:\n"
#     "{instruction_str}"
#     "Output:\n"
# )
DEFAULT_PANDAS_TMPL = (
    "您正在使用 Python 中的 pandas 数据帧。\n"
    "数据帧的名称是 `df`。\n"
    "这是 `print(df.head())` 的结果：\n"
    "{df_str}\n\n"
    "这是输入的查询：{query_str}。\n"
    "根据 df 信息和输入的查询，请遵循以下说明：\n"
    "{instruction_str}"
    "输出：\n"
)

DEFAULT_PANDAS_PROMPT = PromptTemplate(DEFAULT_PANDAS_TMPL, prompt_type=PromptType.PANDAS)


############################################
# JSON Path
############################################

# DEFAULT_JSON_PATH_TMPL = (
#     "We have provided a JSON schema below:\n"
#     "{schema}\n"
#     "Given a task, respond with a JSON Path query that "
#     "can retrieve data from a JSON value that matches the schema.\n"
#     "Task: {query_str}\n"
#     "JSONPath: "
# )
DEFAULT_JSON_PATH_TMPL = (
    "我们在下面提供了一个 JSON 模式：\n"
    "{schema}\n"
    "根据任务，使用一个 JSON Path 查询来检索与模式匹配的 JSON 值中的数据。\n"
    "任务：{query_str}\n"
    "JSONPath："
)

DEFAULT_JSON_PATH_PROMPT = PromptTemplate(
    DEFAULT_JSON_PATH_TMPL, prompt_type=PromptType.JSON_PATH
)

############################################
# Choice Select
############################################

# DEFAULT_CHOICE_SELECT_PROMPT_TMPL = (
#     "A list of documents is shown below. Each document has a number next to it along "
#     "with a summary of the document. A question is also provided. \n"
#     "Respond with the numbers of the documents "
#     "you should consult to answer the question, in order of relevance, as well \n"
#     "as the relevance score. The relevance score is a number from 1-10 based on "
#     "how relevant you think the document is to the question.\n"
#     "Do not include any documents that are not relevant to the question. \n"
#     "Example format: \n"
#     "Document 1:\n<summary of document 1>\n\n"
#     "Document 2:\n<summary of document 2>\n\n"
#     "...\n\n"
#     "Document 10:\n<summary of document 10>\n\n"
#     "Question: <question>\n"
#     "Answer:\n"
#     "Doc: 9, Relevance: 7\n"
#     "Doc: 3, Relevance: 4\n"
#     "Doc: 7, Relevance: 3\n\n"
#     "Let's try this now: \n\n"
#     "{context_str}\n"
#     "Question: {query_str}\n"
#     "Answer:\n"
# )
DEFAULT_CHOICE_SELECT_PROMPT_TMPL = (
    "下面显示了一份文档列表。每个文档旁边都有一个数字，以及文档的摘要。还提供了一个问题。\n"
    "请按照相关性顺序回答，列出您认为用于回答问题的文档的编号以及相关性评分（1-10）。\n"
    "请勿包括与问题无关的文档。\n"
    "示例格式：\n"
    "文档 1：\n<文档 1 的摘要>\n\n"
    "文档 2：\n<文档 2 的摘要>\n\n"
    "...\n\n"
    "文档 10：\n<文档 10 的摘要>\n\n"
    "问题： <问题>\n"
    "答案：\n"
    "文档：9，相关性：7\n"
    "文档：3，相关性：4\n"
    "文档：7，相关性：3\n\n"
    "现在让我们试一试：\n\n"
    "{context_str}\n"
    "问题： {query_str}\n"
    "答案：\n"
)
DEFAULT_CHOICE_SELECT_PROMPT = PromptTemplate(
    DEFAULT_CHOICE_SELECT_PROMPT_TMPL, prompt_type=PromptType.CHOICE_SELECT
)