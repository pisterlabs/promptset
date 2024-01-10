import logging
from operator import itemgetter
from typing import List

from cassandra.cluster import ResultSet
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

from pydantic_models.ExecutionFailure import ExecutionFailure
from pydantic_models.TableExecutionInfo import TableExecutionInfo
from pydantic_models.TableSchema import TableSchema
from pydantic_models.UserInfo import UserInfo

from langchain_core.runnables import RunnableBranch, RunnableSerializable


def get_python_code_gen_prefix() -> str:
    """
    Returns a prefix for generating Python code generation prompts.
    This prefix instructs the assistant to focus solely on generating Python code based on a given scenario without any additional explanation or summary.
    Returns:
        str: The prefix string for Python code generation prompts.
    """
    return f"""You're a helpful assistant. I will give you a scenario, and I want you to generate Python code that performs the task. Don't give me any summary information, explanation, or introduction. Don't say anything other than the code. \n"""


def get_cql_code_gen_prefix() -> str:
    """
    Returns a prefix for generating CQL (Cassandra Query Language) code generation prompts.
    This prefix guides the assistant to generate valid CQL code for AstraDB/Cassandra tasks without offering any extra information or context.
    Returns:
        str: The prefix string for CQL code generation prompts.
    """
    return f"""You're a helpful assistant. I will give you a scenario, and I want you to generate valid CQL code (for 
    AstraDB/Cassandra) that performs the task. Don't give me any summary information, explanation, or introduction. 
    Don't say anything other than the code. \n"""


def get_personal_response_prefix() -> str:
    """
    Returns a prefix for generating personal and empathetic responses in customer interactions.
    This prefix sets the tone for respectful and empathetic communication with customers.
    Returns:
        str: The prefix string for personal response generation.
    """
    return f"""You're a helpful assistant. You're talking to a customer, so be respectful and show empathy.
    \n"""


def get_helpful_assistant_prefix() -> str:
    """
    Returns a prefix for generating concise and direct responses from the assistant.
    This prefix instructs the assistant to provide responses strictly adhering to the specified request without any additional information or elaboration.
    Returns:
        str: The prefix string for concise and direct assistant responses.
    """
    return f"""You're a helpful assistant. Don't give me any summary information, explanation, or introduction. In 
    fact, don't say anything other than what I specify. \n"""


def build_filter_and_summary_joiner_prompt() -> PromptTemplate:
    """
    Builds and returns a prompt template for combining filters and user information into a single JSON object.
    The prompt is designed to join user summary information with filters to create a combined JSON object for subsequent processing.
    Returns:
        PromptTemplate: The constructed prompt template for filter and summary joining.
    """
    prompt = (
        get_helpful_assistant_prefix()
        + """You will be given a list of filters in this format: [{{"metadata.path_segment_X": "VALUE"}}]
    You will also be given a summary of user information. You need to combine this information into one dictionary object like this:
    [{{"filter": {{"metadata.path_segment_X": "VALUE"}}, "user_summary": summary_contents  }},
     {{"filter": {{"metadata.path_segment_Y": "VALUE"}}, "user_summary": summary_contents  }}]
    so that the summary_contents data is joined to each of the unique metadata.path_segment objects.
    Return only the resulting JSON list.
    
    FILTERS:
    {PathSegmentValues}
    
    USER SUMMARY:
    {UserInformationSummary}
    """
    )
    return PromptTemplate.from_template(prompt)


def build_table_identification_prompt() -> PromptTemplate:
    """
    Builds and returns a prompt template for identifying relevant tables based on user information and table schemas.
    The prompt aims to select tables with columns that match or are likely to match user properties, helping to focus on relevant data.
    Returns:
        PromptTemplate: The constructed prompt template for table identification.
    """
    prompt: str = (
        get_helpful_assistant_prefix()
        + """You are also a great Cassandra database engineer. Your goal is to identify tables that likely contain 
        information that will be relevant for a user (whom I will describe below) so that we can take additional 
        steps to gather more specific information from the relevant tables. I will give you a list of tables, 
        along with their columns and descriptive information, and I will also give you information about that 
        particular user, potentially including conversational history with them, and I want you to return a list with 
        only the tables that have at least one column name that appears to match (or very likely matches) at least 
        one of the properties of the provided user information. I want you to ONLY return the list as a JSON array in 
        the following format: [{{ \"keyspace_name\":\"example_keyspace1\", \"table_name\":\"example_table1\"}}, 
        [{{\"keyspace_name\":\"example_keyspace2\", \"table_name\":\"example_table2\"}}]
        Don't return ANY other information since I need to parse your response as valid JSON.
        
        
        TABLE LIST:
        {TableList}
        
        
        USER INFORMATION:
        {UserInfo}
        
        """
    )
    return PromptTemplate.from_template(prompt)


def build_final_response_prompt() -> PromptTemplate:
    """
    Constructs a prompt template for generating a final recommendation to a customer.
    The prompt synthesizes user summaries, business insights, and conversation history to formulate a recommendation that aligns with the customer's needs and interests.
    Returns:
        PromptTemplate: The constructed prompt template for final response generation.
    """
    prompt = (
        get_personal_response_prefix()
        + """"You are responsible for representing the business to the customer for both sales and customer support purposes. 
        Your ability to address the user's intent is critical to the success of our business.
        You will be given a rich set of summaries of basically everything we know about this customer, 
        and then you will be given a second set of summaries with a lot of information that is likely relevant to 
        their needs based on a previously run search algorithm that we ran internally on their information. Your job 
        is to use this information to make the best possible recommendation to the customer. The recommendation 
        should be grounded in what we know about them and our business, based on the information we obtained. 
        Recommend what is best for the customer, but if it's good for the business also, that's a double win.
        You will also be provided (at the end) with the customer's most recent messages, ordered from oldest to most recent.
        Be sure that your recommendation to them is relevant to their messages, especially their most recent message.
        Always do what is in the customer's best interest.
        Also, don't provide a message signature, and don't say anything that we already said to the customer.
        
        USER SUMMARIES:
        {UserSummary}
        
        
        
        BUSINESS SUMMARIES:
        {BusinessSummary}
        
        
        
        USER MESSAGES:
        {UserMessages}
        
        
        
        OUR PRIOR RESPONSES:
        {OurResponses}
        """
    )
    return PromptTemplate.from_template(prompt)


def build_summarization_prompt() -> PromptTemplate:
    """
    Creates a prompt template for summarizing provided information, focusing on technical details that could influence recommendations or decisions.
    The prompt is intended to yield concise summaries that highlight key information relevant to customer support or business insights.
    Returns:
        PromptTemplate: The constructed prompt template for summarization.
    """
    prompt = (
        "You're a helpful assistant. I'm will give you some information, and I want you to summarize what I'm "
        "providing you. This information will be used to either summarize something about a customer or "
        "something we know internally that we will use to make a recommendation, so keep that in mind as you "
        "write the summary. You want to focus your summary around technical information that might influence a "
        "recommendation or decision."
        "Don't be wordy, but provide enough detail so that patterns can "
        "be identified when this summary is combined with others. Any device-specific or plan-specific details should "
        "be included.)\n You want to provide a technical summary that can be used for subsequent steps where the "
        "information will be assimiliated by a customer support agent to answer a question for a customer. Focus on "
        "information that might be relevant for a customer. If the information I provide is all blank after I say "
        '"Here is the information I want you to summarize:", return "skipped"'
        "Here is the information I want you to summarize:"
        ""
        ""
        "{Information}"
    )
    return PromptTemplate.from_template(prompt)


def build_collection_vector_find_prompt() -> PromptTemplate:
    """
    Develops a prompt template for generating a JSON object with metadata path segment keys based on vector data and user information.
    The prompt guides the assistant to select the most appropriate metadata path segments and construct JSON objects to aid in article retrieval.
    Returns:
        PromptTemplate: The constructed prompt template for collection vector find.
    """
    prompt = (
        get_helpful_assistant_prefix()
        + """ Based on the provided summary

Generate a JSON object containing one or more of the following keys: metadata.path_segment_1, 
metadata.path_segment_2, metadata.path_segment_3, metadata.path_segment_4, metadata.path_segment_5, 
metadata.path_segment_6 Based on the following VECTOR DATA, I want you to determine which values of those 
metadata path segment keys are most likely to be associated with the USER INFORMATION below. Each path 
segment is more specific than the one prior to it. Use the most granular path_segment that makes sense. Avoid 
using more than one filter but always use at least one. Once you select a filter, I want you to return the 
information in a list of JSON objects with the following example syntax: [{{"metadata.path_segment_X": "VALUE"}}] where X 
is the selected path segment value, and VALUE is the value you've chosen based on the PATH SEGMENT VALUES 
available below. Only select values from the PATH SEGMENT VALUES below. Don't create any other path segment 
values. Also, ensure that the path segment value you selected corresponds to the correct path_segment number. 
For example, if the data below shows that 'residential' is associated with path_segment_2, don't use 
'residential' as a path segment value for any path other than path_segment_2. 

I want you to construct at least 10 (but less than 20) such JSON objects, and they should cover different subjects associated with the provided USER INFORMATION SUMMARY below.
If you can find at least 3 involving metadata.path_segment_6, 3 involving metadata.path_segment_5, 3 involving metadata.path_segment_4,
3 involving metadata.path_segment_3, and 3 involving metadata.path_segment_2, that would be good.
Select matches that are as specific as possible. 

Return ONLY this list of JSON objects."""
        + """
PATH SEGMENT VALUES:

{PathSegmentValues}

USER INFORMATION SUMMARY:

{UserInformationSummary}
        """
    )
    return PromptTemplate.from_template(prompt)


def build_collection_vector_find_prompt_v3() -> PromptTemplate:
    prompt = (
        get_helpful_assistant_prefix()
        + """
Based on the provided vector data and user information, generate a list of JSON objects using the most appropriate metadata path segments. The task involves selecting values from the specified path segments, where each path segment represents a level of specificity. You are to use the most granular path segment that applies.
I will give you 6 lists of keywords and information about a customer. These JSON objects will be used in a later step to construct queries that will be used to find articles with information that should help the customer. 
It is critical that the keywords match the customer's intent so that we can retrieve articles that will resonate with the customer. 
If the keywords don't relate to the customer's information, then it will be very bad because you will cause the customer to receive information that won't relate to them and could upset or offend them.

Guidelines:
- Use at least one path segment for each JSON object, but avoid using more than one.
- Ensure the path segment value corresponds to the correct path segment number based on the provided lists.
- Create 10 to 19 JSON objects, covering different subjects related to the user information.
- Aim for diversity in path segment usage, with at least 3 objects for each path segment from 2 to 6, selecting the most specific matches possible.
- Ensure the path segment value is strongly associated with at least some of the content of the USER INFORMATION SUMMARY below.
- Don't use the same path segment value more than once. Prefer the most specific match if there are multiple matches.
Output Format:
- The output should be in the form of a list of JSON objects: [{{"metadata.path_segment_X": "VALUE"}}], where X is the path segment number, and VALUE is the selected value.

Example Output:
[{{"metadata.path_segment_3": "how-to-use-a-verizon-jetpack"}}, {{"metadata.path_segment_2": "verizon-5g-home-router-troubleshooting"}}]

Return ONLY this list of JSON objects."""
        + """
================
PATH SEGMENT VALUES:

{PathSegmentValues}
================
USER INFORMATION SUMMARY:

{UserInformationSummary}
"""
    )
    return PromptTemplate.from_template(prompt)


def build_collection_vector_find_prompt_v4() -> PromptTemplate:
    prompt = (
        get_helpful_assistant_prefix()
        + """ 
I will give you 6 lists of path segment values and information about a customer. I want you to use the information to create a list of JSON objects. These JSON objects will be used in a later step to construct queries that will be used to find articles with information that should help the customer. 
It is critical that the path segment values match the customer's intent so that we can retrieve articles that will resonate with the customer. 
If the path segment values don't relate to the customer's information, then it will be very bad because you will cause the customer to receive information that won't relate to them and could upset or offend them.
You must follow these rules that apply to each JSON object in the list:
- The value of the JSON object MUST exist in the corresponding list that I will provide below. 
EXAMPLE: if the value "how-to-use-a-verizon-jetpack" exists in the list for path segment values of "metadata.path_segment_3", you may use "how-to-use-a-verizon-jetpack" as the value for the JSON object if and only if:
    -- The path segment key is "metadata.path_segment_3"
    -- The path segment key (in this case "how-to-use-a-verizon-jetpack") is strongly associated with at least some of the content of the USER INFORMATION SUMMARY below.
    -- There is not another path segment value from the "metadata.path_segment_3" list that is a better match to some of the USER INFORMATION SUMMARY.
    -- The value (in this case "how-to-use-a-verizon-jetpack") does not exist more than once in the JSON list you provide.
ADDITIONAL GUIDELINES:
- You should always select the most specific matches available. For example, if the USER INFORMATION SUMMARY mentions an "iPhone 13", if the path segment value "iPhone 13" is available (in one of the path segment value lists), you should prefer the more specific ("iPhone 13" in this case) over "iPhone" or "phone". 
- You should NEVER create a JSON object using a value that doesn't exist in the available path segment values. 
- You should NEVER create a JSON object using a value that is strongly unrelated to any content in the USER INFORMATION SUMMARY.
- Pay very careful attention to which path segment values (path segment values) are part of which list to ensure you don't try to use a path segment value as a value for a key to which it does not belong.
- Use at least one path segment for each JSON object, but avoid using more than one.
- Ensure the path segment value corresponds to the correct path segment number based on the provided lists.
- Create 10 to 19 JSON objects, covering different subjects related to the user information.
- Aim for diversity in path segment usage, with at least 3 objects for each path segment from 2 to 6, selecting the most specific matches possible.
- Ensure the path segment value is strongly associated with at least some of the content of the USER INFORMATION SUMMARY below.
- Don't use the same path segment value more than once. Prefer the most specific match if there are multiple matches.
- Select at least three from metadata.path_segment_5

Here is the USER INFORMATION SUMMARY. I will repeat it again at the end. Remember to only find path segment values that strongly relate to information in the USER INFORMATION SUMMARY.
For example, if the user is asking for support, select support-related path segment values, not security-related path segment values. If the user is interested in upgrading, don't bring up path segment values about firewalls. Bring up promotional path segment values instead.
"""
        + """
USER INFORMATION SUMMARY:

{UserInformationSummary}


PATH SEGMENT VALUES:

{PathSegmentValues}

USER INFORMATION SUMMARY:

{UserInformationSummary}
        """
    )
    return PromptTemplate.from_template(prompt)


def build_collection_vector_find_prompt_v2() -> PromptTemplate:
    """
    Constructs a version 2 prompt template for creating JSON objects using metadata path segments.
    This prompt specifically focuses on aligning the JSON objects with the customer's intent and information, ensuring relevance in article retrieval.
    Returns:
        PromptTemplate: The constructed prompt template for collection vector find version 2.
    """
    prompt = (
        get_helpful_assistant_prefix()
        + """ I will give you 6 lists of keywords and information about a customer. I want you to use the information to create a list of JSON objects. These JSON objects will be used in a later step to construct queries that will be used to find articles with information that should help the customer. 
        It is critical that the keywords match the customer's intent so that we can retrieve articles that will resonate with the customer. 
        If the keywords don't relate to the customer's information, then it will be very bad because you will cause the customer to receive information that won't relate to them and could upset or offend them.
        You must follow these rules that apply to each JSON object in the list:
1. The JSON object will contain a single key and a single value. 
2. The key will be one of the following values: "metadata.path_segment_1", "metadata.path_segment_2", "metadata.path_segment_3", "metadata.path_segment_4", "metadata.path_segment_5", "metadata.path_segment_6"
3. The value of the JSON object MUST exist in the corresponding list that I will provide below. For example, if the value "how-to-use-a-verizon-jetpack" exists in the list for keywords of "metadata.path_segment_3", you may use "how-to-use-a-verizon-jetpack" as the value for the JSON object if and only if:
    a. The key is "metadata.path_segment_3"
    b. The value (in this case "how-to-use-a-verizon-jetpack") is strongly associated with at least some of the content of the USER INFORMATION SUMMARY below.
    c. There is not another keyword from the "metadata.path_segment_3" list that is a better match to some of the USER INFORMATION SUMMARY.
    d. The value (in this case "how-to-use-a-verizon-jetpack") does not exist more than once in the JSON list you provide.
    e. If you're not sure, it's much better to use a more generic keyword like something related to "promos" or "promotions"
4. The JSON list should contain at least 5 distinct objects.
5. The JSON list should NEVER contain more than 20 objects.
6. The JSON list MUST never contain 0 objects.
7. You should always select the most specific matches available. For example, if the USER INFORMATION SUMMARY mentions an "iPhone 13", if the keyword "iPhone 13" is available (in one of the keyword lists), you should prefer the more specific ("iPhone 13" in this case) over "iPhone" or "phone". 
8. You should NEVER create a JSON object using a value that doesn't exist in the available keywords. 
9. You should NEVER create a JSON object using a value that is strongly unrelated to any content in the USER INFORMATION SUMMARY.
10. You must NEVER return anything other than a list of JSON objects.
11. Pay careful attention to which keywords are part of which list to ensure you don't try to use a keyword as a value for a key to which it does not belong.

Additional guidelines:
- Use at least one path segment for each JSON object, but avoid using more than one.
- Ensure the path segment value corresponds to the correct path segment number based on the provided lists.
- Create 10 to 19 JSON objects, covering different subjects related to the user information.
- Aim for diversity in path segment usage, with at least 3 objects for each path segment from 2 to 6, selecting the most specific matches possible.
- Ensure the path segment value is strongly associated with at least some of the content of the USER INFORMATION SUMMARY below.
- Don't use the same path segment value more than once. Prefer the most specific match if there are multiple matches.

Here is the USER INFORMATION SUMMARY. I will repeat it again at the end. Remember to only find keywords that strongly relate to information in the USER INFORMATION SUMMARY.
For example, if the user is asking for support, select support-related keywords, not security-related keywords. If the user is interested in upgrading, don't bring up keywords about firewalls. Bring up promotional keywords instead.
"""
        + """
USER INFORMATION SUMMARY:

{UserInformationSummary}


AVAILABLE KEYWORDS:

{PathSegmentValues}

USER INFORMATION SUMMARY:

{UserInformationSummary}
        """
    )
    return PromptTemplate.from_template(prompt)


def get_relevant_user_tables(tables: list[TableSchema], user_info: UserInfo):
    prefix = get_python_code_gen_prefix()
    prompt = f"""{prefix} \n I will give you a list of table schemas and some user info, and I want you to return the names of tables that have any column name that matches any of the user info property names."""
    for table in tables:
        prompt += table.to_lcel_json()


def build_select_query_for_top_three_rows() -> PromptTemplate:
    """
    Builds a prompt template for generating a SELECT query to retrieve the top three rows from a given table schema.
    The prompt specifically requests a SELECT query with a LIMIT of 3, adhering to CQL syntax.
    Returns:
        PromptTemplate: The constructed prompt template for SELECT query generation.
    """
    prefix = get_cql_code_gen_prefix()
    prompt = (
        prefix
        + """\n Write a SELECT * query limit 3 for the given table and columns. Return ONLY a SELECT query. NEVER return anything other than a SELECT query.
    {TableSchema}
    
    RESPONSE:
"""
    )
    return PromptTemplate.from_template(prompt)


def build_select_query_for_top_three_rows_parallelizable(
    table_schema: TableSchema,
) -> PromptTemplate:
    """
    Constructs a parallelizable prompt template for creating a SELECT query to fetch the top three rows of a specified table schema.
    The prompt directs the generation of a CQL SELECT query with a LIMIT of 3, focusing on the provided table schema.
    Parameters:
        table_schema (TableSchema): The schema of the table for query generation.
    Returns:
        PromptTemplate: The constructed prompt template for parallelizable SELECT query generation.
    """

    prefix = get_cql_code_gen_prefix()
    prompt = (
        prefix
        + f"""\n Write a SELECT * query limit 3 for the given table and columns. Return ONLY a SELECT query. NEVER return anything other than a SELECT query.
    {table_schema}

    RESPONSE:
"""
    )
    return PromptTemplate.from_template(prompt)


def build_select_query_with_where() -> PromptTemplate:
    """
    Creates a prompt template for formulating a SELECT query with a WHERE clause based on matching table schema properties and user information.
    The prompt aims to generate a relevant SELECT query that filters data according to user-related criteria, ensuring query validity and relevance.
    Returns:
        PromptTemplate: The constructed prompt template for SELECT query generation with a WHERE clause.
    """
    prefix = get_cql_code_gen_prefix()
    prompt = (
        prefix
        + """\n I will give you a table schema and some info where at least one of the properties is believed to 
        match a column name. I'm also including some example rows from the table to help you get an idea of what kind 
        of data exists in the table. I want you to construct a SELECT statement where the matching property's value 
        is used in a WHERE clause with its corresponding column name. If a property name closely matches the name of 
        a column, consider that a match. If no such matches are found, if a property's value looks like it is very 
        likely to be useful in a where clause, you can use it, but remember that the goal is to find relevant data 
        that matches the user's request. For example, if they ask about a honda, and you have a table named 'cars' 
        with a column named 'make', then you can include WHERE make = 'honda' as a query predicate. However, 
        try not to filter more than necessary. Give me only the complete SELECT statement that you come up with. If 
        execution_counter is greater than 0, then it means the previous attempt to write a query failed, and if such 
        failures have occurred, I'll give you the failure reasons we received so you can try to improve the query 
        based on those results to avoid getting the same error. Never use execution_counter or anything from the 
        failure reasons in the actual query parameters you generate. Additionally, I want you to ensure that the 
        SELECT statement you generate is valid CQL based on the available keys and indexes of the table (to the 
        extent possible based on the information provided) and I want you to assume that ALLOW FILTERING is disabled 
        on the table.

    {TableSchema}
    
    {PropertyInfo}
    
    RESPONSE:
"""
    )
    return PromptTemplate.from_template(prompt)


def build_select_query_with_where_parallelizable(
    table_schema: TableSchema,
) -> PromptTemplate:
    """
    Develops a parallelizable prompt template for generating a SELECT query with a WHERE clause, given a table schema and property information.
    This prompt focuses on constructing a query that filters data based on properties matching user information, tailored to the specific table schema.
    Parameters:
        table_schema (TableSchema): The schema of the table for query generation.
    Returns:
        PromptTemplate: The constructed prompt template for parallelizable SELECT query generation with a WHERE clause.
    """
    prefix = get_cql_code_gen_prefix()
    prompt = (
        prefix
        + f"""\n I will give you a table schema and some info where at least one of the properties is believed to 
        match a column name. I'm also including some example rows from the table to help you get an idea of what kind 
        of data exists in the table. I want you to construct a SELECT statement where the matching property's value 
        is used in a WHERE clause with its corresponding column name. If a property name closely matches the name of 
        a column, consider that a match. If no such matches are found, if a property's value looks like it is very 
        likely to be useful in a where clause, you can use it, but remember that the goal is to find relevant data 
        that matches the user's request. For example, if they ask about a honda, and you have a table named 'cars' 
        with a column named 'make', then you can include WHERE make = 'honda' as a query predicate. However, 
        try not to filter more than necessary. Give me only the complete SELECT statement that you come up with. If 
        execution_counter is greater than 0, then it means the previous attempt to write a query failed, and if such 
        failures have occurred, I'll give you the failure reasons we received so you can try to improve the query 
        based on those results to avoid getting the same error. Never use execution_counter or anything from the 
        failure reasons in the actual query parameters you generate. Additionally, I want you to ensure that the 
        SELECT statement you generate is valid CQL based on the available keys and indexes of the table (to the 
        extent possible based on the information provided) and I want you to assume that ALLOW FILTERING is disabled 
        on the table.

    {table_schema}
    
    PROPERTY INFO:

    {{PropertyInfo}}

    RESPONSE:
"""
    )
    return PromptTemplate.from_template(prompt)


def build_select_query_with_where_v2() -> PromptTemplate:
    """
    Constructs a prompt for generating a SELECT query with a WHERE clause using table schema, example rows, and user info.
    This prompt helps create a CQL query that utilizes user-related property values in WHERE clauses for more relevant data retrieval.
    Returns:
        PromptTemplate: The prompt template for SELECT query generation with WHERE clauses.
    """
    prefix = get_cql_code_gen_prefix()
    prompt = (
        prefix
        + """\n I will give you a table schema and some info where at least one of the properties is believed to 
        match a column name. I'm also including some example rows from the table to help you get an idea of what kind 
        of data exists in the table. I want you to construct a SELECT statement where the matching property's value 
        is used in a WHERE clause with its corresponding column name. If a property name closely matches the name of 
        a column, consider that a match. If no such matches are found, if a property's value looks like it is very 
        likely to be useful in a where clause, you can use it, but remember that the goal is to find relevant data 
        that matches the user's request. For example, if they ask about a honda, and you have a table named 'cars' 
        with a column named 'make', then you can include WHERE make = 'honda' as a query predicate. However, 
        try not to filter more than necessary. Give me only the complete SELECT statement that you come up with. If 
        execution_counter is greater than 0, then it means the previous attempt to write a query failed, and if such 
        failures have occurred, I'll give you the failure reasons we received so you can try to improve the query 
        based on those results to avoid getting the same error. Never use execution_counter or anything from the 
        failure reasons in the actual query parameters you generate. Additionally, I want you to ensure that the 
        SELECT statement you generate is valid CQL based on the available keys and indexes of the table (to the 
        extent possible based on the information provided) and I want you to assume that ALLOW FILTERING is disabled 
        on the table.

    {TableSchema}
    
    {Top3Rows}

    {PropertyInfo}

    RESPONSE:
"""
    )
    return PromptTemplate.from_template(prompt)


def build_select_query_with_where_one_variable() -> PromptTemplate:
    """
    Builds a prompt for generating a SELECT query with a WHERE clause based on a table schema and specific user info.
    This prompt focuses on creating a CQL query where user information is matched with table column names in the WHERE clause.
    Returns:
        PromptTemplate: The prompt template for SELECT query generation with a specific WHERE clause.
    """
    prefix = get_cql_code_gen_prefix()
    prompt = (
        prefix
        + """\n I will give you a table schema and some info where at least one of the properties is believed to match a column name. I'm also including
some example rows from the table to help you get an idea of what kind of data exists in the table. 
I want you to construct a SELECT statement where the matching property's value is used in a WHERE clause with its corresponding column name.
If a property name closely matches the name of a column, consider that a match. If no such matches are found, if a property's value looks like it
is very likely to be useful in a where clause, you can use it, but remember that the goal is to find relevant data that matches the user's request.
For example, if they ask about a honda, and you have a table named 'cars' with a column named 'make', then you can include WHERE make = 'honda' as 
a query predicate. However, try not to filter more than necessary. Give me only the complete SELECT statement that you come up with.
If execution_counter is greater than 0, then it means the previous attempt to write a query failed, and if such failures have occurred, 
I'll give you the failure reasons we received so you can try to improve the query based on those results to avoid getting the same error.
Never use execution_counter or anything from the failure reasons in the actual query parameters you generate.

    {context}

    RESPONSE:
"""
    )
    return PromptTemplate.from_template(prompt)


def determine_failure_type() -> PromptTemplate:
    """
    Creates a prompt to categorize the type of error encountered in a failed query.
    This prompt helps identify if the failure was due to token count exceeded, syntax error, or other reasons.
    Returns:
        PromptTemplate: The prompt template for error categorization.
    """
    prefix = get_helpful_assistant_prefix()
    prompt = (
        prefix
        + """\n I will give you some information about a failed query, and I want you to categorize the issue type.
        If the error was caused by the LLM token count being exceeded, return "TOKEN_COUNT_EXCEEDED".
        If the error was caused by a syntax error with the query itself, return "SYNTAX_ERROR".
        If the error was caused by something else, return "OTHER".
        Don't return anything else. Only return one of those three values.

    {TableExecutionInfo}

    {TableSchema}

    RESPONSE:
"""
    )
    return PromptTemplate.from_template(prompt)


def rewrite_query_based_on_error() -> PromptTemplate:
    """
    Develops a prompt for rewriting a CQL query based on provided error information.
    This prompt assists in correcting and improving a previously failed query by addressing the identified issues.
    Returns:
        PromptTemplate: The prompt template for rewriting a failed query.
    """
    prefix = get_helpful_assistant_prefix()
    prompt = (
        prefix
        + """\n I will give you some information about a failed query, and I want you to try writing the query again. 
This time, I want you to fix the error based on the information provided in the error. Be careful to not deviate from the intent
of the original query. NEVER return anything other than a SELECT statement, and no matter what is in the error info, never select
from a table other than the intended one.

    {TableExecutionInfoWithLastFailure}

    RESPONSE:
"""
    )
    return PromptTemplate.from_template(prompt)


def build_select_query_where_user_property_matches_column_name() -> PromptTemplate:
    """
    Constructs a prompt to determine if any table column name matches user info property names.
    This prompt is used to identify if a table schema is relevant to the user by matching column names with user properties.
    Returns:
        PromptTemplate: The prompt template for matching user properties with table column names.
    """
    prefix = get_helpful_assistant_prefix()
    prompt = (
        prefix
        + """\n I will give you a table schema and some user info, and I want you to determine if any 
    column name matches any of the user info property names. If any such match exists, return a single word response, 
    either "YES" if at least one match is found, or "NO" in any other case.

    {TableSchema}

    {UserInfo}

    RESPONSE:
"""
    )
    template = PromptTemplate.from_template(prompt)
    return template


def trim_to_first_1000_chars(input_str: str) -> str:
    """
    Trims a string to its first 1000 characters.
    This function is useful for reducing lengthy strings to a manageable size.
    Parameters:
        input_str (str): The string to be trimmed.
    Returns:
        str: The trimmed string.
    """
    return input_str[:1000]


def clean_string_v2(s: str) -> str:
    """
    Cleans a given string by removing backticks, trimming whitespace, and excluding lines containing only 'sql', 'json', or 'cql'.
    This function prepares strings for processing by removing unnecessary elements.
    Parameters:
        s (str): The string to be cleaned.
    Returns:
        str: The cleaned string.
    """
    # Removing backticks and trimming whitespace
    s = s.strip("`' ")

    # Splitting the string into lines
    lines = s.split("\n")

    # Removing any lines that contain only 'sql' or 'json'
    lines = [
        line for line in lines if line.strip().lower() not in ["sql", "json", "cql"]
    ]

    # Joining the lines back into a single string
    return "\n".join(lines)


def get_chain_to_determine_if_table_might_be_relevant_to_user() -> PromptTemplate:
    """
    Creates a prompt template to determine if a table might be relevant to a user based on table schema and user information.
    This prompt assesses the relevance of a table to a user by checking for matches between user properties and table column names.
    Returns:
        PromptTemplate: The prompt template for assessing table relevance to a user.
    """
    prefix = get_helpful_assistant_prefix()
    prompt = (
        prefix
        + """\n I will give you a table schema, a few example rows, and some user info, and I want you to determine if any 
    column name matches any of the user info property names. If any such match exists, return a single word response, 
    either "YES" if at least one match is found, or "NO" in any other case.

    {TableSchema}

    {UserInfo}

    RESPONSE:
"""
    )
    return PromptTemplate.from_template(prompt)
