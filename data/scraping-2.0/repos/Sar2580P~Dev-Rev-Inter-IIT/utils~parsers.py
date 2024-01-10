from langchain.output_parsers import StructuredOutputParser, ResponseSchema


critique_response_schemas = [
    ResponseSchema(name="answer", description="Return 1 if user query can be answered by the available tools based on tool description, else return 0."),
    ResponseSchema(name="reason", description="Reason why available tools can/ cannot answer the user query based on tool descriptions.")
]
critique_parser = StructuredOutputParser.from_response_schemas(critique_response_schemas)

#_______________________________________________________________________________________________________________________________________________________________
sub_task_response_schemas = [
    ResponseSchema(name="tool_input", description="The next consecutive sub-task in intermediate steps for the above tool based on above user_query and tool description"),
    ResponseSchema(name="reason", description="Reason why the tool should be chosen as next consecutive tool in intermediate steps based on tool_description and user_query, at max 20 words and atleast 15 words")
]
sub_task_parser = StructuredOutputParser.from_response_schemas(sub_task_response_schemas) 

#_______________________________________________________________________________________________________________________________________________________________
arg_filter_response_schemas = [
    ResponseSchema(name="Arguments", description="The list of filtered arguments whose value is available in the user query"),
]
arg_filter_parser = StructuredOutputParser.from_response_schemas(arg_filter_response_schemas)
