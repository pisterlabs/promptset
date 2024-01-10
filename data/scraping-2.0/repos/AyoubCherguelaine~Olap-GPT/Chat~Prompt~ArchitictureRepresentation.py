from langchain import PromptTemplate


template = """The following text represents a Database, including Tables, Columns, and their relations such as "Primary Keys" and "Foreign Keys." 
    
Please provide a textual Explain for each table:
    |{Representation}|
    """

promptArchitictureRepresentation = PromptTemplate(
        input_variables=["Representation"],
        template=template,
    )
    
