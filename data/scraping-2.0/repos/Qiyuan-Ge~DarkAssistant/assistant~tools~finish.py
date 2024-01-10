from langchain.agents import Tool

finish_tool = Tool(
    name="Final Response",
    func=lambda x: None,
    description='provide a detailed answer to the user, args: <answer in markdown format>',
    )
